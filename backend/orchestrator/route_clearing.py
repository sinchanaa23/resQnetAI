"""F2: AI-powered emergency route clearing.

route prediction -> population / inflow estimation -> dynamic warning corridor ->
channel optimisation (Cell Broadcast vs targeted SMS + voice) -> continuous update.
"""
import itertools
import random
from collections import Counter

from ..sim.geo import haversine, interpolate, slice_polyline
from .security import Timer

LOOKAHEAD_S = 240        # corridor covers the next ~4 minutes of the drive
MIN_CORRIDOR_M = 450
SAMPLE_M = 40
NEAR_ROUTE_KM = 0.15     # people within 150 m of the corridor are "near the route"
LEAD_OK_S = 45           # an alert needs >= 45 s lead time to clear the road
TURNOUT_S = 40           # crew boarding time: the corridor ahead is warned before wheels roll
OFF_ROUTE_COST = 0.004   # optimiser penalty per person alerted who is not near the route
_trip_ids = itertools.count(1)


class EtaMap:
    """Edge-node cache of AI-predicted travel time for every road segment (refreshed every 5 sim-min)."""

    def __init__(self, sim):
        self.sim = sim
        self.hour = None
        self.times = []

    def refresh(self, force=False):
        hour = self.sim.hour
        if not force and self.hour is not None and abs(hour - self.hour) < 5 / 60:
            return
        rng = random.Random(int(hour * 100))
        m, g = self.sim.models, self.sim.graph
        with Timer() as tm:
            self.times = [m.predict_edge_time(i, hour, m.observe_probe(i, hour, rng)) for i in range(len(g.length))]
        self.hour = hour
        self.refresh_ms = tm.ms


class Trip:
    def __init__(self, sim, responder, dest, incident):
        self.sim = sim
        self.id = f"TRIP-{next(_trip_ids):03d}"
        self.responder = responder
        self.incident = incident
        self.dest = dest
        self.pos = responder.pos
        self.d = 0.0
        self.started = sim.t
        self.rolling_at = sim.t + TURNOUT_S
        self.finished = None
        self.alerted = {}       # cell id -> alert record (one alert per cell per trip: no repeats)
        self.corridor = []
        self.corridor_cells = []
        self.passed_cells = set()
        self.coverage_num = 0.0
        self.coverage_den = 0.0
        self.alert_latencies = []
        self.edge_clear = {}    # route edge index -> clearance factor
        self.edge_seen = {}     # route edge index -> lead time (s) when it entered the corridor
        self.speed_kmh = 0.0
        self._next_corr = 0.0
        self.plan()

    # ---- planning -----------------------------------------------------------
    def plan(self):
        sim, g = self.sim, self.sim.graph
        with Timer() as tm:
            sim.eta_map.refresh()
            src = g.nearest_node(self.pos)
            dst = g.nearest_node(self.dest)
            et = sim.eta_map.times
            self.route = g.route(src, dst, lambda e: et[e])
            self.base_route = g.route(src, dst, lambda e: g.length[e])
            self.pred_edge = [et[e] for e, _ in self.route.edges]
            self.pred_total = sum(self.pred_edge)
        hour = sim.hour
        self.base_true = sum(g.true_time(e, hour) for e, _ in self.base_route.edges)
        self.ai_route_true = sum(g.true_time(e, hour) for e, _ in self.route.edges)
        self.plan_ms = tm.ms
        sim.latency.record("route plan", tm.ms, sim.world.avg_load())
        roads = [n for n, _ in Counter(g.name[e] for e, _ in self.route.edges if g.name[e]).most_common(3)]
        self.roads = roads
        flooded = sum(1 for e, _ in self.base_route.edges if g.flood_factor(e) < 1)
        sim.decide("F2", f"Route for {self.responder.id}: {self.route.length / 1000:.1f} km, ETA {self.pred_total / 60:.1f} min",
                   [f"Via {', '.join(roads) or 'local roads'}",
                    f"GBM segment-ETA model over {len(self.route.edges)} segments (live probe data, flood map)",
                    f"Shortest path would be {self.base_route.length / 1000:.1f} km but crosses {flooded} waterlogged "
                    f"segments; predicted {self.base_true / 60:.1f} min" if flooded else
                    f"Shortest path {self.base_route.length / 1000:.1f} km predicted {self.base_true / 60:.1f} min",
                    f"Edge decision in {tm.ms:.0f} ms"],
                   ["Shortest-distance route", "Blanket broadcast to whole area"], tm.ms)

    # ---- movement ------------------------------------------------------------
    def tick(self, dt):
        if self.finished is not None:
            return
        sim, g, r = self.sim, self.sim.graph, self.route
        k = r.edge_at(self.d)
        ei = r.edges[k][0]
        clear = self.edge_clear.get(k, 0.0)
        if sim.t >= self.rolling_at:
            self.speed_kmh = g.true_speed_kmh(ei, sim.hour, clear)
            self.d += self.speed_kmh / 3.6 * min(dt, sim.t - self.rolling_at)
        if self.d >= r.length:
            self.d = r.length
            self.finished = sim.t
        self.pos = interpolate(r.points, r.cum, self.d)
        self.responder.pos = self.pos
        if sim.t >= self._next_corr or self.finished is not None:
            self._next_corr = sim.t + 1.0
            self.update_corridor()

    def eta_remaining(self):
        r = self.route
        k = r.edge_at(self.d)
        frac = 1 - (self.d - r.edge_start[k]) / max(1.0, self.sim.graph.length[r.edges[k][0]])
        return max(0.0, self.rolling_at - self.sim.t) + self.pred_edge[k] * frac + sum(self.pred_edge[k + 1:])

    # ---- corridor + alerts ------------------------------------------------------
    def update_corridor(self):
        sim, g, r, w = self.sim, self.sim.graph, self.route, self.sim.world
        with Timer() as tm:
            # Walk ahead along the route accumulating predicted time until the lookahead horizon.
            samples, t_ahead, d = [], max(0.0, self.rolling_at - sim.t), self.d
            k = r.edge_at(d)
            while d < r.length:
                ei = r.edges[k][0]
                pace = self.pred_edge[k] / max(1.0, g.length[ei])
                if t_ahead > LOOKAHEAD_S and d - self.d > MIN_CORRIDOR_M:
                    break
                samples.append((d, t_ahead, k))
                d += SAMPLE_M
                t_ahead += SAMPLE_M * pace
                while k + 1 < len(r.edges) and r.edge_start[k + 1] <= d:
                    k += 1
            end = d
            self.corridor = slice_polyline(r.points, r.cum, self.d, min(end, r.length)) if self.finished is None else []
            groups = {}
            for dd, ta, kk in samples:
                p = interpolate(r.points, r.cum, dd)
                cell = w.home_cell(p)
                gr = groups.setdefault(cell.id, {"cell": cell, "eta": ta, "len": 0.0, "edge_eta": {},
                                                 "names": Counter()})
                gr["eta"] = min(gr["eta"], ta)
                gr["len"] += SAMPLE_M
                gr["edge_eta"][kk] = min(gr["edge_eta"].get(kk, 1e9), ta)
                nm = g.name[r.edges[kk][0]]
                if nm:
                    gr["names"][nm] += 1
            new = [gr for cid, gr in groups.items() if cid not in self.alerted]
            decisions = [self.choose_channel(gr) for gr in new]
        if new:
            sim.latency.record("corridor update", tm.ms, w.avg_load())
        for gr, choice in zip(new, decisions):
            self.send_alert(gr, choice, tm.ms)
        # Each route segment is scored once, when it first enters the corridor: its lead time is how long
        # before the vehicle arrives the people on it were warned. Already-alerted cells extend their
        # road geofence operator-side (no user is alerted twice).
        cur = r.edge_at(self.d)
        for cid, gr in groups.items():
            rec = self.alerted[cid]
            for kk, lead in gr["edge_eta"].items():
                if kk in self.edge_seen:
                    continue
                self.edge_seen[kk] = lead
                weight = rec["inflow_per_m"] * g.length[r.edges[kk][0]]
                ok = rec["delivered"] and lead >= LEAD_OK_S
                self.coverage_den += weight
                self.coverage_num += weight * (rec["reliability"] if ok else 0)
                if rec["delivered"] and kk >= cur:
                    f = 1.0 if lead >= LEAD_OK_S else max(0.0, lead / LEAD_OK_S)
                    self.edge_clear[kk] = f * rec["reliability"]
        # Cells the vehicle has left stop receiving alerts.
        current = set(groups)
        for cid in list(self.alerted):
            rec = self.alerted[cid]
            if cid not in current and rec["state"] == "active":
                rec["state"] = "passed"
                self.passed_cells.add(cid)
        self.corridor_cells = sorted(current)

    def choose_channel(self, gr):
        """Greedy per-cell optimiser: maximise expected people cleared minus off-route alert cost."""
        sim, cell = self.sim, gr["cell"]
        corr_km = gr["len"] / 1000
        near = cell.density * corr_km * 2 * NEAR_ROUTE_KM
        hist = cell.inflow_hist[-6:] if len(cell.inflow_hist) >= 6 else [10] * 6
        inflow = sim.models.predict_inflow(hist, sim.hour) * min(1.0, corr_km / 0.6)
        slice_on = sim.calls.slice_active
        sms_rel = 0.97 if slice_on else (0.78 if cell.load > 0.85 else 0.95)
        options = []
        if cell.status != "down":
            cb_alerted = cell.devices
            cb_near = min(near, cb_alerted)
            options.append(("Cell Broadcast", 0.99 * inflow - OFF_ROUTE_COST * (cb_alerted - cb_near),
                            cb_alerted, cb_near, 0.99))
        sms_alerted = int(near * 1.16)  # operator-side road geofence also catches some parked / indoor users
        options.append(("Targeted SMS + voice", sms_rel * inflow - OFF_ROUTE_COST * (sms_alerted - near),
                        sms_alerted, near, sms_rel))
        best = max(options, key=lambda o: o[1])
        return {"channel": best[0], "alerted": int(best[2]), "near": int(best[3]), "reliability": best[4],
                "inflow": inflow, "options": options}

    def send_alert(self, gr, choice, compute_ms):
        sim, cell = self.sim, gr["cell"]
        road = gr["names"].most_common(1)[0][0] if gr["names"] else "this road"
        mins = max(1, round(gr["eta"] / 60))
        text = (f"Emergency alert: An ambulance is expected on {road} in approximately {mins} minute"
                f"{'s' if mins > 1 else ''}. Please avoid entering this road and clear the route for the emergency vehicle.")
        res = sim.conn.deliver("Route alert", cell.pos, f"cell {cell.id}")
        lat = compute_ms / 1000 + 0.003 + res["latency_s"]
        self.alert_latencies.append(lat)
        rec = {"cell": cell.id, "t": sim.t, "channel": choice["channel"], "alerted": choice["alerted"],
               "near": choice["near"], "eta_at_alert": gr["eta"], "road": road,
               "inflow_per_m": choice["inflow"] / max(gr["len"], 600),
               "delivered": res["ok"], "path": res["path"], "reliability": choice["reliability"],
               "inflow": choice["inflow"], "state": "active", "text": text}
        self.alerted[cell.id] = rec
        opt = {o[0]: o for o in choice["options"]}
        reasons = [f"{road}: vehicle ETA {gr['eta']:.0f}s, corridor {gr['len']:.0f} m inside {cell.id}",
                   f"Predicted inflow {choice['inflow']:.0f} people (AR forecaster, aggregated CAMARA density)",
                   f"{choice['channel']}: {choice['alerted']:,} recipients, ~{choice['near']:,} near route"]
        alt = [f"{name}: {o[2]:,} recipients (score {o[1]:.1f})" for name, o in opt.items() if name != choice["channel"]]
        sim.decide("F2", f"Alert {cell.id} via {choice['channel']}", reasons, alt, compute_ms)
        if len(self.alerted) <= 2 or len(self.alerted) % 2 == 0:
            sim.emit("F2", f"Route alert → {cell.id} ({choice['channel']})",
                     f"{choice['alerted']:,} phones · “{text[:92]}…”", "info")
        if len(self.alerted) == 1:
            sim.sample_alert = text

    # ---- reporting ------------------------------------------------------------
    def blanket_precision(self):
        """Baseline: broadcast to every cell within 1.2 km of the route."""
        w, r = self.sim.world, self.route
        pts = r.points[::max(1, len(r.points) // 60)]
        cells = {c.id: c for c in w.cells if not c.temp and any(haversine(p, c.pos) < 1200 for p in pts)}
        total = sum(c.devices for c in cells.values())
        near = sum(rec["near"] for rec in self.alerted.values())
        return near / total if total else 0, total

    def summary(self):
        alerted = sum(a["alerted"] for a in self.alerted.values())
        near = sum(min(a["near"], a["alerted"]) for a in self.alerted.values())
        bp, btotal = self.blanket_precision()
        elapsed = max(0.0, (self.finished or self.sim.t) - self.rolling_at)  # drive time, excl. turnout
        return {
            "id": self.id, "vehicle": self.responder.id, "finished": self.finished is not None,
            "elapsed_s": elapsed, "eta_s": self.eta_remaining() if self.finished is None else 0,
            "route_km": self.route.length / 1000, "baseline_route_km": self.base_route.length / 1000,
            "baseline_time_s": self.base_true, "ai_route_time_s": self.ai_route_true, "pred_total_s": self.pred_total,
            "cells_alerted": len(self.alerted), "people_alerted": alerted,
            "coverage": self.coverage_num / self.coverage_den if self.coverage_den else None,
            "precision": near / alerted if alerted else None, "blanket_precision": bp, "blanket_alerted": btotal,
            "alert_latency_s": sum(self.alert_latencies) / len(self.alert_latencies) if self.alert_latencies else None,
            "speed_kmh": self.speed_kmh, "progress": self.d / self.route.length,
            "roads": self.roads,
        }
