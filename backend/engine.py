"""resQnet simulation engine + AI Emergency Orchestrator (runs on the 'MEC node')."""
import itertools
import random
import re

from .camara.client import CamaraClient
from .camara.mock import SIM_EPOCH, MockNetwork
from .ml.training import ModelSuite
from .orchestrator.calls import CallPrioritizer
from .orchestrator.connectivity import Connectivity
from .orchestrator.matching import MedicRequest
from .orchestrator.route_clearing import EtaMap, Trip
from .orchestrator.security import AuditLog, IdentityService, LatencyTracker, Timer
from .sim.geo import circle_polygon, haversine
from .sim.roads import RoadGraph
from .sim.world import FLOOD_CENTER, FLOOD_RADIUS, HOSPITAL, INCIDENT_POS, World

START_HOUR = 19 + 5 / 60
COMMS_EVERY_S = 8


class Simulation:
    def __init__(self, graph=None, models=None):
        self.graph = graph or RoadGraph()
        self.models = models or ModelSuite(self.graph)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        self.t = 0.0
        self.speed = 2.0
        self.paused = False
        self.seq = itertools.count(1)
        self.events, self.decisions, self.api_calls = [], [], []
        self.graph.flood = (FLOOD_CENTER, FLOOD_RADIUS)
        self.scenario_center = INCIDENT_POS
        self.world = World(self.graph, self.models)
        self.network = MockNetwork(self.world, lambda: self.t)
        self.camara = CamaraClient(self.network, self._log_api)
        self.audit = AuditLog()
        self.latency = LatencyTracker()
        self.identity = IdentityService(self)
        self.conn = Connectivity(self)
        self.calls = CallPrioritizer(self)
        self.eta_map = EtaMap(self)
        self.eta_map.refresh(force=True)
        self.incidents = []
        self.trips = []
        self.medics = []
        self.sample_alert = None
        self.director = None
        self.highlight = None
        self._inc_ids = itertools.count(2041)
        self._next_comms = COMMS_EVERY_S
        self._rng = random.Random(1)
        self.background_calls = False
        self.emit("SYS", "resQnet edge orchestrator online",
                  f"MEC node · {len(self.graph.nodes):,} road nodes · {len([c for c in self.world.cells if not c.temp])} "
                  f"cell sites · models trained in {self.models.train_seconds}s", "info")

    @property
    def hour(self):
        return START_HOUR + self.t / 3600

    def clock(self, t=None):
        t = self.t if t is None else t
        h = START_HOUR + t / 3600
        return f"{int(h):02d}:{int((h % 1) * 60):02d}:{int((h * 3600) % 60):02d}"

    # --------------------------------------------------------------- logging
    def emit(self, feature, title, detail="", level="info"):
        self.events.append({"id": next(self.seq), "t": round(self.t, 1), "clock": self.clock(), "f": feature,
                            "title": title, "detail": detail, "level": level})
        self.events = self.events[-300:]

    def decide(self, feature, title, reasons, alternatives, compute_ms):
        self.decisions.append({"id": next(self.seq), "t": round(self.t, 1), "clock": self.clock(), "f": feature,
                               "title": title, "reasons": reasons, "alternatives": alternatives,
                               "ms": round(compute_ms, 2)})
        self.decisions = self.decisions[-120:]

    def _log_api(self, entry):
        entry.update(id=next(self.seq), clock=self.clock())
        self.api_calls.append(entry)
        self.api_calls = self.api_calls[-80:]

    # --------------------------------------------------------------- actions
    def create_incident(self, pos, kind="flood", people=6, injuries=2, critical=False, note="", source="112 call"):
        with Timer() as tm:
            in_zone = bool(self.calls.in_zone(pos)) or haversine(pos, FLOOD_CENTER) < FLOOD_RADIUS
            sev = self.models.severity(kind, people, injuries, critical, in_zone)
        self.latency.record("severity scoring", tm.ms, self.world.avg_load())
        inc = {"id": f"INC-{next(self._inc_ids)}", "pos": tuple(pos), "type": kind, "people": people,
               "injuries": injuries, "critical": critical, "severity": sev, "note": note, "t": self.t,
               "status": "open", "source": source}
        self.incidents.append(inc)
        self.audit.append(self.t, source, "incident_created", inc["id"], f"{kind} sev {sev}")
        self.decide("ALL", f"{inc['id']} severity {sev}/5",
                    [f"Type: {kind}; {people} people affected; {injuries} injured" + ("; life-threatening" if critical else ""),
                     "Inside flood-affected area" if in_zone else "Outside declared zones",
                     "Rule + gradient-boosting hybrid (safety floor: critical ⇒ ≥4)"], [], tm.ms)
        self.emit("ALL", f"Incident {inc['id']} created — severity {sev}/5",
                  f"{kind.title()} · {people} people · {injuries} injured{' · ' + note if note else ''}", "warn")
        return inc

    def incident(self, iid):
        return next((i for i in self.incidents if i["id"].lower() == iid.lower()), None)

    def sms(self, msisdn, text, network_msisdn="__same__"):
        """Responder SMS command channel. network_msisdn is the number the network actually authenticated
        for the sending SIM (differs from msisdn when spoofed)."""
        if network_msisdn == "__same__":
            network_msisdn = msisdn
        m = re.match(r"\s*(CLEAR(?:\s+ROUTE)?|MEDIC)\s+(INC-\d+)", text.upper())
        if not m:
            return {"ok": False, "reply": "resQnet: unknown command. Use CLEAR ROUTE INC-#### or MEDIC INC-####"}
        cmd, iid = m.group(1), m.group(2)
        inc = self.incident(iid)
        if not inc:
            return {"ok": False, "reply": f"resQnet: {iid} not found"}
        action = "route_clearance" if cmd.startswith("CLEAR") else "medic_request"
        with Timer() as tm:
            allowed, r, checks = self.identity.authorize(msisdn, network_msisdn, action)
        self.latency.record("identity check", tm.ms, self.world.avg_load())
        who = r.id if r else msisdn
        self.decide("F1", f"{'ALLOWED' if allowed else 'BLOCKED'}: {action.replace('_', ' ')} from {who}",
                    [f"{'✔' if ok else '✘'} {name} — {detail}" for name, ok, detail in checks], [], tm.ms)
        if not allowed:
            failed = [n for n, ok, _ in checks if not ok]
            self.emit("F1", f"Blocked {action.replace('_', ' ')} request from {who}", "Failed: " + ", ".join(failed), "bad")
            return {"ok": False, "checks": checks, "reply": "resQnet: request rejected — identity checks failed. "
                                                             "Incident logged for review."}
        self.emit("F1", f"Verified {r.id} — {action.replace('_', ' ')} authorised",
                  "Number verified · no recent SIM swap · role permitted", "good")
        if action == "route_clearance":
            trip = self.start_trip(r, inc)
            return {"ok": True, "checks": checks, "reply": f"resQnet: route clearance ACTIVE for {inc['id']}. "
                                                           f"ETA {trip.pred_total / 60:.0f} min via {', '.join(trip.roads[:2])}."}
        req = self.request_medic(inc, r.id)
        top = req.ranked[0]["v"].name if req.ranked else "none"
        return {"ok": True, "checks": checks, "reply": f"resQnet: medic search started for {inc['id']}. Calling {top}."}

    def start_trip(self, responder, inc):
        for t in self.trips:
            if t.responder is responder and t.finished is None:
                return t
        trip = Trip(self, responder, inc["pos"], inc)
        self.trips.append(trip)
        inc["status"] = "ambulance en route"
        self.emit("F2", f"{responder.id} dispatched — route clearance active",
                  f"{trip.route.length / 1000:.1f} km · predicted ETA {trip.pred_total / 60:.1f} min · "
                  f"shortest-path baseline {trip.base_true / 60:.1f} min", "info")
        return trip

    def request_medic(self, inc, by="dashboard"):
        for m in self.medics:
            if m.incident is inc and not m.assigned:
                return m
        req = MedicRequest(self, inc, by)
        self.medics.append(req)
        return req

    def ivr(self, vid, digit):
        for m in self.medics:
            if vid in m.ringing:
                return m.respond(vid, digit)
        return False

    def fail_cell(self, cid):
        c = self.world.cell_by_id.get(cid)
        if c:
            self.conn.fail_cell(c)

    def restore_cell(self, cid):
        c = self.world.cell_by_id.get(cid)
        if c:
            self.conn.restore_cell(c)

    def override(self, action, arg=None, by="Control room officer"):
        """Human override of automated decisions (always audited)."""
        msg = None
        if action == "skip_candidate":
            for m in self.medics:
                if arg in m.ringing:
                    m.respond(arg, "2", "override")
                    msg = f"Skipped candidate {arg}"
        elif action == "stop_alerts":
            for t in self.trips:
                if t.finished is None:
                    t.finished = self.t
                    t.corridor = []
                    msg = f"Route alerts stopped for {t.responder.id}"
        elif action == "lift_zone":
            self.calls.lift_zone()
            msg = "Disaster zone lifted"
        elif action == "revoke":
            r = self.world.responder(arg)
            if r:
                r.revoked = True
                msg = f"Credentials of {r.id} revoked"
        elif action == "reinstate":
            r = self.world.responder(arg)
            if r:
                r.revoked = False
                msg = f"Credentials of {r.id} reinstated"
        elif action == "cloud_link":
            self.latency.cloud_link = not self.latency.cloud_link
            msg = f"Cloud link {'restored' if self.latency.cloud_link else 'CUT — edge keeps deciding locally'}"
        if msg:
            self.audit.append(self.t, by, "override:" + action, "APPLIED", msg)
            self.emit("SYS", f"Human override — {msg}", by, "warn")
        return msg

    # --------------------------------------------------------------- loop
    def step(self, dt_real):
        if self.director:
            self.director.tick(dt_real)
        if self.paused:
            return
        dt = min(4.0, dt_real * self.speed)
        while dt > 1e-6:
            h = min(0.5, dt)
            self._substep(h)
            dt -= h

    def _substep(self, dt):
        self.t += dt
        self.eta_map.refresh()
        for trip in self.trips:
            was = trip.finished
            trip.tick(dt)
            if was is None and trip.finished is not None and trip.d >= trip.route.length:
                trip.incident["status"] = "ambulance on scene"
                s = trip.summary()
                self.emit("F2", f"{trip.responder.id} arrived at {trip.incident['id']}",
                          f"{s['elapsed_s'] / 60:.1f} min vs {s['baseline_time_s'] / 60:.1f} min baseline", "good")
        for m in self.medics:
            m.tick(dt)
        if self.background_calls or self.calls.zone:
            self.calls.tick(dt)
        self.conn.tick()
        # Cell load random walk around the evening-peak level.
        for c in self.world.cells:
            if c.status == "up" and not c.temp:
                c.load = min(0.99, max(0.6, c.load + self._rng.gauss(0, 0.004) + 0.01 * (c.base_load - c.load)))
        # Inflow history grows every 5 sim-minutes (aggregated counts).
        if int((self.t - dt) / 300) != int(self.t / 300):
            for c in self.world.cells:
                if c.inflow_hist:
                    c.inflow_hist.append(c.inflow_hist[-288] if len(c.inflow_hist) > 288 else c.inflow_hist[-1])
        if self.t >= self._next_comms:
            self._next_comms = self.t + COMMS_EVERY_S
            self._periodic_comms()

    def _periodic_comms(self):
        """Critical status traffic to active responders goes through the connectivity orchestrator."""
        if not self.incidents:
            return
        for trip in self.trips:
            if trip.finished is None:
                self.conn.deliver("Incident update", trip.pos, trip.responder.id)
        for m in self.medics:
            if m.assigned and not m.arrived:
                self.conn.deliver("ETA update", m.assigned.pos, m.assigned.name)
        r4 = self.world.responder("RES-04")
        self.conn.deliver("Team comms", r4.pos, "SDRF team R-4", caps={"attenuation_m": 700, "d2d": True})
        # Citizens inside the flood area receive safety updates.
        self.conn.deliver("Safety update", INCIDENT_POS, "citizens near incident")

    # --------------------------------------------------------------- views
    def static_info(self):
        z = self.models.report
        return {
            "bbox": self.graph.bbox, "flood": {"center": FLOOD_CENTER, "radius": FLOOD_RADIUS},
            "hospital": {"name": HOSPITAL[0], "pos": HOSPITAL[1]},
            "model_report": z, "train_seconds": self.models.train_seconds, "epoch": SIM_EPOCH.isoformat(),
            "camara_mode": self.camara.mode,
        }

    def snapshot(self):
        w = self.world
        trips = []
        for t in self.trips:
            s = t.summary()
            s.update(pos=t.pos, corridor=t.corridor, route=t.route.points[::2] + [t.route.points[-1]],
                     baseline_route=t.base_route.points[::3] + [t.base_route.points[-1]], d=t.d,
                     alerted=list(t.alerted.values()),
                     corridor_cells=t.corridor_cells)
            trips.append(s)
        medics = []
        for m in self.medics:
            s = m.summary()
            s["ranked"] = [{"id": c["v"].id, "name": c["v"].name, "spec": c["v"].spec, "dist": round(c["dist"], 2),
                            "p": round(c["p"], 2), "eta": round(c["eta"], 1), "score": round(c["score"], 3)}
                           for c in m.ranked]
            s["route"] = m.route.points[::2] if m.route and not m.arrived else None
            medics.append(s)
        ringing = None
        for m in self.medics:
            for vid in m.ringing:
                v = w.volunteer(vid)
                ringing = {"vid": vid, "name": v.name, "text": v.ivr_text, "since": round(self.t - m.ringing[vid], 1),
                           "scripted": vid in m.script}
        return {
            "t": round(self.t, 2), "clock": self.clock(), "speed": self.speed, "paused": self.paused,
            "load": round(w.avg_load(), 3),
            "cells": [{"id": c.id, "pos": c.pos, "r": c.radius, "load": round(c.load, 2), "status": c.status,
                       "temp": c.temp, "devices": c.devices} for c in w.cells],
            "incidents": self.incidents,
            "zone": dict(self.calls.zone, poly=circle_polygon(self.calls.zone["center"], self.calls.zone["radius"]),
                         buffer_poly=circle_polygon(self.calls.zone["center"],
                                                    self.calls.zone["radius"] + self.calls.zone["buffer"]))
            if self.calls.zone else None,
            "trips": trips, "medics": medics, "ringing": ringing,
            "volunteers": [{"id": v.id, "name": v.name, "role": v.role, "spec": v.spec, "pos": v.pos,
                            "state": v.state, "available": v.available, "assigned": v.assigned} for v in w.volunteers],
            "responders": [{"id": r.id, "name": r.name, "role": r.role, "msisdn": r.msisdn, "pos": r.pos,
                            "revoked": r.revoked, "sim_swap_min": r.sim_swap_minutes_ago} for r in w.responders],
            "links": [{"from": l["from"], "via": l["via"], "path": l["path"], "label": l["label"]} for l in self.conn.links],
            "calls": self.calls.summary(), "call_feed": self.calls.feed[-14:],
            "conn": self.conn.summary(), "latency": self.latency.summary(),
            "events": self.events[-60:], "decisions": self.decisions[-30:], "api_calls": self.api_calls[-25:],
            "audit": {"count": len(self.audit.entries), "recent": self.audit.entries[-12:], **self.audit.verify()},
            "director": self.director.state() if self.director else None,
            "sample_alert": self.sample_alert,
        }

    def results(self):
        trip = self.trips[0].summary() if self.trips else None
        med = self.medics[0].summary() if self.medics else None
        conn, calls, lat = self.conn.summary(), self.calls.summary(), self.latency.summary()
        rep = self.models.report

        def row(name, measure, base, ours, target, passed, unit=""):
            return {"metric": name, "measures": measure, "baseline": base, "resqnet": ours, "target": target,
                    "pass": passed, "unit": unit}

        rows = []
        if trip:
            red = 1 - trip["elapsed_s"] / trip["baseline_time_s"] if trip["finished"] else None
            rows.append(row("Ambulance travel time", "No-resQnet (shortest path, no alerts) vs resQnet",
                            f"{trip['baseline_time_s'] / 60:.1f} min",
                            f"{trip['elapsed_s'] / 60:.1f} min" + (f" (−{red:.0%})" if red is not None else " (en route)"),
                            "≥ 15% shorter", red is not None and red >= 0.15))
            if trip["alert_latency_s"] is not None:
                rows.append(row("Alert-delivery latency", "Request → alert delivered (edge)", "—",
                                f"{trip['alert_latency_s']:.2f} s", "< 2 s", trip["alert_latency_s"] < 2))
            if trip["coverage"] is not None:
                rows.append(row("Route-clearance coverage", "Predicted entrants alerted in time", "—",
                                f"{trip['coverage']:.0%}", "≥ 85%", trip["coverage"] >= 0.85))
            if trip["precision"] is not None:
                rows.append(row("Alert precision", "Alerted people actually near route",
                                f"{trip['blanket_precision']:.0%} (blanket broadcast)", f"{trip['precision']:.0%}",
                                "≥ 70%", trip["precision"] >= 0.7))
        if med and med["selection_s"] is not None:
            rows.append(row("Responder selection time", "Request → first acceptance", "—",
                            f"{med['selection_s']:.0f} s", "< 60 s", med["selection_s"] < 60))
            rows.append(row("Escalation rounds", "Candidates contacted before acceptance",
                            f"{rep['ranking']['baseline']:.2f} avg (nearest-first)",
                            f"{med['contacted']} (model avg {rep['ranking']['ai']:.2f})", "≤ 2 on average",
                            rep["ranking"]["ai"] <= 2))
        if calls["recall"] is not None:
            rows.append(row("Emergency-call recall", "Emergency calls correctly prioritised (live)",
                            f"{rep['classifier']['baseline']:.0%} keyword (held-out)",
                            f"{calls['recall']:.0%} live · {rep['classifier']['ai']:.0%} held-out", "≥ 95%",
                            calls["recall"] >= 0.95))
        if conn["load_sent"]:
            rows.append(row("Emergency-message delivery", "Critical messages delivered at ~90% load (working cells)",
                            f"{conn['load_baseline']:.1%}", f"{conn['load_success']:.1%}", "≥ 98%",
                            conn["load_success"] >= 0.98))
        if conn["failover_success"] is not None:
            rows.append(row("Failover success rate", "Delivered after tower failure",
                            f"{conn['failover_baseline']:.0%}", f"{conn['failover_success']:.0%}", "≥ 95%",
                            conn["failover_success"] >= 0.95))
        if lat["n"] and lat["ratio"]:
            rows.append(row("Edge vs cloud latency", "Decision time on MEC vs cloud-only",
                            f"{lat['cloud_ms']:.0f} ms (cloud)", f"{lat['edge_ms']:.0f} ms (edge) · {lat['ratio']:.0f}× faster",
                            "Edge ≥ 3× faster", lat["ratio"] >= 3))
        if calls["connected"] is not None:
            rows.append(row("Emergency calls connected", "Call set-up success inside zone (report only)",
                            f"{calls['connected_base']:.0%}", f"{calls['connected']:.0%}", "report only", None))
        return {"rows": rows, "clock": self.clock(), "sim_seconds": round(self.t),
                "note": "All values come from this simulated run (synthetic data); targets are prototype goals, "
                        "not field claims."}
