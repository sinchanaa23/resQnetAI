"""F5: resilient connectivity — pick the most reliable supported path for every critical message.

Fallback options (used only where infrastructure and device support exist):
cellular -> neighbouring cell -> temporary emergency cell (COW) -> satellite D2D -> mesh / D2D relay.
"""
import random

from ..ml.training import PATH_LATENCY, path_success_prob
from ..sim.geo import haversine, offset
from ..sim.world import Cell
from .security import Timer

PATH_LABEL = {"cellular": "Home cell", "neighbour_cell": "Neighbouring cell", "temp_cell": "Temporary cell (COW)",
              "satellite": "Satellite D2D", "d2d_relay": "Mesh / D2D relay"}
COW_DEPLOY_SECONDS = 75


class Connectivity:
    def __init__(self, sim, seed=9):
        self.sim = sim
        self.rng = random.Random(seed)
        self.cows = []  # {"cell": Cell, "live_at": t}
        self.links = []  # recent non-default delivery paths for the map
        self.stats = {"sent": 0, "delivered": 0, "first_try": 0, "base_delivered": 0,
                      "failover_sent": 0, "failover_delivered": 0, "failover_base": 0,
                      "by_path": {k: 0 for k in PATH_LATENCY}, "retries": 0,
                      "load_sent": 0, "load_delivered": 0, "load_base": 0}
        self._last_reason = {}

    # ---- context ----------------------------------------------------------
    def context(self, pos, caps=None):
        caps = caps or {}
        w = self.sim.world
        home = w.home_cell(pos)
        if home.status == "down":
            state = "down"
        else:
            state = "congested" if home.load > 0.85 else "up"
        nb, nb_d = w.neighbour_cell(pos, home if home.status == "down" else None, caps.get("attenuation_m", 0))
        if home.status != "down":
            nb = None  # neighbour only matters when the home cell is gone
        cow = self.live_cow_near(pos)
        ctx = (state, bool(self.sim.calls.slice_active), nb is not None, cow is not None,
               bool(caps.get("satellite")), caps.get("d2d", True))
        return ctx, home, nb, cow

    def live_cow_near(self, pos):
        for c in self.cows:
            if self.sim.t >= c["live_at"] and haversine(pos, c["cell"].pos) <= c["cell"].radius:
                return c["cell"]
        return None

    # ---- delivery ---------------------------------------------------------
    def deliver(self, kind, pos, label, caps=None, log=False):
        """Deliver one critical message. Returns dict(ok, path, attempts, latency_s)."""
        sim = self.sim
        with Timer() as tm:
            ctx, home, nb, cow = self.context(pos, caps)
            ranked = sim.models.choose_paths(ctx)
        sim.latency.record("path selection", tm.ms, sim.world.avg_load())
        ok, used, latency, attempts = False, None, 0.0, 0
        for arm in (ranked + ranked)[:3]:  # best paths first; retransmit if fewer than 3 options
            attempts += 1
            latency += PATH_LATENCY[arm]
            if self.rng.random() < path_success_prob(ctx, arm):
                ok, used = True, arm
                break
        # Counterfactual "no resQnet": only the home cell, no priority slice, no fallback.
        base_ctx = (ctx[0], False) + ctx[2:]
        base_ok = ctx[0] != "down" and self.rng.random() < path_success_prob(base_ctx, "cellular")

        s = self.stats
        s["sent"] += 1
        s["delivered"] += ok
        s["base_delivered"] += base_ok
        s["first_try"] += ok and attempts == 1
        s["retries"] += max(0, attempts - 1)
        if used:
            s["by_path"][used] += 1
        if ctx[0] != "down":
            s["load_sent"] += 1
            s["load_delivered"] += ok
            s["load_base"] += base_ok
        if ctx[0] == "down":
            s["failover_sent"] += 1
            s["failover_delivered"] += ok
            s["failover_base"] += base_ok

        if used and used != "cellular":
            via = {"neighbour_cell": nb.pos if nb else None, "temp_cell": cow.pos if cow else None}.get(used)
            if used == "d2d_relay":
                relay_cell, _ = sim.world.neighbour_cell(pos, home, 0)
                hop = offset(pos, 260, -180)
                via = [hop, relay_cell.pos] if relay_cell else [hop]
            self.links.append({"t": sim.t, "from": pos, "via": via, "path": used, "label": label})
            if log or self._last_reason.get(label) != used:
                self._last_reason[label] = used
                alts = [PATH_LABEL[a] for a in ranked if a != used]
                state = "DOWN" if ctx[0] == "down" else f"{ctx[0]} ({home.load:.0%} load)"
                sim.decide("F5", f"Path for {label}: {PATH_LABEL[used]}",
                           [f"Home cell {home.id} is {state}",
                            f"Bandit success estimate {sim.models.bandit.mean(ctx, used):.0%} "
                            f"for {PATH_LABEL[used]} in this context",
                            f"Delivered in {latency:.1f}s after {attempts} attempt(s)"],
                           alts, tm.ms)
                sim.emit("F5", f"{kind} → {label} rerouted via {PATH_LABEL[used]}",
                         f"home cell {home.id} {state}; delivered in {latency:.1f}s", "warn")
        elif not ok:
            sim.emit("F5", f"{kind} to {label} failed on all paths", "will retry next cycle", "bad")
        self.links = [l for l in self.links if sim.t - l["t"] < 20]
        return {"ok": ok, "path": used, "attempts": attempts, "latency_s": latency, "base_ok": base_ok}

    # ---- tower failure & temporary cells -----------------------------------
    def fail_cell(self, cell):
        sim = self.sim
        if cell.status == "down":
            return
        cell.status = "down"
        cell.load = 0
        sim.audit.append(sim.t, "network", "cell_failure", "DETECTED", cell.id)
        sim.emit("F5", f"Cell site {cell.id} DOWN (flooded power room)",
                 f"{cell.devices:,} devices lost their home cell — failover engaged", "bad")
        # Neighbours absorb traffic.
        for c in sim.world.cells:
            if c is not cell and not c.temp and c.status == "up" and haversine(c.pos, cell.pos) < 1100:
                c.load = min(0.99, c.load + 0.05)
        # Dispatch a cell-on-wheels to the edge of the dry area.
        stage = offset(cell.pos, 330, -250)
        cow = Cell(f"COW-{len(self.cows) + 1:02d}", stage, cell.density, self.rng)
        cow.temp, cow.radius, cow.load = True, 520, 0.35
        self.cows.append({"cell": cow, "live_at": sim.t + COW_DEPLOY_SECONDS, "for": cell.id, "announced": False})
        sim.world.cells.append(cow)
        cow.status = "deploying"
        sim.decide("F5", f"Dispatch temporary cell {cow.id} for {cell.id}",
                   ["Site outage covers part of the incident zone and ambulance corridor",
                    f"Nearest staging point on dry ground ~{int(haversine(stage, cell.pos))} m from failed site",
                    f"Estimated on-air in {COW_DEPLOY_SECONDS}s; neighbours + D2D bridge the gap"],
                   ["Wait for site repair (hours)", "Satellite only (few capable devices)"], 0.4)
        sim.emit("F5", f"Temporary cell {cow.id} dispatched", f"on air in ~{COW_DEPLOY_SECONDS}s", "info")

    def restore_cell(self, cell):
        cell.status = "up"
        cell.load = cell.base_load
        self.sim.emit("F5", f"Cell site {cell.id} restored", "", "good")

    def tick(self):
        for c in self.cows:
            if not c["announced"] and self.sim.t >= c["live_at"]:
                c["announced"] = True
                c["cell"].status = "up"
                self.sim.emit("F5", f"Temporary cell {c['cell'].id} ON AIR",
                              f"covering the {c['for']} outage area; traffic moves to it", "good")

    def summary(self):
        s = self.stats
        sent = max(1, s["sent"])
        fs = max(1, s["failover_sent"])
        return {"sent": s["sent"], "success": s["delivered"] / sent, "baseline_success": s["base_delivered"] / sent,
                "first_try": s["first_try"] / sent, "failover_sent": s["failover_sent"],
                "failover_success": s["failover_delivered"] / fs if s["failover_sent"] else None,
                "failover_baseline": s["failover_base"] / fs if s["failover_sent"] else None,
                "by_path": s["by_path"], "load_sent": s["load_sent"],
                "load_success": s["load_delivered"] / max(1, s["load_sent"]),
                "load_baseline": s["load_base"] / max(1, s["load_sent"])}
