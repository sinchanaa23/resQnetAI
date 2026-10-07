"""F4: citizen emergency-call prioritisation inside an authority-declared disaster zone."""
import itertools
import math
import random

from ..camara.mock import circle_area
from ..ml.training import EMERG_CORE, NONEMERG_CORE, PREFIX, SUFFIX
from ..sim.geo import haversine, offset
from .security import Timer

EMERGENCY_NUMBERS = {"112", "108", "101", "100"}
SLICE_SHARE = 0.20          # capacity reserved for the emergency slice in the zone
CALL_RATE = 0.55            # calls per sim-second across the monitored area
_call_ids = itertools.count(1001)


class CallPrioritizer:
    def __init__(self, sim, seed=13):
        self.sim = sim
        self.rng = random.Random(seed)
        self.zone = None
        self.slice_active = False
        self.qod = None
        self.feed = []
        self.stats = {"total": 0, "auto": 0, "prompted": 0, "true_emerg": 0, "tp": 0, "fn": 0, "fp": 0,
                      "emerg_connected": 0, "emerg_connected_base": 0, "emerg_wait": 0.0, "emerg_wait_base": 0.0,
                      "normal_in_zone": 0, "normal_degraded": 0}
        self._acc = 0.0

    # ---- zone & emergency mode -------------------------------------------------
    def declare_zone(self, center, radius=1000, buffer=500, by="District Disaster Mgmt. Authority"):
        sim = self.sim
        self.zone = {"center": center, "radius": radius, "buffer": buffer, "declared_by": by, "t": sim.t}
        with Timer() as tm:
            self.qod = sim.camara.qod_session("QOS_E_EMERGENCY_SLICE", circle_area(center, radius + buffer))
        self.slice_active = True
        sim.latency.record("zone activation", tm.ms, sim.world.avg_load())
        sim.audit.append(sim.t, by, "declare_zone", "ACTIVE", f"r={radius}m+{buffer}m buffer; {self.qod['sessionId']}")
        sim.audit.append(sim.t, "resQnet", "emergency_mode", "ON", "regulator-authorised, zone-limited, auto-expiring")
        sim.decide("F4", "Emergency mode ON for declared zone",
                   [f"Declared by {by}: {radius} m radius + {buffer} m buffer",
                    f"Emergency traffic → 5G emergency slice / QoD session {self.qod['sessionId']} "
                    f"({int(SLICE_SHARE * 100)}% capacity reserved)",
                    "Calls to 112/108/101/100 and verified responders prioritised instantly (no prompt)",
                    "Other calls in zone get a short emergency prompt; best-effort kept for everyone else",
                    "Mode ends automatically when the declaration is lifted"],
                   ["Blanket priority for all traffic (breaks neutrality)", "No prioritisation"], tm.ms)
        sim.emit("F4", "Disaster zone declared — emergency mode ON",
                 f"{radius} m + {buffer} m buffer · QoD {self.qod['sessionId']} · emergency slice active", "warn")

    def lift_zone(self):
        sim = self.sim
        if not self.zone:
            return
        if self.qod:
            sim.network.qod_delete_session(self.qod["sessionId"])
        sim.audit.append(sim.t, "District Disaster Mgmt. Authority", "declare_zone", "LIFTED", "")
        sim.audit.append(sim.t, "resQnet", "emergency_mode", "OFF", "auto-ended with declaration")
        self.zone, self.qod, self.slice_active = None, None, False
        sim.emit("F4", "Zone lifted — emergency mode OFF", "all traffic back to normal best-effort", "good")

    def in_zone(self, p):
        if not self.zone:
            return None
        d = haversine(p, self.zone["center"])
        if d <= self.zone["radius"]:
            return "zone"
        if d <= self.zone["radius"] + self.zone["buffer"]:
            return "buffer"
        return None

    # ---- call handling ---------------------------------------------------------------
    def handle_call(self, pos, dialed, caller="citizen", reply=None, truth=None, manual=False):
        sim = self.sim
        cid = f"C{next(_call_ids)}"
        where = self.in_zone(pos)
        rec = {"id": cid, "t": sim.t, "pos": pos, "dialed": dialed, "where": where or "outside", "caller": caller}
        with Timer() as tm:
            if dialed in EMERGENCY_NUMBERS:
                rec.update(priority=True, route="auto", why=f"emergency number {dialed} — bypasses classifier")
                is_emerg = True
            elif caller == "responder":
                rec.update(priority=True, route="auto", why="verified responder SIM — instant priority")
                is_emerg = True
            elif where and self.slice_active:
                rec["prompt"] = "Do you need emergency assistance? Say what is happening, or press 1."
                rec["reply"] = reply
                res = sim.models.classify(reply)
                rec.update(priority=res["label"] == "emergency", route="classified", p=res["p"],
                           why=(", ".join(res["why"]) or "—") + (" · uncertain → treated as emergency"
                                                                 if res.get("uncertain") and res["label"] == "emergency" else ""))
                is_emerg = res["label"] == "emergency"
            else:
                rec.update(priority=False, route="normal", why="outside zone / no emergency mode")
                is_emerg = False
        sim.latency.record("call classification", tm.ms, sim.world.avg_load())
        self._account(rec, is_emerg, truth)
        self.feed.append(rec)
        self.feed = self.feed[-60:]
        if manual:
            sim.emit("F4", f"Call {cid} ({dialed}) → {'PRIORITY' if rec['priority'] else 'normal'}",
                     rec["why"], "good" if rec["priority"] else "info")
        return rec

    def _account(self, rec, predicted, truth):
        s = self.stats
        s["total"] += 1
        if rec["route"] == "auto":
            s["auto"] += 1
        if rec["route"] == "classified":
            s["prompted"] += 1
        if truth is None:
            truth = predicted
        if truth:
            s["true_emerg"] += 1
            if predicted:
                s["tp"] += 1
            else:
                s["fn"] += 1
            # Call set-up at ~90% load: priority calls on the slice vs. everyone competing (baseline).
            ok = self.rng.random() < (0.995 if rec["priority"] and self.slice_active else 0.82)
            ok_base = self.rng.random() < 0.82
            s["emerg_connected"] += ok
            s["emerg_connected_base"] += ok_base
            s["emerg_wait"] += 0.4 if rec["priority"] and self.slice_active else self.rng.uniform(3, 11)
            s["emerg_wait_base"] += self.rng.uniform(3, 11)
            rec["connected"] = ok
        elif predicted:
            s["fp"] += 1
        if rec["where"] in ("zone", "buffer") and not rec["priority"]:
            s["normal_in_zone"] += 1
            s["normal_degraded"] += self.slice_active

    def synth_call(self):
        """Background call traffic around the incident area."""
        r = self.rng
        center = self.zone["center"] if self.zone else self.sim.scenario_center
        ang, dist = r.uniform(0, 6.283), r.uniform(0, 1900)
        pos = offset(center, dist * math.cos(ang), dist * math.sin(ang))
        u = r.random()
        if u < 0.06:
            return self.handle_call(pos, r.choice(sorted(EMERGENCY_NUMBERS)), truth=True)
        if u < 0.09:
            return self.handle_call(pos, "+91 98450 " + str(r.randint(10000, 99999)), caller="responder", truth=True)
        # True emergencies come from inside the declared zone; elsewhere people dial 112 directly.
        emerg = bool(self.in_zone(pos)) and r.random() < 0.25
        core = r.choice(EMERG_CORE if emerg else NONEMERG_CORE)
        reply = " ".join(x for x in (r.choice(PREFIX), core, r.choice(SUFFIX)) if x)
        if r.random() < 0.2:
            reply = "1" if emerg else "2"
        return self.handle_call(pos, "+91 9" + str(r.randint(100000000, 999999999)), reply=reply, truth=emerg)

    def tick(self, dt):
        self._acc += dt * CALL_RATE
        while self._acc >= 1:
            self._acc -= 1
            self.synth_call()

    def summary(self):
        s = self.stats
        te = max(1, s["true_emerg"])
        return {"zone": self.zone, "slice_active": self.slice_active, "qod": self.qod["sessionId"] if self.qod else None,
                "total": s["total"], "auto": s["auto"], "prompted": s["prompted"],
                "recall": s["tp"] / te if s["true_emerg"] else None,
                "false_alarms": s["fp"], "emergencies": s["true_emerg"],
                "connected": s["emerg_connected"] / te if s["true_emerg"] else None,
                "connected_base": s["emerg_connected_base"] / te if s["true_emerg"] else None,
                "wait": s["emerg_wait"] / te if s["true_emerg"] else None,
                "wait_base": s["emerg_wait_base"] / te if s["true_emerg"] else None,
                "slice_share": SLICE_SHARE}
