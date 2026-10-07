"""F1 + F6: verified responder identity, role-based access, rate limits, tamper-evident audit log,
and edge-vs-cloud decision latency tracking."""
import hashlib
import json
import random
import time

# Which roles may trigger which protected actions (proposal §4, Feature 1).
PERMISSIONS = {
    "route_clearance": {"ambulance", "fire"},
    "medic_request": {"ambulance", "fire", "police", "rescue"},
    "declare_zone": {"authority"},
}
RATE_LIMIT = (3, 600)  # max route-clearance requests per responder per 10 sim-minutes


class AuditLog:
    """Append-only, hash-chained log: editing any entry breaks every later hash."""

    def __init__(self):
        self.entries = []

    def append(self, t, actor, action, outcome, detail=""):
        prev = self.entries[-1]["hash"] if self.entries else "0" * 64
        e = {"seq": len(self.entries) + 1, "t": round(t, 1), "actor": actor, "action": action,
             "outcome": outcome, "detail": detail, "prev": prev}
        e["hash"] = hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()
        self.entries.append(e)
        return e

    def verify(self):
        prev = "0" * 64
        for e in self.entries:
            body = {k: v for k, v in e.items() if k != "hash"}
            if e["prev"] != prev or hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != e["hash"]:
                return {"valid": False, "broken_at": e["seq"]}
            prev = e["hash"]
        return {"valid": True, "entries": len(self.entries), "head": prev[:16]}


class IdentityService:
    def __init__(self, sim):
        self.sim = sim

    def authorize(self, msisdn, network_msisdn, action):
        """Returns (allowed, responder, checks[]) — each check is (name, passed, detail)."""
        sim, w = self.sim, self.sim.world
        checks = []
        r = w.responder_by_msisdn(msisdn)
        checks.append(("Registered responder", r is not None, f"{r.id} · {r.role}" if r else "number not in registry"))
        nv = sim.camara.verify_number(msisdn, network_msisdn)["devicePhoneNumberVerified"]
        checks.append(("CAMARA Number Verification", nv, "network confirms the SIM" if nv else "network identity mismatch"))
        sw = sim.camara.sim_swap(msisdn, max_age_h=4)["swapped"]
        if r and sw:
            checks.append(("CAMARA SIM Swap", False, f"SIM swapped {r.sim_swap_minutes_ago} min ago"))
        else:
            checks.append(("CAMARA SIM Swap", True, "no swap in last 4 h"))
        if r:
            checks.append(("Credential active", not r.revoked, "revoked by authority" if r.revoked else "active"))
            allowed_roles = PERMISSIONS.get(action, set())
            checks.append(("Role permission", r.role in allowed_roles,
                           f"{r.role} may {action.replace('_', ' ')}" if r.role in allowed_roles
                           else f"{r.role} may not {action.replace('_', ' ')}"))
            if action == "route_clearance":
                recent = [t for t in r.requests if sim.t - t < RATE_LIMIT[1]]
                ok = len(recent) < RATE_LIMIT[0]
                checks.append(("Rate limit", ok, f"{len(recent)}/{RATE_LIMIT[0]} in 10 min"))
        allowed = all(c[1] for c in checks)
        if allowed and r and action == "route_clearance":
            r.requests.append(sim.t)
        sim.audit.append(sim.t, r.id if r else msisdn, action, "ALLOWED" if allowed else "BLOCKED",
                         "; ".join(f"{n}: {'ok' if p else 'FAIL'}" for n, p, _ in checks))
        return allowed, r, checks


class LatencyTracker:
    """Measures real decision compute time and adds modelled network round trips.

    Edge (MEC in the operator's aggregation site): ~2-4 ms RTT.
    Cloud region over a congested backhaul at 90 % load: ~45-70 ms each way plus queueing.
    """

    def __init__(self, seed=5):
        self.rng = random.Random(seed)
        self.samples = []
        self.cloud_link = True

    def record(self, kind, compute_ms, load):
        edge = compute_ms + self.rng.uniform(2.0, 4.0)
        cloud = compute_ms + 2 * self.rng.uniform(45, 70) * (1 + 1.5 * max(0.0, load - 0.7)) if self.cloud_link else None
        s = {"kind": kind, "compute": round(compute_ms, 2), "edge": round(edge, 2),
             "cloud": round(cloud, 1) if cloud else None}
        self.samples.append(s)
        self.samples = self.samples[-300:]
        return s

    def summary(self):
        if not self.samples:
            return {"edge_ms": 0, "cloud_ms": 0, "ratio": 0, "n": 0, "recent": []}
        e = sum(s["edge"] for s in self.samples) / len(self.samples)
        cl = [s["cloud"] for s in self.samples if s["cloud"] is not None]
        c = sum(cl) / len(cl) if cl else 0
        return {"edge_ms": round(e, 1), "cloud_ms": round(c, 1), "ratio": round(c / e, 1) if e and c else 0,
                "n": len(self.samples), "recent": self.samples[-24:], "cloud_link": self.cloud_link}


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t0) * 1000
