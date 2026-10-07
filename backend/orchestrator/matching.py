"""F3: AI-based medical responder matching with IVR accept / decline and automatic escalation."""
from ..sim.geo import haversine, interpolate
from .security import Timer

IVR_TIMEOUT_S = 45
TWO_WHEELER_BOOST = 1.25  # volunteers on two-wheelers filter through traffic


class MedicRequest:
    def __init__(self, sim, incident, requested_by):
        self.sim = sim
        self.incident = incident
        self.requested_by = requested_by
        self.t0 = sim.t
        self.parallel = incident["severity"] >= 5
        self.contacted = []      # volunteer ids in contact order
        self.ringing = {}        # vid -> ring start time
        self.script = {}         # vid -> (response, after_s) used by the scripted demo
        self.assigned = None
        self.accept_t = None
        self.route = None
        self.d = 0.0
        self.arrived = False
        self.rank()
        self.contact_next()

    def rank(self):
        sim, w = self.sim, self.sim.world
        target = self.incident["pos"]
        with Timer() as tm:
            ranked, skipped = [], []
            for v in w.volunteers:
                if not v.available:
                    skipped.append((v, "not available (opted out) — location not requested"))
                    continue
                if v.assigned:
                    skipped.append((v, "currently assigned to another case"))
                    continue
                loc = sim.camara.locate(v.msisdn)
                if "area" in loc:
                    v.pos = (loc["area"]["center"]["latitude"], loc["area"]["center"]["longitude"])
                f = w.volunteer_features(v, target)
                score, p, eta = sim.models.match_score(f)
                ranked.append({"v": v, "score": score, "p": p, "eta": eta, "dist": f[0], "fit": f[4]})
            ranked.sort(key=lambda c: -c["score"])
        sim.latency.record("responder ranking", tm.ms, w.avg_load())
        self.ranked = ranked
        self.skipped = skipped
        reasons = [f"#{i + 1} {c['v'].name}: {c['dist']:.1f} km · {c['v'].spec} · p(accept) {c['p']:.0%} · "
                   f"ETA {c['eta']:.0f} min" for i, c in enumerate(ranked[:4])]
        alts = [f"{v.name}: {why}" for v, why in skipped]
        sim.decide("F3", f"Ranked {len(ranked)} volunteer medics for {self.incident['id']}", reasons, alts, tm.ms)
        sim.emit("F3", f"Medic request for {self.incident['id']} — {len(ranked)} candidates ranked",
                 ("Parallel mode: top 3 contacted at once" if self.parallel else
                  f"Top: {ranked[0]['v'].name} ({ranked[0]['dist']:.1f} km, {ranked[0]['v'].spec})") if ranked else
                 "no candidates", "info")

    def contact_next(self):
        if self.assigned:
            return
        n = 3 if self.parallel and not self.contacted else 1
        pool = [c for c in self.ranked if c["v"].id not in self.contacted]
        if not pool:
            if not self.ringing:
                self.sim.emit("F3", "No volunteer accepted — escalating to 108 ALS unit", "", "bad")
            return
        for c in pool[:n]:
            v = c["v"]
            self.contacted.append(v.id)
            self.ringing[v.id] = self.sim.t
            v.state = "ringing"
            km = haversine(v.pos, self.incident["pos"]) / 1000
            v.ivr_text = (f"Emergency medical assistance is required approximately {km:.1f} km from your current "
                          f"location. Press 1 to accept the request or 2 to decline.")
            res = self.sim.conn.deliver("IVR call", v.pos, v.name)
            self.sim.emit("F3", f"Automated call → {v.name} (#{self.contacted.index(v.id) + 1})",
                          f"“{v.ivr_text}”" + ("" if res["ok"] else " (delivery retry)"), "info")

    def respond(self, vid, digit, source="IVR"):
        if vid not in self.ringing:
            return False
        sim = self.sim
        v = sim.world.volunteer(vid)
        del self.ringing[vid]
        if digit == "1" and not self.assigned:
            v.state = "accepted"
            self.assigned = v
            self.accept_t = sim.t
            for other in list(self.ringing):
                sim.world.volunteer(other).state = "idle"
                del self.ringing[other]
            g = sim.graph
            self.route = g.route(g.nearest_node(v.pos), g.nearest_node(self.incident["pos"]),
                                 lambda e: g.true_time(e, sim.hour) / TWO_WHEELER_BOOST)
            self.eta = sum(g.true_time(e, sim.hour) / TWO_WHEELER_BOOST for e, _ in self.route.edges)
            sim.camara.geofence(v.msisdn, {"areaType": "CIRCLE", "center": {"latitude": self.incident["pos"][0],
                                           "longitude": self.incident["pos"][1]}, "radius": 100})
            sim.audit.append(sim.t, v.id, "medic_assignment", "ACCEPTED", self.incident["id"])
            sim.emit("F3", f"{v.name} ACCEPTED (pressed 1)",
                     f"Incident team notified · ETA {self.eta / 60:.0f} min · selection time "
                     f"{sim.t - self.t0:.0f}s after {len(self.contacted)} contact(s)", "good")
            c = next(c for c in self.ranked if c["v"] is v)
            sim.decide("F3", f"Assigned {v.name}",
                       [f"{c['dist']:.1f} km away · {v.spec} · NMC-verified",
                        f"Available (consented location sharing), no active assignment",
                        f"Model p(accept) was {c['p']:.0%}; ETA {self.eta / 60:.0f} min by two-wheeler"],
                       [f"{sim.world.volunteer(x).name}: {sim.world.volunteer(x).state}" for x in self.contacted
                        if x != vid], 0.2)
        else:
            v.state = "declined" if digit == "2" else "timeout"
            sim.emit("F3", f"{v.name} {'declined (pressed 2)' if digit == '2' else 'did not answer'}",
                     "escalating to next-best candidate automatically", "warn")
            if not self.ringing:
                self.contact_next()
        return True

    def tick(self, dt):
        sim = self.sim
        for vid, t0 in list(self.ringing.items()):
            sc = self.script.get(vid)
            if sc and sim.t - t0 >= sc[1]:
                self.respond(vid, sc[0], "script")
            elif sim.t - t0 > IVR_TIMEOUT_S:
                self.respond(vid, "timeout")
        if self.assigned and self.route and not self.arrived:
            v = self.assigned
            k = self.route.edge_at(self.d)
            ei = self.route.edges[k][0]
            self.d += sim.graph.true_speed_kmh(ei, sim.hour) * TWO_WHEELER_BOOST / 3.6 * dt
            v.state = "en_route"
            if self.d >= self.route.length:
                self.d = self.route.length
                self.arrived = True
                v.state = "arrived"
                sim.emit("F3", f"{v.name} arrived at {self.incident['id']}", "first medical response on scene", "good")
            v.pos = interpolate(self.route.points, self.route.cum, self.d)

    def eta_remaining(self):
        if not self.route:
            return None
        frac = 1 - self.d / max(1.0, self.route.length)
        return self.eta * frac

    def summary(self):
        return {"incident": self.incident["id"], "contacted": len(self.contacted),
                "assigned": self.assigned.name if self.assigned else None,
                "selection_s": (self.accept_t - self.t0) if self.accept_t is not None else None,
                "eta_s": self.eta_remaining(), "arrived": self.arrived, "parallel": self.parallel,
                "ringing": list(self.ringing)}
