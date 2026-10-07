"""Hero demo director (proposal §7): one continuous story that touches all six features."""
from .sim.world import FLOOD_CENTER, INCIDENT_POS


def _incident(sim):
    sim.calls.handle_call(INCIDENT_POS, "112", truth=True, manual=True)
    sim.create_incident(INCIDENT_POS, "flood", people=7, injuries=2, critical=True,
                        note="elderly man unconscious, water waist-deep", source="112 call")


def _zone(sim):
    sim.calls.declare_zone(FLOOD_CENTER, 1000, 500)


def _spoof(sim):
    inc = sim.incidents[0]["id"]
    sim.sms("+91 98450 11203", f"CLEAR ROUTE {inc}")  # AMB-03's number, SIM swapped 47 min ago


def _dispatch(sim):
    sim.speed = 6
    inc = sim.incidents[0]["id"]
    sim.last_sms = sim.sms("+91 98450 11207", f"CLEAR ROUTE {inc}")


def _medic(sim):
    req = sim.request_medic(sim.incidents[0], "AMB-07")
    if len(req.ranked) >= 2:
        req.script[req.ranked[0]["v"].id] = ("2", 16)   # first choice declines
        req.script[req.ranked[1]["v"].id] = ("1", 12)   # second accepts


def _tower(sim):
    cell = sim.world.home_cell(INCIDENT_POS)
    sim.fail_cell(cell.id)


def _edge(sim):
    sim.highlight = "latency"
    sim.emit("F6", "Edge vs cloud: decisions are taken on the MEC node",
             f"avg {sim.latency.summary()['edge_ms']} ms on edge vs {sim.latency.summary()['cloud_ms']} ms cloud-only", "info")


def _arrive(sim):
    sim.speed = 10
    sim.highlight = None


def _medic_done(sim):
    return bool(sim.medics and sim.medics[0].assigned)


def _trip_done(sim):
    return bool(sim.trips and sim.trips[0].finished is not None)


STEPS = [
    {"title": "Monsoon evening in Koramangala", "f": "SIM",
     "caption": "19:05, heavy flooding in Ejipura. Evening peak traffic; cell sites at ~90% load. "
                "Everything you see runs on the edge orchestrator against mock CAMARA network APIs.",
     "run": lambda s: None, "hold": 5},
    {"title": "1 · Citizen dials 112 from the flooded area", "f": "F4",
     "caption": "Emergency-number calls bypass every prompt and are prioritised instantly. "
                "The orchestrator creates an incident and scores its severity.",
     "run": _incident, "hold": 6},
    {"title": "2 · Authority declares a disaster zone", "f": "F4",
     "caption": "1 km zone + 500 m buffer. Emergency traffic moves to a priority slice (CAMARA Quality on Demand). "
                "Other calls in the zone get a short prompt; the multilingual classifier decides priority.",
     "run": _zone, "hold": 8},
    {"title": "3a · Spoofed route-clearance request is blocked", "f": "F1",
     "caption": "Someone uses ambulance AMB-03's number from a SIM swapped 47 minutes ago. "
                "The CAMARA SIM Swap check fails, so the request is blocked and audited.",
     "run": _spoof, "hold": 5},
    {"title": "3b · Verified ambulance requests route clearance by SMS", "f": "F1 · F2",
     "caption": "AMB-07 passes Number Verification + SIM Swap + role checks. The AI predicts the route and ETA, "
                "forecasts crowd inflow and alerts only the cells and users ahead. The corridor moves with the ambulance.",
     "run": _dispatch, "hold": 9},
    {"title": "4 · Nearest suitable doctor is matched", "f": "F3",
     "caption": "Volunteers are ranked by distance, specialisation, availability, workload and acceptance probability. "
                "The first choice declines; the system escalates automatically and the next accepts by pressing 1.",
     "run": _medic, "until": _medic_done, "hold": 5},
    {"title": "5 · A flooded cell site goes down", "f": "F5",
     "caption": "Messages switch to a neighbouring cell; an isolated SDRF boat team is reached by D2D relay; "
                "a temporary cell-on-wheels is dispatched and takes over when it comes on air.",
     "run": _tower, "hold": 22},
    {"title": "6 · Edge vs cloud", "f": "F6",
     "caption": "Every decision is timed. On the MEC node decisions take a few milliseconds; "
                "a cloud-only round trip over a congested backhaul is far slower.",
     "run": _edge, "hold": 7},
    {"title": "7 · Ambulance arrives — results", "f": "ALL",
     "caption": "The same scenario is compared with a 'no resQnet' baseline: shortest-path route, no alerts, "
                "no priority, no failover.",
     "run": _arrive, "until": _trip_done, "hold": 1},
]


class Director:
    def __init__(self, sim, auto=True):
        self.sim = sim
        self.auto = auto
        self.i = -1
        self.elapsed = 0.0
        self.ready = False
        self.done = False
        self.waiting_next = False
        self.advance()

    def advance(self):
        if self.i + 1 >= len(STEPS):
            self.done = True
            return
        self.i += 1
        self.elapsed = 0.0
        self.ready = False
        self.waiting_next = False
        STEPS[self.i]["run"](self.sim)

    def next(self):
        if not self.done:
            self.advance()

    def tick(self, dt_real):
        if self.done or self.sim.paused:
            return
        step = STEPS[self.i]
        if not self.ready:
            if step.get("until") and not step["until"](self.sim):
                return
            self.ready = True
            self.elapsed = 0.0
        self.elapsed += dt_real
        if self.elapsed >= step["hold"]:
            if self.i == len(STEPS) - 1:
                self.done = True
            elif self.auto:
                self.advance()
            else:
                self.waiting_next = True

    def state(self):
        s = STEPS[self.i]
        return {"i": self.i, "n": len(STEPS), "title": s["title"], "caption": s["caption"], "f": s["f"],
                "auto": self.auto, "waiting_next": self.waiting_next, "done": self.done,
                "steps": [x["title"] for x in STEPS]}
