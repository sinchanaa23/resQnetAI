"""Simulated world: cell sites, aggregated population, responders and medical volunteers.

Every person and phone number here is fictional. Locations are a real Bengaluru
locality (Koramangala / Ejipura / HSR Layout) so the map is recognisable.
"""
import math
import random

from .geo import haversine, offset

# Scenario anchors (lat, lon)
FLOOD_CENTER = (12.9372, 77.6233)   # low-lying Ejipura / Koramangala 6th Block area
FLOOD_RADIUS = 650
INCIDENT_POS = (12.9356, 77.6268)
AMBULANCE_BASE = (12.9135, 77.6385)  # HSR Layout ambulance standby point
HOSPITAL = ("St. John's Medical College Hospital", (12.9293, 77.6197))

CELL_SPACING = 720
CELL_RADIUS = 560


class Cell:
    def __init__(self, cid, pos, density, rng):
        self.id = cid
        self.pos = pos
        self.radius = CELL_RADIUS
        self.density = density  # active devices per km² (aggregated)
        self.base_load = rng.uniform(0.84, 0.95)
        self.load = self.base_load
        self.status = "up"  # up | down
        self.temp = False
        self.inflow_hist = []

    @property
    def devices(self):
        return int(self.density * math.pi * (self.radius / 1000) ** 2 * 0.62)  # share inside own best-server area


class Volunteer:
    def __init__(self, vid, name, role, spec, pos, rate, signal, available=True, assigned=False, night=0):
        self.id, self.name, self.role, self.spec = vid, name, role, spec
        self.pos = pos
        self.accept_rate = rate
        self.signal = signal
        self.available = available
        self.assigned = assigned
        self.night = night
        self.verified = True  # checked against the National Medical Commission register at onboarding
        self.state = "idle"   # idle | ringing | declined | timeout | accepted | en_route | arrived | skipped
        self.msisdn = "+91 99001 50" + vid[-3:]


class Responder:
    def __init__(self, rid, name, role, msisdn, pos, sim_swap_minutes_ago=100000, device="registered"):
        self.id, self.name, self.role, self.msisdn = rid, name, role, msisdn
        self.pos = pos
        self.sim_swap_minutes_ago = sim_swap_minutes_ago
        self.device = device
        self.revoked = False
        self.requests = []  # sim times of protected requests (rate limiting)


# Fictional people; specialisations reflect what an urban volunteer pool would contain.
VOLUNTEERS = [
    # id, name, role, specialisation, (dlat_m, dlon_m) from incident, accept-rate, signal, available, assigned
    ("V-101", "Dr. Arjun Mehta", "doctor", "Emergency Medicine", (-650, 420), 0.48, 0.55, True, False),
    ("V-102", "Dr. Priya Nair", "doctor", "Emergency Medicine", (-1150, 820), 0.86, 0.9, True, False),
    ("V-103", "Dr. Kavya Reddy", "doctor", "General Physician", (380, -260), 0.7, 0.8, True, True),
    ("V-104", "Rohan D'Souza", "paramedic", "Paramedic (ALS)", (-1500, -900), 0.75, 0.85, True, False),
    ("V-105", "Dr. Farhan Ali", "doctor", "Orthopaedics", (1650, 1200), 0.62, 0.7, True, False),
    ("V-106", "Sister Meera Joseph", "nurse", "Critical Care Nursing", (-2100, 300), 0.8, 0.75, True, False),
    ("V-107", "Dr. Vikram Shetty", "doctor", "Trauma Surgery", (520, 610), 0.66, 0.6, False, False),
    ("V-108", "Dr. Ananya Rao", "doctor", "Paediatrics", (-2600, 1500), 0.72, 0.9, True, False),
    ("V-109", "Dr. Suresh Gowda", "doctor", "General Physician", (-3100, -400), 0.58, 0.8, True, False),
    ("V-110", "Lakshmi Prasad", "paramedic", "Paramedic (BLS)", (900, -1500), 0.69, 0.7, True, False),
]

SPEC_FIT = {"Emergency Medicine": 1, "Trauma Surgery": 1, "Paramedic (ALS)": 1, "Critical Care Nursing": 1,
            "General Physician": 0, "Orthopaedics": 0, "Paediatrics": 0, "Paramedic (BLS)": 1}


class World:
    def __init__(self, graph, models, seed=3):
        self.g = graph
        self.models = models
        self.rng = random.Random(seed)
        self._build_cells()
        self.volunteers = [Volunteer(v[0], v[1], v[2], v[3], offset(INCIDENT_POS, v[4][1], v[4][0]), v[5], v[6],
                                     v[7], v[8]) for v in VOLUNTEERS]
        self.responders = [
            Responder("AMB-07", "Ravi Kumar (driver, 108 ambulance)", "ambulance", "+91 98450 11207", AMBULANCE_BASE),
            Responder("AMB-03", "Imran Pasha (driver, 108 ambulance)", "ambulance", "+91 98450 11203",
                      offset(AMBULANCE_BASE, -900, 600), sim_swap_minutes_ago=47),
            Responder("FIRE-02", "Station Officer B. Naik", "fire", "+91 98450 22102", (12.9265, 77.6110)),
            Responder("POL-21", "SI Deepa Hegde (Koramangala PS)", "police", "+91 98450 33121", (12.9335, 77.6180)),
            Responder("RES-04", "SDRF boat team R-4", "rescue", "+91 98450 44104", offset(FLOOD_CENTER, 160, 120)),
        ]
        # Phone-to-network identity mapping used by the mock CAMARA Number Verification API.
        self.network_identity = {r.msisdn: r.msisdn for r in self.responders}
        self.spoofed_sessions = {}

    def _build_cells(self):
        la0, lo0, la1, lo1 = self.g.bbox
        origin = (la0, lo0)
        cells, k = [], 1
        rows = int(haversine((la0, lo0), (la1, lo0)) / (CELL_SPACING * 0.866)) + 2
        cols = int(haversine((la0, lo0), (la0, lo1)) / CELL_SPACING) + 2
        # Road density near each site drives aggregated population density.
        for r in range(rows):
            for c in range(cols):
                x = c * CELL_SPACING + (CELL_SPACING / 2 if r % 2 else 0) - 150
                y = r * CELL_SPACING * 0.866 - 150
                p = offset(origin, x, y)
                if not (la0 - 0.002 <= p[0] <= la1 + 0.002 and lo0 - 0.002 <= p[1] <= lo1 + 0.002):
                    continue
                roads = sum(1 for m in self.g.mid[::7] if haversine(m, p) < 450)
                density = 4000 + roads * 260 + self.rng.uniform(-1500, 1500)
                cell = Cell(f"KRM-{k:03d}", p, max(2500, density), self.rng)
                cell.inflow_hist = self.models.inflow_series(self.rng, base=40 + roads * 1.6, days=2)
                cells.append(cell)
                k += 1
        self.cells = cells
        self.cell_by_id = {c.id: c for c in cells}

    # ---- radio helpers --------------------------------------------------
    def serving_cell(self, p, include_down=False):
        best, bd = None, 1e18
        for c in self.cells:
            if (c.status == "down" and not include_down) or c.temp:
                continue
            d = haversine(p, c.pos)
            if d < bd:
                best, bd = c, d
        return best, bd

    def home_cell(self, p):
        """The cell that would normally serve this location (even if it is down)."""
        return self.serving_cell(p, include_down=True)[0]

    def neighbour_cell(self, p, exclude, attenuation_m=0):
        """Nearest *other* working cell whose extended coverage still reaches p."""
        best, bd = None, 1e18
        for c in self.cells:
            if c is exclude or c.status == "down" or c.temp:
                continue
            d = haversine(p, c.pos)
            if d < bd:
                best, bd = c, d
        reach = (best.radius * 1.55 - attenuation_m) if best else 0
        return (best, bd) if best and bd <= reach else (None, bd)

    def avg_load(self):
        up = [c.load for c in self.cells if c.status == "up" and not c.temp]
        return sum(up) / len(up)

    def volunteer(self, vid):
        return next((v for v in self.volunteers if v.id == vid), None)

    def responder_by_msisdn(self, msisdn):
        norm = msisdn.replace(" ", "")
        return next((r for r in self.responders if r.msisdn.replace(" ", "") == norm), None)

    def responder(self, rid):
        return next((r for r in self.responders if r.id == rid), None)

    def volunteer_features(self, v, target):
        dist_km = haversine(v.pos, target) / 1000
        return [dist_km, v.night, v.accept_rate, 1 if v.assigned else 0, SPEC_FIT.get(v.spec, 0), v.signal]
