"""Mock GSMA CAMARA network APIs backed by the simulator.

Request and response bodies follow the public CAMARA API shapes (simplified), so the
orchestrator can later point the same client at a real operator sandbox such as
Nokia Network as Code without code changes in the decision logic.
"""
import itertools
import math
from datetime import datetime, timedelta, timezone

from ..sim.geo import haversine

_ids = itertools.count(1)
SIM_EPOCH = datetime(2026, 7, 21, 19, 5, tzinfo=timezone.utc)  # simulated monsoon evening


class MockNetwork:
    def __init__(self, world, clock):
        self.w = world
        self.clock = clock  # callable -> sim seconds
        self.qod_sessions = {}
        self.geofences = {}

    def _now(self):
        return SIM_EPOCH + timedelta(seconds=self.clock())

    # ---- Number Verification --------------------------------------------
    def number_verification_verify(self, body, network_msisdn):
        """POST /number-verification/v0/verify. network_msisdn = number the network authenticated for this session."""
        claimed = body["phoneNumber"].replace(" ", "")
        return {"devicePhoneNumberVerified": network_msisdn is not None and network_msisdn.replace(" ", "") == claimed}

    # ---- SIM Swap ----------------------------------------------------------
    def sim_swap_check(self, body):
        """POST /sim-swap/v0/check {phoneNumber, maxAge (hours)}."""
        r = self.w.responder_by_msisdn(body["phoneNumber"])
        minutes = r.sim_swap_minutes_ago if r else 10 ** 6
        return {"swapped": minutes <= body.get("maxAge", 240) * 60}

    def sim_swap_retrieve_date(self, body):
        r = self.w.responder_by_msisdn(body["phoneNumber"])
        minutes = r.sim_swap_minutes_ago if r else 10 ** 6
        return {"latestSimChange": (self._now() - timedelta(minutes=minutes)).isoformat()}

    # ---- Population Density Data -----------------------------------------
    def population_density_retrieve(self, body):
        """POST /population-density-data/v0.1/retrieve — aggregated, anonymised counts per cell area."""
        area = body["area"]
        if area["areaType"] == "CIRCLE":
            c = (area["center"]["latitude"], area["center"]["longitude"])
            cells = [x for x in self.w.cells if haversine(c, x.pos) <= area["radius"] + x.radius]
        else:
            ids = set(body.get("cellIds", []))
            cells = [x for x in self.w.cells if x.id in ids]
        start = self._now()
        return {
            "status": "SUPPORTED_AREA",
            "timedPopulationDensityData": [{
                "startTime": start.isoformat(),
                "endTime": (start + timedelta(minutes=10)).isoformat(),
                "cellPopulationDensityData": [{
                    "cellId": x.id, "geohash": f"{x.pos[0]:.4f},{x.pos[1]:.4f}",
                    "dataType": "DENSITY_ESTIMATION",
                    "pplDensity": round(x.density, 1),
                    "minPplDensity": round(x.density * 0.85, 1), "maxPplDensity": round(x.density * 1.15, 1),
                } for x in cells]}],
        }

    # ---- Quality on Demand -----------------------------------------------
    def qod_create_session(self, body):
        """POST /quality-on-demand/v0/sessions."""
        sid = f"qod-{next(_ids):04d}"
        self.qod_sessions[sid] = {"sessionId": sid, "qosProfile": body["qosProfile"], "device": body.get("device"),
                                  "area": body.get("area"), "duration": body.get("duration", 3600),
                                  "qosStatus": "AVAILABLE", "startedAt": self._now().isoformat()}
        return self.qod_sessions[sid]

    def qod_delete_session(self, sid):
        s = self.qod_sessions.pop(sid, None)
        return {"sessionId": sid, "qosStatus": "UNAVAILABLE" if s else "NOT_FOUND"}

    # ---- Location Retrieval & Geofencing ----------------------------------
    def location_retrieve(self, body):
        """POST /location-retrieval/v0/retrieve — only for consenting volunteers / responders."""
        num = body["device"]["phoneNumber"].replace(" ", "")
        who = next((v for v in self.w.volunteers if v.msisdn.replace(" ", "") == num), None) or \
            self.w.responder_by_msisdn(num)
        if who is None:
            return {"status": 404, "code": "NOT_FOUND"}
        if getattr(who, "available", True) is False:
            return {"status": 403, "code": "CONSENT_NOT_GRANTED"}
        return {"lastLocationTime": self._now().isoformat(),
                "area": {"areaType": "CIRCLE", "center": {"latitude": who.pos[0], "longitude": who.pos[1]},
                         "radius": 50}}

    def geofencing_subscribe(self, body):
        """POST /geofencing-subscriptions/v0.3/subscriptions."""
        sid = f"geo-{next(_ids):04d}"
        self.geofences[sid] = body
        return {"subscriptionId": sid, "startsAt": self._now().isoformat(), "protocol": "HTTP",
                "types": body.get("types", ["org.camaraproject.geofencing-subscriptions.v0.area-entered"])}


def circle_area(center, radius):
    return {"areaType": "CIRCLE", "center": {"latitude": center[0], "longitude": center[1]}, "radius": int(radius)}


def area_km2(radius_m):
    return math.pi * (radius_m / 1000) ** 2
