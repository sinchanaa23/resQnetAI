"""CAMARA API client used by the orchestrator.

mode="mock" (default) calls the in-process simulator. mode="http" sends the same
request bodies to a real CAMARA gateway, e.g. an operator sandbox:

    RESQNET_CAMARA_MODE=http
    RESQNET_CAMARA_BASE_URL=https://<operator-sandbox>/
    RESQNET_CAMARA_TOKEN=<token>
"""
import json
import os
import time
import urllib.request

ENDPOINTS = {
    "number_verification": "number-verification/v0/verify",
    "sim_swap": "sim-swap/v0/check",
    "population_density": "population-density-data/v0.1/retrieve",
    "qod_create": "quality-on-demand/v0/sessions",
    "location": "location-retrieval/v0/retrieve",
    "geofence": "geofencing-subscriptions/v0.3/subscriptions",
}


class CamaraClient:
    def __init__(self, network, on_call=None):
        self.net = network
        self.on_call = on_call or (lambda entry: None)
        self.mode = os.environ.get("RESQNET_CAMARA_MODE", "mock")
        self.base = os.environ.get("RESQNET_CAMARA_BASE_URL", "")
        self.token = os.environ.get("RESQNET_CAMARA_TOKEN", "")

    def _call(self, name, body, mock_fn, summary):
        t0 = time.perf_counter()
        if self.mode == "http":
            req = urllib.request.Request(self.base.rstrip("/") + "/" + ENDPOINTS[name], data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.token}"})
            resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
        else:
            resp = mock_fn()
        self.on_call({"api": ENDPOINTS[name], "summary": summary(resp),
                      "ms": round((time.perf_counter() - t0) * 1000, 2)})
        return resp

    def verify_number(self, phone, network_msisdn):
        body = {"phoneNumber": phone}
        return self._call("number_verification", body,
                          lambda: self.net.number_verification_verify(body, network_msisdn),
                          lambda r: f"{phone} -> verified={r['devicePhoneNumberVerified']}")

    def sim_swap(self, phone, max_age_h=4):
        body = {"phoneNumber": phone, "maxAge": max_age_h}
        return self._call("sim_swap", body, lambda: self.net.sim_swap_check(body),
                          lambda r: f"{phone} swapped<{max_age_h}h={r['swapped']}")

    def population_density(self, cell_ids):
        body = {"area": {"areaType": "CELLS"}, "cellIds": list(cell_ids), "precision": 7}
        return self._call("population_density", body, lambda: self.net.population_density_retrieve(body),
                          lambda r: f"{len(r['timedPopulationDensityData'][0]['cellPopulationDensityData'])} cells, aggregated")

    def qod_session(self, profile, area, duration=3600):
        body = {"qosProfile": profile, "area": area, "duration": duration,
                "applicationServer": {"ipv4Address": "10.20.0.5"}}
        return self._call("qod_create", body, lambda: self.net.qod_create_session(body),
                          lambda r: f"{r['sessionId']} {profile} {r['qosStatus']}")

    def locate(self, phone):
        body = {"device": {"phoneNumber": phone}, "maxAge": 60}
        return self._call("location", body, lambda: self.net.location_retrieve(body),
                          lambda r: "located (consented)" if "area" in r else r.get("code", "error"))

    def geofence(self, phone, area):
        body = {"device": {"phoneNumber": phone}, "area": area,
                "types": ["org.camaraproject.geofencing-subscriptions.v0.area-left"]}
        return self._call("geofence", body, lambda: self.net.geofencing_subscribe(body),
                          lambda r: r["subscriptionId"])
