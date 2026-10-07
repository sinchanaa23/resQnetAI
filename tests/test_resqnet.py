"""End-to-end and per-feature tests. Run: python -m pytest -q"""
import pytest
from fastapi.testclient import TestClient

from backend.demo import Director
from backend.engine import Simulation
from backend.sim.world import INCIDENT_POS


@pytest.fixture(scope="session")
def base():
    return Simulation()  # trains the models once


@pytest.fixture
def sim(base):
    base.reset()
    return base


def run(sim, seconds, dt=0.5):
    for _ in range(int(seconds / dt)):
        sim._substep(dt)


def test_models_beat_baselines(sim):
    for name, m in sim.models.report.items():
        better = m["ai"] < m["baseline"] if m["lower_is_better"] else m["ai"] > m["baseline"]
        assert better, name
    assert sim.models.report["classifier"]["ai"] >= 0.95  # recall on held-out phrasings


@pytest.mark.parametrize("text,label", [
    ("ghar mein paani bhar gaya bachao", "emergency"),
    ("ಸಹಾಯ ಮಾಡಿ ಮನೆಗೆ ನೀರು ನುಗ್ಗಿದೆ", "emergency"),
    ("my father collapsed please help", "emergency"),
    ("1", "emergency"),
    ("all safe here just calling office", "normal"),
])
def test_classifier(sim, text, label):
    assert sim.models.classify(text)["label"] == label


def test_f1_identity_checks(sim):
    inc = sim.create_incident(INCIDENT_POS)
    swapped = sim.sms("+91 98450 11203", f"CLEAR ROUTE {inc['id']}")
    assert not swapped["ok"]
    police = sim.sms("+91 98450 33121", f"CLEAR ROUTE {inc['id']}")
    assert not police["ok"]  # role not allowed to request route clearance
    spoof = sim.sms("+91 98450 11207", f"CLEAR ROUTE {inc['id']}", network_msisdn="+91 90000 00000")
    assert not spoof["ok"]
    ok = sim.sms("+91 98450 11207", f"CLEAR ROUTE {inc['id']}")
    assert ok["ok"] and sim.trips
    assert sim.audit.verify()["valid"]


def test_audit_log_is_tamper_evident(sim):
    sim.create_incident(INCIDENT_POS)
    sim.audit.append(sim.t, "x", "y", "z")
    sim.audit.entries[0]["detail"] = "edited"
    assert not sim.audit.verify()["valid"]


def test_f2_route_clearance_corridor(sim):
    inc = sim.create_incident(INCIDENT_POS)
    sim.sms("+91 98450 11207", f"CLEAR ROUTE {inc['id']}")
    run(sim, 120)
    trip = sim.trips[0]
    s = trip.summary()
    assert s["cells_alerted"] >= 2 and trip.corridor
    assert s["precision"] > s["blanket_precision"]
    assert len(set(trip.alerted)) == len(trip.alerted)  # one alert per cell per trip


def test_f3_matching_escalates_on_decline(sim):
    inc = sim.create_incident(INCIDENT_POS, critical=False, people=2, injuries=1)
    req = sim.request_medic(inc)
    first = req.contacted[0]
    assert sim.world.volunteer(first).state == "ringing"
    assert all(v.name != c["v"].name for v, _ in req.skipped for c in req.ranked)  # unavailable never ranked
    sim.ivr(first, "2")
    second = req.contacted[1]
    sim.ivr(second, "1")
    assert req.assigned.id == second and req.accept_t is not None


def test_f3_timeout_escalates(sim):
    inc = sim.create_incident(INCIDENT_POS, critical=False, people=2, injuries=1)
    req = sim.request_medic(inc)
    run(sim, 50)
    assert len(req.contacted) >= 2


def test_f4_zone_priority(sim):
    sim.calls.declare_zone(INCIDENT_POS)
    assert sim.calls.slice_active and sim.network.qod_sessions
    r112 = sim.calls.handle_call(INCIDENT_POS, "112")
    assert r112["priority"] and r112["route"] == "auto"
    rc = sim.calls.handle_call(INCIDENT_POS, "+91 9876543210", reply="bachao hum phas gaye hain")
    assert rc["priority"] and rc["route"] == "classified"
    rn = sim.calls.handle_call(INCIDENT_POS, "+91 9876543210", reply="all safe here thank you")
    assert not rn["priority"]
    sim.calls.lift_zone()
    assert not sim.calls.slice_active


def test_f5_failover(sim):
    sim.create_incident(INCIDENT_POS)
    cell = sim.world.home_cell(INCIDENT_POS)
    sim.fail_cell(cell.id)
    results = [sim.conn.deliver("test", INCIDENT_POS, "x") for _ in range(40)]
    assert sum(r["ok"] for r in results) / 40 >= 0.9
    assert all(r["path"] != "cellular" for r in results if r["ok"])
    run(sim, 80)  # temporary cell comes on air
    assert any(c.temp and c.status == "up" for c in sim.world.cells)


def test_f6_edge_faster_than_cloud(sim):
    sim.create_incident(INCIDENT_POS)
    sim.sms("+91 98450 11207", "CLEAR ROUTE INC-2041")
    s = sim.latency.summary()
    assert s["ratio"] >= 3


def test_hero_demo_meets_targets(sim):
    sim.director = Director(sim)
    n = 0
    while not sim.director.done and n < 9000:
        sim.step(0.1)
        n += 1
    assert sim.director.done
    rows = sim.results()["rows"]
    failed = [r["metric"] for r in rows if r["pass"] is False]
    assert not failed, failed


def test_api_and_camara_endpoints():
    from backend.app import app
    with TestClient(app) as c:
        assert c.get("/").status_code == 200
        r = c.post("/api/incident", json={"lat": INCIDENT_POS[0], "lon": INCIDENT_POS[1], "critical": True})
        iid = r.json()["id"]
        r = c.post("/api/sms", json={"sender": "+91 98450 11207", "text": f"CLEAR ROUTE {iid}"})
        assert r.json()["ok"]
        r = c.post("/camara/sim-swap/v0/check", json={"phoneNumber": "+91 98450 11203", "maxAge": 4})
        assert r.json() == {"swapped": True}
        r = c.post("/camara/number-verification/v0/verify", json={"phoneNumber": "+91 98450 11207"},
                   headers={"X-Network-MSISDN": "+91 98450 11207"})
        assert r.json()["devicePhoneNumberVerified"]
        r = c.post("/camara/quality-on-demand/v0/sessions", json={"qosProfile": "QOS_E"})
        assert r.json()["qosStatus"] == "AVAILABLE"
        assert len(c.get("/api/roads").json()) > 5000
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "init"
