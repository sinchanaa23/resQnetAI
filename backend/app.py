"""resQnet edge service: FastAPI + WebSocket live feed + mock CAMARA endpoints.

Run:  python -m uvicorn backend.app:app --port 8000     (or ./run.bat / ./run.sh)
"""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .demo import Director
from .engine import Simulation

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
TICK_S = 0.1
BROADCAST_EVERY = 2  # ticks

sim: Simulation = None
clients: set = set()


async def loop():
    n, last = 0, time.perf_counter()
    while True:
        await asyncio.sleep(TICK_S)
        now = time.perf_counter()
        try:
            sim.step(now - last)
        except Exception as e:  # keep the demo alive; surface the error in the feed
            sim.emit("SYS", "Simulation error", repr(e), "bad")
        last = now
        n += 1
        if n % BROADCAST_EVERY == 0 and clients:
            msg = json.dumps({"type": "state", "data": sim.snapshot()}, default=list)
            for ws in list(clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    clients.discard(ws)


@asynccontextmanager
async def lifespan(_app):
    global sim
    sim = Simulation()
    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(title="resQnet edge orchestrator", version="0.1.0", lifespan=lifespan,
              description="AI-driven emergency connectivity & responder orchestration — prototype")
app.add_middleware(GZipMiddleware, minimum_size=2000)
_roads_cache = None


# ---------------------------------------------------------------- pages & feed
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.websocket("/ws")
async def ws_feed(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    await ws.send_text(json.dumps({"type": "init", "data": sim.static_info(), "state": sim.snapshot()}, default=list))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)


@app.get("/api/state")
async def state():
    return JSONResponse(json.loads(json.dumps(sim.snapshot(), default=list)))


@app.get("/api/info")
async def info():
    return sim.static_info()


@app.get("/api/roads")
async def roads():
    """Street network for the map's own vector layer (works without internet map tiles)."""
    global _roads_cache
    if _roads_cache is None:
        from .sim.roads import CLASS_RANK
        g = sim.graph
        _roads_cache = [[CLASS_RANK[g.cls[i]], [[round(p[0], 5), round(p[1], 5)] for p in geom]]
                        for i, geom in enumerate(g.geom)]
    return _roads_cache


@app.get("/api/results")
async def results():
    return sim.results()


@app.get("/api/audit")
async def audit():
    return {"verify": sim.audit.verify(), "entries": sim.audit.entries}


# ---------------------------------------------------------------- scenario control
class SpeedBody(BaseModel):
    speed: float


@app.post("/api/reset")
async def reset():
    sim.reset()
    return {"ok": True}


@app.post("/api/demo/start")
async def demo_start(auto: bool = True):
    sim.reset()
    sim.director = Director(sim, auto=auto)
    return {"ok": True}


@app.post("/api/demo/next")
async def demo_next():
    if sim.director:
        sim.director.next()
    return {"ok": True}


@app.post("/api/demo/stop")
async def demo_stop():
    sim.director = None
    return {"ok": True}


@app.post("/api/speed")
async def set_speed(b: SpeedBody):
    sim.speed = max(0.25, min(20.0, b.speed))
    return {"speed": sim.speed}


@app.post("/api/pause")
async def pause():
    sim.paused = not sim.paused
    return {"paused": sim.paused}


# ---------------------------------------------------------------- orchestrator actions
class IncidentBody(BaseModel):
    lat: float
    lon: float
    kind: str = "flood"
    people: int = 5
    injuries: int = 1
    critical: bool = False
    note: str = ""


@app.post("/api/incident")
async def incident(b: IncidentBody):
    inc = sim.create_incident((b.lat, b.lon), b.kind, b.people, b.injuries, b.critical, b.note, "dashboard")
    return inc


class ZoneBody(BaseModel):
    lat: float
    lon: float
    radius: int = 1000
    buffer: int = 500


@app.post("/api/zone")
async def zone(b: ZoneBody):
    sim.calls.declare_zone((b.lat, b.lon), b.radius, b.buffer)
    return {"ok": True}


@app.delete("/api/zone")
async def lift_zone():
    sim.calls.lift_zone()
    return {"ok": True}


class SmsBody(BaseModel):
    sender: str
    text: str
    spoofed: bool = False  # simulate a SIM that the network authenticates as a different number


@app.post("/api/sms")
async def sms(b: SmsBody):
    res = sim.sms(b.sender, b.text, network_msisdn="+91 90000 00000" if b.spoofed else "__same__")
    res.pop("checks", None)
    return res


class IvrBody(BaseModel):
    vid: str
    digit: str


@app.post("/api/ivr")
async def ivr(b: IvrBody):
    return {"ok": sim.ivr(b.vid, b.digit)}


class MedicBody(BaseModel):
    incident: str


@app.post("/api/medic")
async def medic(b: MedicBody):
    inc = sim.incident(b.incident)
    if not inc:
        return JSONResponse({"error": "incident not found"}, 404)
    req = sim.request_medic(inc, "dashboard")
    return req.summary()


class CallBody(BaseModel):
    lat: float
    lon: float
    dialed: str = "+91 99887 76655"
    reply: str = ""


@app.post("/api/call")
async def call(b: CallBody):
    rec = sim.calls.handle_call((b.lat, b.lon), b.dialed, reply=b.reply or "1", manual=True)
    return json.loads(json.dumps(rec, default=list))


@app.post("/api/cell/{cid}/fail")
async def fail_cell(cid: str):
    sim.fail_cell(cid)
    return {"ok": True}


@app.post("/api/cell/{cid}/restore")
async def restore_cell(cid: str):
    sim.restore_cell(cid)
    return {"ok": True}


class OverrideBody(BaseModel):
    action: str
    arg: Optional[str] = None


@app.post("/api/override")
async def override(b: OverrideBody):
    return {"applied": sim.override(b.action, b.arg)}


class ClassifyBody(BaseModel):
    text: str


@app.post("/api/classify")
async def classify(b: ClassifyBody):
    return sim.models.classify(b.text)


# ---------------------------------------------------------------- mock CAMARA network APIs (HTTP)
# Same request shapes as the CAMARA specs; X-Network-MSISDN stands in for the network-authenticated
# identity of the calling SIM (in a real network this comes from the operator, not the client).
camara_tags = ["Mock CAMARA network APIs"]


@app.post("/camara/number-verification/v0/verify", tags=camara_tags)
async def nv(body: dict, x_network_msisdn: Optional[str] = Header(None)):
    return sim.network.number_verification_verify(body, x_network_msisdn)


@app.post("/camara/sim-swap/v0/check", tags=camara_tags)
async def ss(body: dict):
    return sim.network.sim_swap_check(body)


@app.post("/camara/sim-swap/v0/retrieve-date", tags=camara_tags)
async def ss_date(body: dict):
    return sim.network.sim_swap_retrieve_date(body)


@app.post("/camara/population-density-data/v0.1/retrieve", tags=camara_tags)
async def pdd(body: dict):
    return sim.network.population_density_retrieve(body)


@app.post("/camara/quality-on-demand/v0/sessions", tags=camara_tags)
async def qod(body: dict):
    return sim.network.qod_create_session(body)


@app.delete("/camara/quality-on-demand/v0/sessions/{sid}", tags=camara_tags)
async def qod_del(sid: str):
    return sim.network.qod_delete_session(sid)


@app.post("/camara/location-retrieval/v0/retrieve", tags=camara_tags)
async def loc(body: dict):
    return sim.network.location_retrieve(body)


@app.post("/camara/geofencing-subscriptions/v0.3/subscriptions", tags=camara_tags)
async def geo(body: dict):
    return sim.network.geofencing_subscribe(body)
