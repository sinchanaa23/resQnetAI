# resQnet

**AI-driven emergency connectivity & responder orchestration network — functional prototype**

resQnet is an AI layer on top of existing telecom networks. It clears roads for ambulances, finds the
nearest available doctor, puts emergency calls first, and keeps critical messages flowing when the
network is congested or partly down.

This repository is the working proof of concept for the *Nokia Shark Tank Prelims — AI in Telecom*
proposal. It runs fully in simulation on one laptop acting as the MEC edge node: no live network
changes are needed. Every network action goes through **mock GSMA CAMARA APIs** that use the same
request/response shapes as the real specs, so it can later be pointed at an operator sandbox
(for example, Nokia Network as Code).

![features](https://img.shields.io/badge/features-F1–F6-ff4d4f) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![tests](https://img.shields.io/badge/tests-16%20passing-2ecc71)

---

## Quick start (about 1 minute)

```bash
pip install -r requirements.txt
python -m uvicorn backend.app:app --port 8000
```

Then open **http://localhost:8000** and press **▶ Hero demo**.

On Windows you can double-click `run.bat`; on macOS/Linux run `./run.sh`.
Or use Docker (the "edge node" container):

```bash
docker build -t resqnet . && docker run -p 8000:8000 resqnet
```

The dashboard needs **no internet connection**. Map streets are drawn from bundled OpenStreetMap
data and Leaflet is vendored, so it works even on poor venue Wi-Fi.

---

## The hero demo (≈ 3–4 minutes, proposal §7)

| Step | What the judges see | Feature |
|---|---|---|
| 0 | Monsoon evening, flooding in Ejipura (Koramangala, Bengaluru), cell sites at ~90 % load | — |
| 1 | A citizen dials **112** from the flooded area; the call is auto-prioritised and an incident is created with an AI severity score | F4 |
| 2 | The district authority declares a **1 km disaster zone + 500 m buffer**; emergency mode opens a priority slice (CAMARA Quality on Demand); other calls in the zone are prompted and classified (EN / HI / KN) | F4 |
| 3a | A **spoofed** route-clearance request using ambulance AMB-03's number from a SIM swapped 47 minutes ago is **blocked** (CAMARA SIM Swap) and audited | F1 |
| 3b | Verified ambulance AMB-07 sends `CLEAR ROUTE INC-2041` by SMS → AI route + ETA → crowd-inflow forecast → **moving warning corridor**; only the cells and users ahead are alerted | F1, F2 |
| 4 | Volunteer medics are ranked; the first **declines (presses 2)**, the system escalates, and the next, 1.4 km away, **accepts (presses 1)** | F3 |
| 5 | A flooded **cell site fails**; messages move to a neighbouring cell, an isolated SDRF boat team is reached by **D2D relay**, and a **temporary cell (COW)** comes on air | F5 |
| 6 | **Edge vs cloud** decision latency is shown live | F6 |
| 7 | Ambulance arrives → **results screen**: the same scenario with and without resQnet | all |

**Presenter controls:** `Step mode` lets you click **Next ▸** (or press → / `n`) for each step.
`Space` pauses, `r` opens results, and the speed selector changes simulation speed.
The dashboard switches to the relevant panel at each step on its own.

### Typical results from one run (simulated data)

| Metric | No resQnet | resQnet | Target |
|---|---|---|---|
| Ambulance travel time | 14.6 min | 10.4 min (−29 %) | ≥ 15 % shorter |
| Alert-delivery latency | — | 0.38 s | < 2 s |
| Route-clearance coverage | — | 87 % | ≥ 85 % |
| Alert precision | 2 % (blanket broadcast) | 86 % | ≥ 70 % |
| Responder selection time | — | 28 s | < 60 s |
| Emergency-call recall | 32 % keyword match | 97 % on held-out phrasings | ≥ 95 % |
| Emergency-message delivery at ~90 % load | ~80 % | 100 % | ≥ 98 % |
| Failover success after tower failure | 0 % | 100 % | ≥ 95 % |
| Edge vs cloud decision latency | ~148 ms | ~4 ms (≈ 39×) | edge ≥ 3× faster |

Reproduce headless: `python -m tools.headless_demo -v`

> These are prototype numbers from a simulation with synthetic data, not field claims. Key modelled
> assumptions are listed below so they can be challenged.

---

## Interactive mode (beyond the scripted demo)

Everything in the hero demo can also be triggered by hand:

- **Right-click the map** → create an incident, declare a disaster zone, or place a citizen call from that spot.
- **Right-click / click a cell site** (◆) → fail it or restore it.
- **Field devices panel** (bottom-left):
  - *Responder SMS*: send `CLEAR ROUTE INC-####` or `MEDIC INC-####` from a verified ambulance, a SIM-swapped
    number, a police officer (not allowed to clear routes), an unregistered number, or a spoofed SIM.
  - *Volunteer medic IVR*: when a doctor is being called, press **1** (accept) or **2** (decline) yourself;
    no answer within 45 s escalates automatically.
  - *Citizen phone*: shows the route alert a driver receives; place a call and answer the emergency prompt in
    English, Hindi or Kannada.
- **AI decisions** tab: every automated decision with its reasons and rejected alternatives, plus **human
  override** buttons (skip a medic, stop route alerts, lift the zone).
- **Network** tab: path usage, emergency slice, live call classification, edge-vs-cloud chart, and a **Cut cloud
  link** button (the edge keeps working).
- **AI models** tab: every model against its rule baseline, plus a live classifier box to try any sentence.
- **Security** tab: responder registry (revoke / reinstate credentials), a **hash-chained audit log** with live
  verification, and every CAMARA API call made.

---

## How each feature is implemented

| Feature | Where | What it does |
|---|---|---|
| **F1** Verified responder identity | `backend/orchestrator/security.py` | Registry + CAMARA Number Verification + SIM Swap (4 h window), role-based permissions, rate limits, instant revocation, SHA-256 hash-chained audit log |
| **F2** AI route clearing | `backend/orchestrator/route_clearing.py` | Edge ETA map from a gradient-boosted segment-ETA model → Dijkstra route → crowd-inflow forecast per cell (CAMARA Population Density) → greedy optimiser picks **Cell Broadcast vs targeted SMS + voice** per cell → corridor of the next ~4 min moves with the vehicle; no user is alerted twice; cells behind stop receiving alerts |
| **F3** Medic matching | `backend/orchestrator/matching.py` | Consent-based location (CAMARA Location Retrieval), expected-utility ranking (p(accept) × specialisation fit / ETA), IVR accept/decline, 45 s timeout → escalation, parallel mode for severity 5, geofence on arrival |
| **F4** Call prioritisation | `backend/orchestrator/calls.py` | Authority-declared zone + buffer, QoD emergency-slice session, instant priority for 112/108/101/100 and responder SIMs, multilingual prompt classifier where uncertain cases count as emergencies, auto-expiring emergency mode |
| **F5** Resilient connectivity | `backend/orchestrator/connectivity.py` | Contextual bandit ranks home cell → neighbour cell → temporary cell → satellite D2D → mesh/D2D relay per message; retries; COW dispatch after a site failure; counterfactual "no resQnet" delivery tracked for every message |
| **F6** Edge AI | `backend/orchestrator/security.py` (`LatencyTracker`) | Each decision's real compute time is measured on the edge; edge and cloud round trips are modelled |
| Orchestrator | `backend/engine.py` | Ties everything together; incident severity (rule + GBM hybrid); explainability log; human overrides |
| Mock CAMARA | `backend/camara/` | Number Verification, SIM Swap, Population Density Data, Quality on Demand, Location Retrieval, Geofencing — in-process **and** over HTTP at `/camara/...` (see `/docs`) |

### AI models (proposal §5), all trained at start-up on simulator data

| Module | Model | Baseline | Metric |
|---|---|---|---|
| Segment ETA (F2) | Gradient-boosted trees (histogram, depth 3) | Free-flow shortest path | MAE s/segment |
| Crowd inflow (F2) | Autoregressive + time-of-day regression | Persistence | MAE people/cell |
| Acceptance probability (F3) | Logistic regression | Distance-only | AUC |
| Responder ranking (F3) | Expected-utility ranking | Nearest-first | Escalation rounds |
| Emergency classifier (F4) | Naive Bayes on words + char n-grams + multilingual danger lexicon | Keyword match | Recall on **held-out phrasings** |
| Path selection (F5) | Contextual bandit (Beta posteriors) | Fixed fallback order | First-attempt delivery |
| Incident severity | Rule + gradient-boosting hybrid (critical ⇒ ≥ 4) | Rule table | Agreement with expert labels |

The ML is pure Python (no NumPy/scikit-learn), so the edge node installs anywhere in seconds. The
interfaces mirror scikit-learn, so they can be swapped for XGBoost / LightGBM / IndicBERT / MuRIL when
real data is available.

### Modelled assumptions (stated openly)

- Traffic: evening-peak speed factors by road class with per-segment variation; flooded segments run at 25–55 % speed.
- Route clearance: a road segment warned at least 45 s ahead recovers 45 % of the gap to 85 % of free-flow speed (drivers yield or avoid the road). Flood slow-down is not removed.
- Ambulance turnout (crew boarding) is 40 s; the corridor ahead is warned during it. Travel times compare drive time only.
- Congested cell at ~90 % load: 78 % delivery without priority, 99.5 % on the emergency slice. Neighbour cell 93 %, temporary cell 97 %, satellite 90 %, D2D relay 86 %.
- Cloud-only decisions: 45–70 ms each way over a congested backhaul; edge: 2–4 ms round trip. Compute time is measured, not modelled.
- All people, phone numbers and responders are fictional. Streets are real (© OpenStreetMap contributors, ODbL).

---

## Architecture

```
 Inputs                     Edge node (MEC)                         Network APIs (CAMARA)        Delivery paths
 ───────                    ─────────────────────────────           ──────────────────────       ──────────────
 112 calls / ERSS  ─┐       AI Emergency Orchestrator               Number Verification          Cell Broadcast
 Responder SMS/app ─┼──►    F1 identity · F2 route clearing   ◄──►  SIM Swap                     Targeted SMS / IVR
 Volunteer registry ┤       F3 medic matching · F4 call priority    Population Density           Neighbour cell
 Vehicle GPS        ┤       F5 connectivity · severity scoring      Quality on Demand (slice)    Temporary cell (COW)
 Network telemetry ─┘       explainability · audit · overrides      Location / Geofencing        Satellite / D2D relay
                                     ▲            │
                                     │ models     │ analytics
                                     └── Cloud: training, history ──┘
```

```
resQnet/
├── backend/
│   ├── app.py               FastAPI: REST, WebSocket feed, mock CAMARA HTTP endpoints
│   ├── engine.py            simulation loop + orchestrator glue + results
│   ├── demo.py              hero-demo director (7 steps)
│   ├── orchestrator/        F1–F6 decision logic
│   ├── camara/              mock CAMARA network + client (mock | http)
│   ├── ml/                  pure-Python models + synthetic training data
│   ├── sim/                 road graph, traffic model, cells, responders, volunteers
│   └── data/roads.json      Koramangala / HSR Layout street graph from OpenStreetMap
├── frontend/                command dashboard (Leaflet, vanilla JS, vendored for offline use)
├── tests/                   16 tests: per feature, API, and full hero demo vs targets
└── tools/                   build_roads.py (refresh OSM data), headless_demo.py
```

### Pointing at a real operator sandbox

```bash
set RESQNET_CAMARA_MODE=http
set RESQNET_CAMARA_BASE_URL=https://<operator-camara-gateway>/
set RESQNET_CAMARA_TOKEN=<token>
```

The orchestrator's calls go through `backend/camara/client.py`, which sends the same request bodies to
the real endpoints. The simulator still provides the world state for the demo.

---

## Tests

```bash
python -m pytest -q
```

They cover F1 (SIM swap, role, spoofed SIM blocked; verified allowed), audit-chain tamper detection, F2 corridor
and precision, F3 decline / timeout escalation, F4 zone priority and classification, F5 failover and COW, F6
latency ratio, the HTTP API and CAMARA endpoints, and the full hero demo meeting every proposal target.

## Roadmap (proposal §14)

Sandbox integration (3–6 months) → campus or district pilot with a partner operator (6–12 months) →
smart-city deployment with 112 ERSS and traffic-signal control (12–24 months) → multi-operator national scale.

## Privacy & compliance by design

Citizens are only ever counted as aggregated, anonymised density per cell. Targeted alerts are sent by the
operator inside its own network, and resQnet never sees citizens' numbers or locations. Volunteer location is
consent-based and requested only while they are marked available. Emergency mode is authority-triggered,
zone-limited, auto-expiring and audited (DPDP Act 2023; DoT/TRAI emergency exceptions).
