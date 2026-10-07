/* resQnet command dashboard — vanilla JS + Leaflet, fed by the edge node's WebSocket. */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x, d = 0) => (x == null ? "—" : (x * 100).toFixed(d) + "%");
const mins = (s) => (s == null ? "—" : s >= 60 ? (s / 60).toFixed(1) + " min" : Math.round(s) + " s");

let S = null;            // latest snapshot
let INFO = null;         // static info
let lastEventId = 0;
let lastDecisionId = 0;
let feedFilter = "all";
let lastStoryStep = -1;
let resultsShown = false;
let activeTab = "feed";

async function api(path, body, method) {
  const opt = { method: method || (body ? "POST" : "GET"), headers: { "Content-Type": "application/json" } };
  if (body) opt.body = JSON.stringify(body);
  const r = await fetch(path, opt);
  return r.json();
}

/* ------------------------------------------------------------------ map */
const map = L.map("map", { zoomControl: true, preferCanvas: true, attributionControl: true }).setView([12.928, 77.627], 15);
// Explicit panes keep the stacking order stable: streets < cells < zone/flood < routes < failover links < markers.
const pane = (name, z) => { map.createPane(name).style.zIndex = z; return name; };
const canvas = L.canvas({ padding: 0.3, pane: pane("streets", 350) });
const cellCanvas = L.canvas({ padding: 0.3, pane: pane("cellsP", 360) });
const zoneSvg = L.svg({ pane: pane("zoneP", 370) });
const routeSvg = L.svg({ pane: pane("routeP", 390) });
const linkSvg = L.svg({ pane: pane("linkP", 395) });
// No tile server: the street network is drawn from the edge node's own OSM road graph, so the map works offline.
map.attributionControl.addAttribution("Streets © OpenStreetMap contributors (ODbL)");

const L_roads = L.layerGroup().addTo(map);
const L_cells = L.layerGroup().addTo(map);
const L_flood = L.layerGroup().addTo(map);
const L_zone = L.layerGroup().addTo(map);
const L_route = L.layerGroup().addTo(map);
const L_links = L.layerGroup().addTo(map);
const L_people = L.layerGroup().addTo(map);
const L_labels = L.layerGroup().addTo(map);

api("/api/roads").then((roads) => {
  const style = [
    { color: "#2a3a50", weight: 1 }, { color: "#3a4e6a", weight: 1.6 }, { color: "#4a6283", weight: 2.2 },
    { color: "#5a769d", weight: 3 }, { color: "#6886ad", weight: 3.6 },
  ];
  for (const [rank, pts] of roads) L.polyline(pts, { ...style[rank], renderer: canvas, interactive: false }).addTo(L_roads);
});

const divIcon = (cls, html, size) => L.divIcon({ className: "", html: `<div class="mk ${cls}">${html}</div>`, iconSize: [size, size], iconAnchor: [size / 2, size / 2] });
const markers = {};       // key -> marker
const cellShapes = {};    // id -> {circle, marker}
function upsertMarker(key, pos, icon, tip, layer = L_people) {
  let m = markers[key];
  if (!m) {
    m = markers[key] = L.marker(pos, { icon }).addTo(layer);
    if (tip) m.bindTooltip(tip, { direction: "top", offset: [0, -10] });
    m._iconKey = icon.options.html;
  } else {
    m.setLatLng(pos);
    if (m._iconKey !== icon.options.html) { m.setIcon(icon); m._iconKey = icon.options.html; }
    if (tip) m.setTooltipContent(tip);
  }
  m._seen = true;
  return m;
}

function drawStatic() {
  L_flood.clearLayers();
  const f = INFO.flood;
  L.circle(f.center, { renderer: zoneSvg, radius: f.radius, color: "#4c8dff", weight: 1, fillColor: "#4c8dff", fillOpacity: 0.16, dashArray: "4 4", interactive: false }).addTo(L_flood);
  L.circle(f.center, { renderer: zoneSvg, radius: f.radius * 0.35, stroke: false, fillColor: "#4c8dff", fillOpacity: 0.22, interactive: false }).addTo(L_flood);
  L.marker(f.center, { icon: L.divIcon({ className: "", html: '<div class="lbl" style="border-color:#4c8dff;color:#9cc0ff">≋ Flooded · Ejipura</div>', iconAnchor: [50, -14] }), interactive: false }).addTo(L_flood);
  const h = INFO.hospital;
  L.marker(h.pos, { icon: divIcon("mk-hosp", "H", 22) }).bindTooltip(h.name).addTo(L_flood);
}

function drawCells() {
  const alerted = {};
  const corridorCells = new Set();
  for (const t of S.trips) {
    for (const a of t.alerted) alerted[a.cell] = a;
    for (const c of t.corridor_cells) corridorCells.add(c);
  }
  for (const c of S.cells) {
    let sh = cellShapes[c.id];
    const a = alerted[c.id];
    const st = c.status === "down" ? "down" : c.temp ? (c.status === "deploying" ? "temp deploying" : "temp") : "";
    let color = "#2c3c52", fill = "#000", fo = 0, w = 0.6, dash = null;
    if (c.status === "down") { color = "#ff4d4f"; fill = "#ff4d4f"; fo = 0.13; w = 1.6; dash = "5 4"; }
    else if (c.temp) { color = "#b37cff"; fill = "#b37cff"; fo = c.status === "deploying" ? 0.03 : 0.12; w = 1.4; dash = c.status === "deploying" ? "3 5" : null; }
    else if (a && a.state === "active") { color = "#ff9f1c"; fill = "#ff9f1c"; fo = corridorCells.has(c.id) ? 0.2 : 0.1; w = 1.4; }
    else if (a) { color = "#7a5a2c"; fill = "#ff9f1c"; fo = 0.04; w = 0.8; }
    const tip = `<b>${c.id}</b>${c.temp ? " (temporary cell)" : ""}<br>status: ${c.status}<br>load: ${pct(c.load)}<br>devices: ${c.devices.toLocaleString()}` +
      (a ? `<br><span style="color:#ff9f1c">alerted · ${a.channel}<br>${a.alerted.toLocaleString()} phones · lead ${Math.round(a.eta_at_alert)}s</span>` : "");
    if (!sh) {
      const circle = L.circle(c.pos, { radius: c.r, renderer: cellCanvas, interactive: false }).addTo(L_cells);
      const marker = L.marker(c.pos, { icon: divIcon("mk-cell " + st, "", 12) }).addTo(L_cells);
      marker.bindTooltip(tip);
      marker.on("click", (e) => cellMenu(c.id, e.originalEvent));
      sh = cellShapes[c.id] = { circle, marker, st: null };
    }
    sh.circle.setStyle({ color, weight: w, fillColor: fill, fillOpacity: fo, dashArray: dash, opacity: c.status === "up" && !a && !c.temp ? 0.35 : 0.9 });
    if (sh.st !== st) { sh.marker.setIcon(divIcon("mk-cell " + st, "", 12)); sh.st = st; }
    sh.marker.setTooltipContent(tip);
  }
}

let routeKey = "";
let fitted = {};
function drawRoutesAndZone() {
  const key = S.trips.map((t) => t.id + ":" + Math.round(t.d / 25)).join("|") + (S.zone ? "Z" : "") + S.medics.map((m) => m.arrived + ":" + (m.route ? m.route.length : 0)).join();
  if (key !== routeKey) {
    routeKey = key;
    L_route.clearLayers();
    L_zone.clearLayers();
    if (S.zone) {
      L.polygon(S.zone.buffer_poly, { renderer: zoneSvg, color: "#ff4d4f", weight: 1, opacity: 0.6, dashArray: "2 6", fillOpacity: 0.03, interactive: false }).addTo(L_zone);
      L.polygon(S.zone.poly, { renderer: zoneSvg, color: "#ff4d4f", weight: 2, dashArray: "8 6", fillColor: "#ff4d4f", fillOpacity: 0.05, interactive: false }).addTo(L_zone);
    }
    for (const t of S.trips) {
      const R = { renderer: routeSvg, interactive: false };
      L.polyline(t.baseline_route, { ...R, color: "#c3cddb", weight: 3, opacity: 0.75, dashArray: "6 7" }).addTo(L_route);
      L.polyline(t.route, { ...R, color: "#38bdf8", weight: 5, opacity: 0.8 }).addTo(L_route);
      if (t.corridor && t.corridor.length > 1) {
        L.polyline(t.corridor, { ...R, color: "#ff9f1c", weight: 18, opacity: 0.25 }).addTo(L_route);
        L.polyline(t.corridor, { ...R, color: "#ffb347", weight: 7, opacity: 1 }).addTo(L_route);
      }
      if (!fitted[t.id]) {
        fitted[t.id] = true;
        map.fitBounds(L.latLngBounds(t.route.concat(t.baseline_route, [S.incidents[0] ? S.incidents[0].pos : t.route[0]])).pad(0.05), { paddingTopLeft: [300, 110], paddingBottomRight: [170, 80] });
      }
    }
    for (const m of S.medics) if (m.route) L.polyline(m.route, { renderer: routeSvg, color: "#2ecc71", weight: 3, dashArray: "4 6", opacity: 0.95, interactive: false }).addTo(L_route);
  }
  // Failover links
  L_links.clearLayers();
  const col = { neighbour_cell: "#ffd166", temp_cell: "#b37cff", satellite: "#38bdf8", d2d_relay: "#ff6fb5" };
  for (const l of S.links) {
    if (!l.via) continue;
    const pts = Array.isArray(l.via[0]) ? [l.from, ...l.via] : [l.from, l.via];
    L.polyline(pts, { renderer: linkSvg, color: col[l.path] || "#fff", weight: 2.5, dashArray: "3 6", opacity: 0.95, interactive: false }).addTo(L_links);
  }
}

function drawPeople() {
  for (const k in markers) markers[k]._seen = false;
  for (const inc of S.incidents) {
    upsertMarker("inc" + inc.id, inc.pos, divIcon("mk-inc", "!", 26),
      `<b>${inc.id}</b> · severity ${inc.severity}/5<br>${esc(inc.type)} · ${inc.people} people · ${inc.injuries} injured<br>${esc(inc.note)}<br>status: ${esc(inc.status)}`);
  }
  const roleIcon = { ambulance: "🚑", fire: "🚒", police: "🚓", rescue: "⛵" };
  for (const r of S.responders) {
    const trip = S.trips.find((t) => t.vehicle === r.id && !t.finished);
    const icon = trip ? divIcon("mk-amb", "🚑", 30) : divIcon("mk-res", roleIcon[r.role] || "•", 22);
    upsertMarker("r" + r.id, r.pos, icon, `<b>${r.id}</b> · ${esc(r.name)}${r.revoked ? " · REVOKED" : ""}`);
  }
  for (const v of S.volunteers) {
    const st = !v.available ? "unavailable" : v.assigned && v.state === "idle" ? "assigned" : v.state;
    upsertMarker("v" + v.id, v.pos, divIcon("mk-vol " + st, "✚", st === "accepted" || st === "en_route" || st === "arrived" ? 24 : 18),
      `<b>${esc(v.name)}</b><br>${esc(v.spec)}<br>${st.replace("_", " ")}`);
  }
  for (const k in markers) if (!markers[k]._seen && !k.startsWith("lbl")) { markers[k].remove(); delete markers[k]; }
}

/* ------------------------------------------------------------------ context menus */
const ctx = $("ctxMenu");
function showMenu(ev, items) {
  ctx.innerHTML = "";
  for (const [label, fn] of items) {
    const b = document.createElement("button");
    b.textContent = label;
    b.onclick = () => { hideMenu(); fn(); };
    ctx.appendChild(b);
  }
  ctx.style.left = Math.min(ev.clientX, innerWidth - 240) + "px";
  ctx.style.top = Math.min(ev.clientY, innerHeight - 200) + "px";
  ctx.classList.remove("hidden");
}
const hideMenu = () => ctx.classList.add("hidden");
document.addEventListener("click", (e) => { if (!ctx.contains(e.target)) hideMenu(); });

map.on("contextmenu", (e) => {
  const { lat, lng } = e.latlng;
  showMenu(e.originalEvent, [
    ["⚠ Create incident here (flood, critical)", () => api("/api/incident", { lat, lon: lng, kind: "flood", people: 6, injuries: 2, critical: true, note: "reported via dashboard" })],
    ["⚠ Create incident here (road accident)", () => api("/api/incident", { lat, lon: lng, kind: "accident", people: 3, injuries: 2, critical: false })],
    ["◎ Declare 1 km disaster zone here", () => api("/api/zone", { lat, lon: lng, radius: 1000, buffer: 500 })],
    ["☎ Place citizen call from here", () => placeCall(lat, lng)],
  ]);
});

function cellMenu(cid, ev) {
  L.DomEvent.stop(ev);
  const c = S.cells.find((x) => x.id === cid);
  showMenu(ev, c && c.status === "down"
    ? [[`Restore ${cid}`, () => api(`/api/cell/${cid}/restore`, {})]]
    : [[`✖ Fail cell site ${cid}`, () => api(`/api/cell/${cid}/fail`, {})]]);
}

/* ------------------------------------------------------------------ header / story / KPIs */
function renderHeader() {
  $("clock").textContent = S.clock;
  $("load").textContent = pct(S.load);
  $("loadbar").style.width = pct(S.load);
  const em = S.calls.slice_active;
  $("emode").textContent = em ? "ON · slice" : "OFF";
  $("emode").className = "pill " + (em ? "on" : "off");
  $("edgems").textContent = S.latency.n ? S.latency.edge_ms + " ms" : "-- ms";
  $("btnPause").textContent = S.paused ? "▶" : "❚❚";
  $("speed").value = String(S.speed);
}

const STEP_TAB = { F4: "network", F1: "security", "F1 · F2": "explain", F3: "explain", F5: "network", F6: "network", ALL: "feed" };
function renderStory() {
  const d = S.director;
  $("story").classList.toggle("hidden", !d);
  $("btnNext").classList.toggle("hidden", !d || d.auto || d.done);
  if (!d) return;
  $("storySteps").innerHTML = d.steps.map((_, i) => `<i class="${i < d.i ? "done" : i === d.i ? "cur" : ""}"></i>`).join("");
  $("storyTitle").textContent = d.title;
  $("storyCaption").textContent = d.caption + (d.waiting_next ? "  ▸ click Next" : "");
  const f = d.f.split(" ")[0];
  $("storyF").className = "ftag " + (f === "SIM" ? "SYS" : f);
  $("storyF").textContent = d.f;
  if (d.i !== lastStoryStep) {
    lastStoryStep = d.i;
    const tab = STEP_TAB[d.f];
    if (tab) selectTab(tab, true);
    if (d.f === "F6") $("latH").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  if (d.done && !resultsShown) { resultsShown = true; setTimeout(openResults, 1200); }
}

function renderKpis() {
  const t = S.trips[0];
  if (t) {
    $("kAmb").textContent = t.finished ? "on scene" : "ETA " + mins(t.eta_s);
    $("kAmbS").textContent = t.finished ? `${mins(t.elapsed_s)} vs ${mins(t.baseline_time_s)} baseline` : `${t.vehicle} · ${Math.round(t.speed_kmh)} km/h · ${(t.progress * 100).toFixed(0)}%`;
    $("kAlerts").textContent = t.people_alerted.toLocaleString();
    $("kAlertsS").textContent = `${t.cells_alerted} cells · precision ${pct(t.precision)} · coverage ${pct(t.coverage)}`;
  } else { $("kAmb").textContent = "standby"; $("kAmbS").innerHTML = "&nbsp;"; $("kAlerts").textContent = "0"; $("kAlertsS").innerHTML = "&nbsp;"; }
  const m = S.medics[0];
  if (m) {
    $("kMedic").textContent = m.assigned ? (m.arrived ? "on scene" : "ETA " + mins(m.eta_s)) : m.ringing.length ? "calling…" : "searching";
    $("kMedicS").textContent = m.assigned ? `${m.assigned} · picked in ${Math.round(m.selection_s)}s` : `${m.contacted} contacted`;
  } else { $("kMedic").textContent = "—"; $("kMedicS").innerHTML = "&nbsp;"; }
  const c = S.calls;
  $("kCalls").textContent = c.emergencies || 0;
  $("kCallsS").textContent = c.recall == null ? (c.zone ? "monitoring zone" : "no zone declared") : `recall ${pct(c.recall)} · ${c.total} calls`;
  const n = S.conn;
  $("kDeliv").textContent = n.sent ? pct(n.success, 1) : "—";
  $("kDelivS").textContent = n.sent ? `baseline ${pct(n.baseline_success, 0)} · ${n.sent} msgs` : " ";
  const l = S.latency;
  $("kEdge").textContent = l.n ? `${l.edge_ms} ms` : "—";
  $("kEdgeS").textContent = l.n ? (l.cloud_link ? `cloud ${l.cloud_ms} ms · ${l.ratio}× faster` : "cloud link CUT · edge only") : " ";
}

/* ------------------------------------------------------------------ phones */
function currentIncident() { return S && S.incidents.length ? S.incidents[S.incidents.length - 1].id : "INC-2041"; }

$("smsSend").onclick = async () => {
  const r = await api("/api/sms", { sender: $("smsFrom").value, text: $("smsText").value, spoofed: $("smsSpoof").checked });
  $("smsReply").textContent = r.reply;
  $("smsReply").className = "bubble " + (r.ok ? "ok" : "bad");
};
document.querySelectorAll("[data-sms]").forEach((b) => (b.onclick = () => ($("smsText").value = `${b.dataset.sms} ${currentIncident()}`)));
document.querySelectorAll("[data-reply]").forEach((b) => (b.onclick = () => ($("callReply").value = b.dataset.reply)));
$("ivr1").onclick = () => S.ringing && api("/api/ivr", { vid: S.ringing.vid, digit: "1" });
$("ivr2").onclick = () => S.ringing && api("/api/ivr", { vid: S.ringing.vid, digit: "2" });
$("callSend").onclick = () => {
  const p = S.zone ? S.zone.center : S.incidents.length ? S.incidents[0].pos : [12.9356, 77.6268];
  placeCall(p[0] + (Math.random() - 0.5) * 0.004, p[1] + (Math.random() - 0.5) * 0.004);
};
async function placeCall(lat, lon) {
  const r = await api("/api/call", { lat, lon, dialed: $("callNum").value, reply: $("callReply").value });
  const where = r.where === "outside" ? "outside any declared zone" : `inside ${r.where}`;
  $("callRes").innerHTML = `<b>${r.priority ? "PRIORITY" : "normal"}</b> · ${where}<br>` +
    (r.route === "classified" ? `Prompt → “${esc(r.reply)}” → p(emergency) ${r.p}<br>` : "") + `<span class="muted">${esc(r.why)}</span>`;
  $("callRes").className = "bubble " + (r.priority ? "ok" : "");
}
$("phonesToggle").onclick = () => { $("phones").classList.toggle("collapsed"); $("phonesToggle").textContent = $("phones").classList.contains("collapsed") ? "+" : "–"; };

function renderPhones() {
  const r = S.ringing;
  $("ivrRing").classList.toggle("hidden", !r);
  const med = S.medics.find((m) => m.assigned);
  $("ivrIdle").classList.toggle("hidden", !!r || !!med);
  $("ivrDone").classList.toggle("hidden", !!r || !med);
  if (r) {
    $("ivrName").textContent = "Calling " + r.name + "…";
    $("ivrText").textContent = "“" + r.text + "”";
    $("ivrAuto").textContent = r.scripted ? "Hero demo: this doctor's answer is scripted — you can still press a key." : `Ringing ${Math.round(r.since)}s · auto-escalates after 45s`;
  } else if (med) {
    $("ivrDone").textContent = `${med.assigned} accepted · ${med.arrived ? "on scene" : "ETA " + mins(med.eta_s)}`;
  }
  const alert = S.sample_alert && S.trips.some((t) => !t.finished);
  $("cbs").classList.toggle("hidden", !alert);
  if (alert) $("cbsText").textContent = S.sample_alert;
}

/* ------------------------------------------------------------------ side panel */
function selectTab(name, flash) {
  activeTab = name;
  document.querySelectorAll("#tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
    if (flash && b.dataset.tab === name) { b.classList.remove("flash"); void b.offsetWidth; b.classList.add("flash"); }
  });
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + name));
  renderTabs(true);
}
document.querySelectorAll("#tabs button").forEach((b) => (b.onclick = () => selectTab(b.dataset.tab)));
document.querySelectorAll("#feedFilters button").forEach((b) => (b.onclick = () => {
  feedFilter = b.dataset.f;
  document.querySelectorAll("#feedFilters button").forEach((x) => x.classList.toggle("on", x === b));
  lastEventId = 0; $("feed").innerHTML = ""; renderFeed();
}));

function renderFeed() {
  const box = $("feed");
  const fresh = S.events.filter((e) => e.id > lastEventId);
  if (!fresh.length) return;
  lastEventId = fresh[fresh.length - 1].id;
  for (const e of fresh) {
    if (feedFilter !== "all" && e.f !== feedFilter) continue;
    const d = document.createElement("div");
    d.className = "ev " + e.level;
    d.innerHTML = `<div class="h"><span class="t">${e.clock}</span><span class="ftag ${e.f}">${e.f}</span><span class="ti">${esc(e.title)}</span></div>` +
      (e.detail ? `<div class="d">${esc(e.detail)}</div>` : "");
    box.prepend(d);
  }
  while (box.children.length > 150) box.lastChild.remove();
}

function renderDecisions(force) {
  const ds = S.decisions;
  if (!ds.length) { $("decisions").innerHTML = '<p class="muted">Every automated decision appears here with its reasons and rejected alternatives.</p>'; return; }
  const top = ds[ds.length - 1].id;
  if (top === lastDecisionId && !force) return;
  lastDecisionId = top;
  $("decisions").innerHTML = ds.slice().reverse().map((d) =>
    `<div class="dec"><div class="h"><span class="ftag ${d.f}">${d.f}</span><b>${esc(d.title)}</b><span class="ms">${d.ms} ms</span></div>` +
    `<ul>${d.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` +
    (d.alternatives.length ? `<div class="alt">Not chosen: ${d.alternatives.map((a) => `<span>${esc(a)}</span>`).join(" · ")}</div>` : "") +
    `<div class="alt">${d.clock}</div></div>`).join("");
}
document.querySelectorAll("[data-ov]").forEach((b) => (b.onclick = () => {
  const arg = b.dataset.ov === "skip_candidate" && S.ringing ? S.ringing.vid : null;
  api("/api/override", { action: b.dataset.ov, arg });
}));

const PATH_LABEL = { cellular: "Home cell", neighbour_cell: "Neighbour cell", temp_cell: "Temporary cell", satellite: "Satellite D2D", d2d_relay: "Mesh / D2D relay" };
const PATH_COL = { cellular: "#4c8dff", neighbour_cell: "#ffd166", temp_cell: "#b37cff", satellite: "#38bdf8", d2d_relay: "#ff6fb5" };
function renderNetwork() {
  const n = S.conn;
  $("nDel").textContent = n.sent ? pct(n.success, 1) : "—";
  $("nBase").textContent = n.sent ? pct(n.baseline_success, 1) : "—";
  $("nFail").textContent = n.failover_success == null ? "—" : pct(n.failover_success, 0);
  $("nSent").textContent = n.sent;
  const tot = Math.max(1, Object.values(n.by_path).reduce((a, b) => a + b, 0));
  $("paths").innerHTML = Object.entries(n.by_path).map(([k, v]) =>
    `<div class="hb"><span>${PATH_LABEL[k]}</span><div class="tr"><i style="width:${(100 * v) / tot}%;background:${PATH_COL[k]}"></i></div><span class="v">${v}</span></div>`).join("");
  const c = S.calls;
  $("slice").innerHTML = c.zone
    ? `<div class="hb"><span>Emergency slice</span><div class="tr"><i style="width:${c.slice_share * 100}%;background:#ff4d4f"></i></div><span class="v">${c.slice_share * 100}%</span></div>
       <div class="hb"><span>Best-effort (others)</span><div class="tr"><i style="width:${(1 - c.slice_share) * 100}%;background:#4c8dff"></i></div><span class="v">${(1 - c.slice_share) * 100}%</span></div>
       <div class="muted" style="margin-top:4px">QoD session <b>${c.qod}</b> · ${c.total} calls · ${c.auto} auto-priority · ${c.prompted} prompted ·
       recall ${pct(c.recall)} · ${c.false_alarms} false alarms · emergency set-up ${c.wait == null ? "—" : c.wait.toFixed(1) + "s"} vs ${c.wait_base == null ? "—" : c.wait_base.toFixed(1) + "s"} baseline</div>`
    : '<div class="muted">No disaster zone declared. Right-click the map → “Declare 1 km disaster zone here”.</div>';
  $("callFeed").innerHTML = S.call_feed.slice().reverse().map((x) =>
    `<tr><td class="mono">${fmtT(x.t)}</td><td class="mono">${esc(x.dialed.length > 6 ? x.dialed.slice(0, 7) + "•••" : x.dialed)}</td><td>${x.where}</td>` +
    `<td><span class="tagp ${x.priority ? "p" : "n"}">${x.priority ? "PRIORITY" : "normal"}</span> ${x.route === "classified" ? `<small class="muted">p=${x.p}</small>` : ""}</td>` +
    `<td>${x.reply ? "“" + esc(x.reply.slice(0, 40)) + "” · " : ""}${esc(x.why)}</td></tr>`).join("");
  const l = S.latency;
  $("lEdge").textContent = l.n ? l.edge_ms + " ms" : "—";
  $("lCloud").textContent = l.n ? (l.cloud_link ? l.cloud_ms + " ms" : "unreachable") : "—";
  $("btnCloud").textContent = l.cloud_link === false ? "Restore cloud link" : "Cut cloud link";
  drawLatChart(l.recent || []);
  $("cellsList").innerHTML = S.cells.map((c) =>
    `<div class="cellc ${c.status === "down" ? "down" : c.temp ? "temp" : ""}" data-cell="${c.id}"><span>${c.id}</span><span>${c.status === "down" ? "DOWN" : pct(c.load)}</span></div>`).join("");
  document.querySelectorAll("[data-cell]").forEach((el) => (el.onclick = (e) => cellMenu(el.dataset.cell, e)));
}
function fmtT(t) {
  const h = 19 + 5 / 60 + t / 3600;
  return `${String(Math.floor(h)).padStart(2, "0")}:${String(Math.floor((h % 1) * 60)).padStart(2, "0")}:${String(Math.floor((h * 3600) % 60)).padStart(2, "0")}`;
}
function drawLatChart(rs) {
  const svg = $("latChart");
  if (!rs.length) { svg.innerHTML = ""; return; }
  const max = Math.max(...rs.map((r) => r.cloud || r.edge), 10);
  const w = 400 / rs.length;
  let h = "";
  rs.forEach((r, i) => {
    const ch = r.cloud ? (100 * r.cloud) / max : 0, eh = Math.max(2, (100 * r.edge) / max);
    if (ch) h += `<rect x="${i * w + 1}" y="${110 - ch}" width="${w / 2 - 1}" height="${ch}" fill="#3a4e6a"><title>${r.kind}: cloud ${r.cloud} ms</title></rect>`;
    h += `<rect x="${i * w + w / 2}" y="${110 - eh}" width="${w / 2 - 1}" height="${eh}" fill="#ffd166"><title>${r.kind}: edge ${r.edge} ms</title></rect>`;
  });
  h += `<text x="6" y="14" fill="#8a9bb2" font-size="10">grey: cloud-only round trip · yellow: edge (MEC) · last ${rs.length} decisions</text>`;
  svg.innerHTML = h;
}
$("btnCloud").onclick = () => api("/api/override", { action: "cloud_link" });

let modelsDrawn = false;
function renderModels() {
  if (modelsDrawn || !INFO) return;
  modelsDrawn = true;
  $("trainS").textContent = INFO.train_seconds;
  $("models").innerHTML = Object.values(INFO.model_report).map((m) => {
    const lo = m.lower_is_better, mx = Math.max(m.baseline, m.ai) || 1;
    const fmt = (v) => (m.metric.includes("MAE") || m.metric.includes("rounds") ? v : (v * 100).toFixed(1) + "%");
    return `<div class="model"><div class="h"><span class="ftag ${m.feature === "All" ? "ALL" : m.feature}">${m.feature}</span><b>${esc(m.task)}</b>
      <span class="gain">${m.improvement_pct > 0 ? "▲" : ""} ${Math.abs(m.improvement_pct)}% better</span></div>
      <div class="sub">${esc(m.model)} · ${esc(m.metric)}${lo ? " (lower is better)" : ""}${m.note ? " · " + esc(m.note) : ""}</div>
      <div class="cmp"><span class="muted">${esc(m.baseline_name)}</span><div class="tr"><i style="width:${(100 * m.baseline) / mx}%;background:#56637a"></i></div><span class="v">${fmt(m.baseline)}</span></div>
      <div class="cmp"><span>resQnet AI</span><div class="tr"><i style="width:${(100 * m.ai) / mx}%;background:#2ecc71"></i></div><span class="v">${fmt(m.ai)}</span></div>
      ${m.precision != null ? `<div class="sub" style="margin-top:4px">precision ${(m.precision * 100).toFixed(1)}% (keyword baseline ${(m.baseline_precision * 100).toFixed(1)}%)</div>` : ""}</div>`;
  }).join("");
}
$("clfBtn").onclick = async () => {
  const r = await api("/api/classify", { text: $("clfText").value });
  $("clfOut").innerHTML = `<b>${r.label.toUpperCase()}</b> · p(emergency) = ${r.p}${r.uncertain ? " · uncertain → treated as emergency" : ""}<br><span class="muted">signals: ${esc((r.why || []).join(", ") || "—")}</span>`;
  $("clfOut").className = "bubble " + (r.label === "emergency" ? "bad" : "ok");
};

function renderSecurity() {
  $("registry").innerHTML = S.responders.map((r) =>
    `<tr class="${r.revoked ? "rev" : ""}"><td class="mono">${r.id}</td><td>${r.role}</td><td class="mono">${r.msisdn}</td>
     <td>${r.sim_swap_min < 240 ? `<span class="pill bad">${r.sim_swap_min} min ago</span>` : '<span class="muted">none recent</span>'}</td>
     <td><button data-rev="${r.id}" data-on="${r.revoked}">${r.revoked ? "Reinstate" : "Revoke"}</button></td></tr>`).join("");
  document.querySelectorAll("[data-rev]").forEach((b) => (b.onclick = () => api("/api/override", { action: b.dataset.on === "true" ? "reinstate" : "revoke", arg: b.dataset.rev })));
  const a = S.audit;
  $("chain").textContent = a.valid ? `chain valid · ${a.count} entries` : `BROKEN at ${a.broken_at}`;
  $("chain").className = "pill " + (a.valid ? "ok" : "bad");
  $("audit").innerHTML = a.recent.slice().reverse().map((e) =>
    `<div><span>#${e.seq}</span><span><b class="${e.outcome}">${esc(e.outcome)}</b> ${esc(e.action)} · ${esc(e.actor)}<br><span class="muted">${esc(e.detail).slice(0, 120)}</span></span><span class="hash">${e.hash.slice(0, 10)}</span></div>`).join("");
  $("camMode").textContent = INFO ? `mode: ${INFO.camara_mode}` : "";
  $("apis").innerHTML = S.api_calls.slice().reverse().map((c) =>
    `<div><span class="muted">${c.clock}</span><span><span class="ep">POST /${esc(c.api)}</span><br>${esc(c.summary)}</span></div>`).join("");
}

function renderTabs(force) {
  if (!S) return;
  renderFeed();
  if (activeTab === "explain") renderDecisions(force);
  if (activeTab === "network") renderNetwork();
  if (activeTab === "models") renderModels();
  if (activeTab === "security") renderSecurity();
}

/* ------------------------------------------------------------------ results */
async function openResults() {
  const r = await api("/api/results");
  $("resNote").textContent = `Simulated time ${r.clock} · ${r.note}`;
  $("resBody").innerHTML = r.rows.length ? r.rows.map((x) =>
    `<tr><td><b>${esc(x.metric)}</b></td><td class="muted">${esc(x.measures)}</td><td>${esc(x.baseline)}</td><td>${esc(x.resqnet)}</td><td>${esc(x.target)}</td>
     <td>${x.pass === true ? '<span class="pass">✔ met</span>' : x.pass === false ? '<span class="miss">✘ not yet</span>' : ""}</td></tr>`).join("")
    : '<tr><td colspan="6" class="muted">Run the hero demo (or dispatch an ambulance) to generate results.</td></tr>';
  $("results").classList.remove("hidden");
}
$("btnResults").onclick = openResults;
$("resClose").onclick = () => $("results").classList.add("hidden");

/* ------------------------------------------------------------------ controls */
function resetUi() {
  lastEventId = 0; lastDecisionId = 0; lastStoryStep = -1; resultsShown = false; routeKey = ""; fitted = {};
  $("feed").innerHTML = ""; $("results").classList.add("hidden");
  $("smsReply").textContent = "Commands: CLEAR ROUTE INC-#### · MEDIC INC-####"; $("smsReply").className = "bubble";
  for (const k in cellShapes) { cellShapes[k].circle.remove(); cellShapes[k].marker.remove(); delete cellShapes[k]; }
  map.setView([12.928, 77.627], 15);
}
$("btnDemo").onclick = async () => { resetUi(); await api("/api/demo/start?auto=true", {}); };
$("btnStepDemo").onclick = async () => { resetUi(); await api("/api/demo/start?auto=false", {}); };
$("btnNext").onclick = () => api("/api/demo/next", {});
$("btnPause").onclick = () => api("/api/pause", {});
$("btnReset").onclick = async () => { resetUi(); await api("/api/reset", {}); };
$("speed").onchange = () => api("/api/speed", { speed: parseFloat($("speed").value) });
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === " ") { e.preventDefault(); api("/api/pause", {}); }
  if (e.key === "ArrowRight" || e.key === "n") api("/api/demo/next", {});
  if (e.key === "r") openResults();
  if (e.key === "Escape") $("results").classList.add("hidden");
});

/* ------------------------------------------------------------------ websocket */
function render() {
  renderHeader();
  drawCells();
  drawRoutesAndZone();
  drawPeople();
  renderStory();
  renderKpis();
  renderPhones();
  renderTabs(false);
}
function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { $("wsdot").classList.add("on"); $("wsstate").textContent = "edge node live"; };
  ws.onclose = () => { $("wsdot").classList.remove("on"); $("wsstate").textContent = "reconnecting…"; setTimeout(connect, 1000); };
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === "init") {
      INFO = msg.data; S = msg.state; modelsDrawn = false;
      drawStatic();
    } else S = msg.data;
    if (S.events.length && S.events[S.events.length - 1].id < lastEventId) resetUi(); // server was reset elsewhere
    render();
  };
}
connect();
