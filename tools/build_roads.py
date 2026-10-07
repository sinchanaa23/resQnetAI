"""Build the compact road graph used by the simulator from OpenStreetMap data.

Usage:
    python tools/build_roads.py            # downloads from Overpass API
    python tools/build_roads.py raw.json   # uses a saved Overpass response

Output: backend/data/roads.json
  nodes: [[lat, lon], ...]            (intersections / dead ends only)
  edges: [[u, v, length_m, cls, name_idx, [[lat, lon], ...]], ...]
  names: [str, ...]

Map data (c) OpenStreetMap contributors, ODbL.
"""
import json
import math
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

BBOX = (12.905, 77.600, 12.945, 77.650)  # Koramangala / Ejipura / HSR Layout, Bengaluru
CLASSES = "primary|secondary|tertiary|primary_link|secondary_link|trunk|residential"
OUT = Path(__file__).resolve().parent.parent / "backend" / "data" / "roads.json"


def haversine(a, b):
    r = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def download():
    q = (f'[out:json][timeout:60];way["highway"~"^({CLASSES})$"]'
         f"({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});(._;>;);out body;")
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=urllib.parse.urlencode({"data": q}).encode(),
        headers={"User-Agent": "resQnet-prototype"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=90).read())


def main():
    raw = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else download()
    coords = {e["id"]: (e["lat"], e["lon"]) for e in raw["elements"] if e["type"] == "node"}
    ways = [e for e in raw["elements"] if e["type"] == "way"]

    # Count how many ways touch each OSM node -> intersections become graph nodes.
    use = defaultdict(int)
    for w in ways:
        for i, n in enumerate(w["nodes"]):
            use[n] += 2 if i in (0, len(w["nodes"]) - 1) else 1

    names, name_idx = [], {}
    node_idx, nodes, edges = {}, [], []

    def nid(osm):
        if osm not in node_idx:
            node_idx[osm] = len(nodes)
            nodes.append([round(coords[osm][0], 6), round(coords[osm][1], 6)])
        return node_idx[osm]

    for w in ways:
        tags = w.get("tags", {})
        cls = tags.get("highway", "residential").replace("_link", "")
        name = tags.get("name", "")
        if name not in name_idx:
            name_idx[name] = len(names)
            names.append(name)
        oneway = tags.get("oneway") == "yes"
        seq = [n for n in w["nodes"] if n in coords]
        start, geom, length = seq[0], [coords[seq[0]]], 0.0
        for prev, cur in zip(seq, seq[1:]):
            length += haversine(coords[prev], coords[cur])
            geom.append(coords[cur])
            if use[cur] > 1 or cur == seq[-1]:
                g = [[round(p[0], 6), round(p[1], 6)] for p in geom]
                edges.append([nid(start), nid(cur), round(length, 1), cls, name_idx[name], g, 1 if oneway else 0])
                start, geom, length = cur, [coords[cur]], 0.0

    # Keep the largest connected component so every route query succeeds.
    adj = defaultdict(set)
    for u, v, *_ in edges:
        adj[u].add(v)
        adj[v].add(u)
    seen, best = set(), set()
    for s in range(len(nodes)):
        if s in seen:
            continue
        comp, stack = {s}, [s]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in comp:
                    comp.add(y)
                    stack.append(y)
        seen |= comp
        if len(comp) > len(best):
            best = comp
    remap = {old: i for i, old in enumerate(sorted(best))}
    nodes = [nodes[old] for old in sorted(best)]
    edges = [[remap[u], remap[v], *rest] for u, v, *rest in edges if u in remap and v in remap and u != v]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"bbox": BBOX, "nodes": nodes, "edges": edges, "names": names,
                               "attribution": "(c) OpenStreetMap contributors, ODbL"}, separators=(",", ":")))
    print(f"{len(nodes)} nodes, {len(edges)} edges -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
