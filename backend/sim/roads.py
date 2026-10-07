"""Road network (real OpenStreetMap streets of Koramangala / HSR Layout) and traffic model."""
import heapq
import json
import math
from pathlib import Path

from .geo import haversine, point_segment_dist

DATA = Path(__file__).resolve().parent.parent / "data" / "roads.json"

# Free-flow speeds (km/h) by OSM road class.
FREE_SPEED = {"trunk": 50, "primary": 45, "secondary": 38, "tertiary": 30, "residential": 22}
# Evening-peak congestion: fraction of free-flow speed actually achieved.
PEAK_FACTOR = {"trunk": 0.34, "primary": 0.38, "secondary": 0.45, "tertiary": 0.55, "residential": 0.72}
CLASS_RANK = {"trunk": 4, "primary": 3, "secondary": 2, "tertiary": 1, "residential": 0}


def _edge_noise(i):
    """Deterministic per-edge variation in [-1, 1] (stands in for local bottlenecks)."""
    x = math.sin(i * 12.9898 + 78.233) * 43758.5453
    return 2 * (x - math.floor(x)) - 1


class Route:
    def __init__(self, graph, edge_seq):
        """edge_seq: list of (edge_index, forward: bool)."""
        self.edges = edge_seq
        pts, cum, starts = [], [0.0], []
        for ei, fwd in edge_seq:
            geom = graph.geom[ei] if fwd else graph.geom[ei][::-1]
            starts.append(cum[-1] if pts else 0.0)
            for p in (geom if not pts else geom[1:]):
                if pts:
                    cum.append(cum[-1] + haversine(pts[-1], p))
                pts.append(tuple(p))
        self.points = pts
        self.cum = cum
        self.edge_start = starts  # distance along route where each edge begins
        self.length = cum[-1]

    def edge_at(self, d):
        """Index into self.edges for distance d along the route."""
        lo, hi = 0, len(self.edge_start) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.edge_start[mid] <= d:
                lo = mid
            else:
                hi = mid - 1
        return lo


class RoadGraph:
    def __init__(self, path=DATA):
        d = json.loads(Path(path).read_text())
        self.bbox = d["bbox"]
        self.nodes = [tuple(n) for n in d["nodes"]]
        self.names = d["names"]
        self.u, self.v, self.length, self.cls, self.name, self.geom = [], [], [], [], [], []
        self.adj = [[] for _ in self.nodes]
        for i, (u, v, length, cls, ni, geom, oneway) in enumerate(d["edges"]):
            self.u.append(u)
            self.v.append(v)
            self.length.append(length)
            self.cls.append(cls if cls in FREE_SPEED else "residential")
            self.name.append(self.names[ni])
            self.geom.append([tuple(p) for p in geom])
            self.adj[u].append((i, v, True))
            if not oneway:
                self.adj[v].append((i, u, False))
        self.mid = [g[len(g) // 2] for g in self.geom]
        # Coarse grid index for nearest-node lookups.
        self._grid = {}
        for i, (la, lo) in enumerate(self.nodes):
            self._grid.setdefault((round(la, 3), round(lo, 3)), []).append(i)
        self.flood = None  # (center, radius_m) set by the scenario

    # ---- lookups -------------------------------------------------------
    def nearest_node(self, p, major_only=False):
        best, bd = None, 1e18
        for r in (1, 2, 4, 8):
            for dla in range(-r, r + 1):
                for dlo in range(-r, r + 1):
                    key = (round(round(p[0], 3) + dla / 1000, 3), round(round(p[1], 3) + dlo / 1000, 3))
                    for n in self._grid.get(key, []):
                        if major_only and not any(CLASS_RANK[self.cls[e]] >= 1 for e, _, _ in self.adj[n]):
                            continue
                        dd = haversine(p, self.nodes[n])
                        if dd < bd:
                            best, bd = n, dd
            if best is not None:
                return best
        return min(range(len(self.nodes)), key=lambda n: haversine(p, self.nodes[n]))

    def edges_near(self, p, radius):
        return [i for i, m in enumerate(self.mid) if haversine(p, m) < radius + self.length[i] / 2
                and any(point_segment_dist(p, a, b) < radius for a, b in zip(self.geom[i], self.geom[i][1:]))]

    # ---- ground-truth traffic model -----------------------------------
    def flood_factor(self, i):
        if not self.flood:
            return 1.0
        c, r = self.flood
        d = haversine(self.mid[i], c)
        if d < r * 0.35:
            return 0.25  # waterlogged, crawling
        if d < r:
            return 0.55
        return 1.0

    def true_speed_kmh(self, i, hour=19.0, clear=0.0):
        """clear in [0, 1]: share of drivers ahead who were alerted in time and yielded (modelled
        assumption: a fully cleared road recovers 45 % of the gap to 85 % of free-flow speed).
        Flood slow-down is physical and is not removed by clearance."""
        cls = self.cls[i]
        peak = PEAK_FACTOR[cls] + (1 - PEAK_FACTOR[cls]) * (1 - _peakness(hour))
        f = max(0.12, min(1.0, peak * (1 + 0.22 * _edge_noise(i))))
        if clear > 0 and f < 0.85:
            f += (0.85 - f) * 0.45 * clear
        return FREE_SPEED[cls] * f * self.flood_factor(i)

    def true_time(self, i, hour=19.0, clear=0.0):
        return self.length[i] / (self.true_speed_kmh(i, hour, clear) / 3.6)

    def free_time(self, i):
        return self.length[i] / (FREE_SPEED[self.cls[i]] / 3.6)

    # ---- routing --------------------------------------------------------
    def route(self, src, dst, cost):
        """Dijkstra from node src to node dst; cost(edge_index) -> seconds."""
        dist = {src: 0.0}
        prev = {}
        pq = [(0.0, src)]
        while pq:
            d, x = heapq.heappop(pq)
            if x == dst:
                break
            if d > dist.get(x, 1e18):
                continue
            for ei, y, fwd in self.adj[x]:
                nd = d + cost(ei)
                if nd < dist.get(y, 1e18):
                    dist[y] = nd
                    prev[y] = (x, ei, fwd)
                    heapq.heappush(pq, (nd, y))
        if dst not in prev and dst != src:
            return None
        seq, x = [], dst
        while x != src:
            px, ei, fwd = prev[x]
            seq.append((ei, fwd))
            x = px
        return Route(self, seq[::-1])


def _peakness(hour):
    """0 at night, 1 at the evening peak (~19:00), with a smaller morning peak."""
    eve = math.exp(-((hour - 19.0) ** 2) / 4.0)
    morn = 0.8 * math.exp(-((hour - 9.5) ** 2) / 3.0)
    return max(eve, morn, 0.25 if 7 <= hour <= 22 else 0.05)
