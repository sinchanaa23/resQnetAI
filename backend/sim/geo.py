"""Small geodesy helpers (no external dependencies)."""
import math

EARTH_R = 6371000.0


def haversine(a, b):
    """Distance in metres between (lat, lon) points."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(h)))


def to_xy(p, origin):
    """Equirectangular projection to metres around origin — accurate enough at city scale."""
    x = math.radians(p[1] - origin[1]) * EARTH_R * math.cos(math.radians(origin[0]))
    y = math.radians(p[0] - origin[0]) * EARTH_R
    return x, y


def from_xy(x, y, origin):
    lat = origin[0] + math.degrees(y / EARTH_R)
    lon = origin[1] + math.degrees(x / (EARTH_R * math.cos(math.radians(origin[0]))))
    return lat, lon


def offset(p, dx_m, dy_m):
    return from_xy(dx_m, dy_m, p)


def point_segment_dist(p, a, b):
    """Distance in metres from point p to segment a-b (all lat/lon)."""
    px, py = to_xy(p, a)
    bx, by = to_xy(b, a)
    L2 = bx * bx + by * by
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, (px * bx + py * by) / L2))
    dx, dy = px - t * bx, py - t * by
    return math.hypot(dx, dy)


def polyline_length(pts):
    return sum(haversine(a, b) for a, b in zip(pts, pts[1:]))


def interpolate(pts, cum, d):
    """Point at distance d along polyline pts with cumulative distances cum."""
    if d <= 0:
        return tuple(pts[0])
    if d >= cum[-1]:
        return tuple(pts[-1])
    lo, hi = 0, len(cum) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cum[mid] <= d:
            lo = mid
        else:
            hi = mid
    seg = cum[hi] - cum[lo]
    t = 0 if seg == 0 else (d - cum[lo]) / seg
    a, b = pts[lo], pts[hi]
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def slice_polyline(pts, cum, d0, d1):
    """Sub-polyline between distances d0 and d1."""
    d0, d1 = max(0, d0), min(cum[-1], d1)
    if d1 <= d0:
        return []
    out = [interpolate(pts, cum, d0)]
    for p, c in zip(pts, cum):
        if d0 < c < d1:
            out.append(tuple(p))
    out.append(interpolate(pts, cum, d1))
    return out


def bearing(a, b):
    la1, la2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    x = math.sin(dl) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def circle_polygon(center, radius_m, n=48):
    return [offset(center, radius_m * math.sin(2 * math.pi * i / n), radius_m * math.cos(2 * math.pi * i / n))
            for i in range(n)]
