"""Build TIN surfaces from survey points, breaklines, depression rings, and overlapping shot clouds.

Points are rxlkit order (north, east, z); faces are 0-based index triples. Plan-only polygons (`boundary`, `band`)
are (east, north).
"""
import numpy as np
import shapely
import triangle
from scipy.spatial import Delaunay

from .compare import inside


def _as_pts(P):
    arr = np.asarray(P, float)
    if arr.size == 0:
        return np.zeros((0, 3), float)
    return arr.reshape(-1, 3)


def _close_ring(R, snap=0.002):
    R = _as_pts(R)
    if len(R) >= 3 and np.hypot(*(R[0, :2] - R[-1, :2])) > snap:
        return np.vstack([R, R[:1]])
    return R


def _steiner_z(q, xy, Z, segs):
    """Linearly interpolate elevation at a Steiner point `q` (e, n) from the breakline segment(s) it lies on."""
    a, b = segs[:, 0], segs[:, 1]
    p0, p1 = xy[a], xy[b]
    d = p1 - p0
    L2 = np.einsum("ij,ij->i", d, d)
    safe = L2 > 0
    u = np.zeros(len(segs), float)
    u[safe] = np.clip(np.einsum("ij,ij->i", q - p0[safe], d[safe]) / L2[safe], 0.0, 1.0)
    proj = p0 + u[:, None] * d
    dist = np.hypot(*(q - proj).T)
    hit = dist <= dist.min() + 1e-6
    z_seg = (1.0 - u[hit]) * Z[a[hit]] + u[hit] * Z[b[hit]]
    return float(np.mean(z_seg))


def tin_from_points(P, breaklines=(), boundary=None, snap=0.002, allow_steiner=False):
    """Constrained Delaunay TIN of `P` (n, e, z) with `breaklines` (each (k, 3) n/e/z) as segments.

    A breakline is closed automatically when its last vertex meets its first within `snap`. Breakline vertices
    within `snap` (plan) of an existing point are reused; others are appended. Faces whose centroid lies outside
    `boundary` ((k, 2) east/north) are dropped. Raises ValueError if crossing breaklines force Steiner points,
    unless `allow_steiner=True` (Steiner elevation is linearly interpolated along the breakline).
    """
    pts = [p.copy() for p in _as_pts(P)]
    segs = []
    for bl in breaklines:
        R = _as_pts(bl)
        if len(R) < 2:
            continue
        idx = []
        for r in R:
            if pts:
                arr = np.asarray(pts, float)
                d = np.hypot(arr[:, 0] - r[0], arr[:, 1] - r[1])
                k = int(np.argmin(d))
                if d[k] <= snap:
                    idx.append(k)
                    continue
            idx.append(len(pts))
            pts.append(r.copy())
        for i, j in zip(idx, idx[1:]):
            if i != j:
                segs.append((i, j))

    if len(pts) < 3:
        raise ValueError(f"need at least 3 points to build a TIN (got {len(pts)})")
    V = np.asarray(pts, float)
    en = V[:, [1, 0]]
    off = en.mean(axis=0)
    xy = (en - off).copy()

    # Deduplicate undirected segments while preserving order
    uniq_segs, seen = [], set()
    for i, j in segs:
        key = (i, j) if i < j else (j, i)
        if key not in seen:
            seen.add(key)
            uniq_segs.append((i, j))

    args = {"vertices": xy}
    if uniq_segs:
        seg_arr = np.asarray(uniq_segs, int)
        args["segments"] = seg_arr
        t = triangle.triangulate(args, "pcQ")
    else:
        seg_arr = np.zeros((0, 2), int)
        t = triangle.triangulate(args, "cQ")

    tv = np.asarray(t["vertices"], float)
    n_steiner = len(tv) - len(V)
    if n_steiner > 0:
        if not allow_steiner:
            raise ValueError(f"breaklines cross ({n_steiner} Steiner point(s) added)")
        extra = []
        for q in tv[len(V):]:
            z = _steiner_z(q, xy, V[:, 2], seg_arr)
            extra.append([q[1] + off[1], q[0] + off[0], z])
        V = np.vstack([V, np.asarray(extra, float)])

    F = np.asarray(t.get("triangles", np.zeros((0, 3), int)), int).reshape(-1, 3)
    if boundary is not None and len(F):
        cen_en = V[F, :2].mean(axis=1)[:, [1, 0]]
        F = F[inside(boundary, cen_en[:, 0], cen_en[:, 1])]
    return V, F


def depression(P_bottom, rings, rim, P_top=None, snap=0.002):
    """TBC-style depression surfaces: `(bottom (P, F), top_or_lid (P, F))`.

    `bottom` is the constrained Delaunay TIN of `P_bottom` with every ring in `rings` as a closed breakline,
    clipped to `rim`. When `P_top` is None, the upper surface is a lid TIN built from `rim` vertices only
    (TBC Stockpile Volume Analysis). When `P_top` is given, `rim` is merged into `P_top` so the top TIN closes
    cleanly out to the rim boundary.
    """
    rim = _as_pts(rim)
    closed = [_close_ring(r, snap=snap) for r in rings]
    Pb, Fb = tin_from_points(P_bottom, breaklines=closed, boundary=rim[:, [1, 0]], snap=snap)
    top_pts = rim if P_top is None else np.vstack([_as_pts(P_top), rim])
    _, idx = np.unique(np.round(top_pts[:, :2], 3), axis=0, return_index=True)
    Pt = top_pts[idx]
    Ft = Delaunay(Pt[:, [1, 0]]).simplices
    return (Pb, Fb), (Pt, Ft)


def station_band(alignment, s0, s1, half_width=5.0, n=40):
    """Polygon (east, north) of the corridor band from station `s0` to `s1`, +/- `half_width` off the CL."""
    left = [alignment.point(s, -half_width) for s in np.linspace(s0, s1, n)]
    right = [alignment.point(s, half_width) for s in np.linspace(s1, s0, n)]
    return np.array([(e, n_) for n_, e in left + right], float)


def overlap_boundary(tins, band=None):
    """Plan polygon (east, north) where every TIN's convex hull overlaps, optionally clipped to `band`."""
    poly = None
    if band is not None:
        poly = band if isinstance(band, shapely.Geometry) else shapely.Polygon(np.asarray(band, float))
    for tin in tins:
        P = tin[0] if isinstance(tin, tuple) else tin
        hull = shapely.MultiPoint(_as_pts(P)[:, [1, 0]]).convex_hull
        poly = hull if poly is None else poly.intersection(hull)
    if poly is None or poly.is_empty:
        raise ValueError("surfaces do not overlap")
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    elif poly.geom_type == "GeometryCollection":
        polys = [g for g in poly.geoms if g.geom_type == "Polygon"]
        if not polys:
            raise ValueError("surfaces overlap has no area")
        poly = max(polys, key=lambda g: g.area)
    if poly.geom_type != "Polygon" or poly.area <= 0:
        raise ValueError("surfaces overlap has no area")
    return np.array(poly.exterior.coords, float)[:-1]
