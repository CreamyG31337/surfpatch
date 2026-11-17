"""Exact surface-to-surface volumes (an isopach) between two TINs, as TBC's Earthwork Volume Report computes them.

The two triangulations are overlaid in plan: every triangle of one surface is cut by the triangles of the other
(and by the boundary polygon), so each resulting piece has both surfaces as flat planes over it. On a piece the
depth is linear, so its volume is area x depth at the centroid, with no grid. Pieces are further split where the
depth crosses zero (cut vs fill) and where either surface crosses an elevation-range limit, so the totals, the
areas and the volumes by elevation range are exact up to floating point.

Points are rxlkit order (north, east, z); faces 0-based. Polygons are (east, north).
"""
from dataclasses import dataclass, field

import numpy as np
import shapely

MIN_AREA = 1e-10   # m2: pieces smaller than this are numerical slivers


@dataclass
class IsopachResult:
    cut: float = 0.0              # m3 where the initial surface is above the final one (excavated)
    fill: float = 0.0             # m3 where the final surface is above the initial one (placed)
    area_cut: float = 0.0         # m2
    area_fill: float = 0.0        # m2
    area_compared: float = 0.0    # m2 where both surfaces exist (inside the boundary, if one is given)
    area_boundary: float = float("nan")   # m2 of the boundary polygon
    max_cut: float = 0.0          # m, deepest excavation
    max_fill: float = 0.0         # m, thickest placement
    zmin: float = float("nan")    # lowest elevation of either surface over the compared area
    zmax: float = float("nan")
    bins: list = field(default_factory=list)   # (z1, z2, cut m3, fill m3) per elevation range
    pieces: int = 0               # overlay pieces, for information
    zero_lines: object = None     # (k, 2, 2) east/north segments where the depth is zero, when asked for

    @property
    def net(self):
        return self.cut - self.fill

    @property
    def area_zero(self):
        return max(self.area_compared - self.area_cut - self.area_fill, 0.0)


# ------------------------------------------------------------------ triangle geometry (local plan coordinates)

def _area(T):
    """Signed plan area of (n, 3, 2) triangles (positive = counter-clockwise)."""
    return 0.5 * ((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
                  - (T[:, 1, 1] - T[:, 0, 1]) * (T[:, 2, 0] - T[:, 0, 0]))


def _planes(T, Z):
    """Plane z = a x + b y + c through each triangle: (n, 3) array of a, b, c."""
    x0, y0 = T[:, 0, 0], T[:, 0, 1]
    ux, uy, uz = T[:, 1, 0] - x0, T[:, 1, 1] - y0, Z[:, 1] - Z[:, 0]
    vx, vy, vz = T[:, 2, 0] - x0, T[:, 2, 1] - y0, Z[:, 2] - Z[:, 0]
    det = ux * vy - uy * vx
    a = (uz * vy - vz * uy) / det
    b = (ux * vz - vx * uz) / det
    return np.c_[a, b, Z[:, 0] - a * x0 - b * y0]


def _at(A, T):
    """Plane values at the vertices: (n, 3)."""
    return A[:, :1] * T[:, :, 0] + A[:, 1:2] * T[:, :, 1] + A[:, 2:3]


def _centroid_value(A, T):
    c = T.mean(axis=1)
    return A[:, 0] * c[:, 0] + A[:, 1] * c[:, 1] + A[:, 2]


def clip(T, s):
    """Keep the part of each triangle where the linear function with vertex values s (n, 3) is > 0.

    Returns (triangles, index into the input). A triangle with one vertex inside becomes one triangle,
    with two inside a quad split in two. Orientation is kept."""
    pos = s > 0
    k = pos.sum(axis=1)
    keep = np.flatnonzero(k == 3)
    outT, outI = [T[keep]], [keep]
    for kk in (1, 2):
        m = np.flatnonzero(k == kk)
        if not len(m):
            continue
        odd = np.argmax(pos[m] if kk == 1 else ~pos[m], axis=1)   # the vertex alone on its side
        r = (odd[:, None] + np.arange(3)) % 3                       # cyclic roll keeps orientation
        t = np.take_along_axis(T[m], r[:, :, None], axis=1)
        v = np.take_along_axis(s[m], r, axis=1)
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        ab = a + (v[:, 0] / (v[:, 0] - v[:, 1]))[:, None] * (b - a)
        ac = a + (v[:, 0] / (v[:, 0] - v[:, 2]))[:, None] * (c - a)
        if kk == 1:
            outT.append(np.stack([a, ab, ac], axis=1)); outI.append(m)
        else:
            outT += [np.stack([ab, b, c], axis=1), np.stack([ab, c, ac], axis=1)]; outI += [m, m]
    T2, I2 = np.concatenate(outT), np.concatenate(outI)
    ok = _area(T2) > MIN_AREA
    return T2[ok], I2[ok]


def _triangles(P, F, off):
    """(n, 3, 2) counter-clockwise plan triangles in local coordinates and (n, 3) planes; vertical faces dropped."""
    P = np.asarray(P, float); F = np.asarray(F, int)
    T = np.stack([P[F][:, :, 1] - off[0], P[F][:, :, 0] - off[1]], axis=2)
    Z = P[F][:, :, 2]
    a = _area(T)
    cw = a < 0
    T[cw] = T[cw][:, [0, 2, 1]]; Z[cw] = Z[cw][:, [0, 2, 1]]
    ok = np.abs(a) > MIN_AREA
    return T[ok], _planes(T[ok], Z[ok])


def _polys(T):
    return shapely.polygons(np.concatenate([T, T[:, :1]], axis=1))


# ------------------------------------------------------------------ overlay

def overlay(Ta, Tb, chunk=200_000):
    """Pieces where triangles of A and B overlap in plan: (triangles, index into A, index into B)."""
    tree = shapely.STRtree(_polys(Tb))
    outT, outA, outB = [], [], []
    for s in range(0, len(Ta), chunk):
        ia, ib = tree.query(_polys(Ta[s:s + chunk]))      # bounding-box candidates
        ia = ia + s
        T, idx = Ta[ia], np.arange(len(ia))
        for e in range(3):                                # clip by each edge of the B triangle (inside = left)
            p, q = Tb[ib[idx], e], Tb[ib[idx], (e + 1) % 3]
            d = q - p
            d /= np.hypot(d[:, 0], d[:, 1])[:, None]
            sd = d[:, None, 0] * (T[:, :, 1] - p[:, None, 1]) - d[:, None, 1] * (T[:, :, 0] - p[:, None, 0])
            T, k = clip(T, sd)
            idx = idx[k]
        outT.append(T); outA.append(ia[idx]); outB.append(ib[idx])
    if not outT:
        return np.zeros((0, 3, 2)), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(outT), np.concatenate(outA), np.concatenate(outB)


def _side(T, bnd):
    """Per triangle: 1 inside the boundary, 0 crossing it, -1 outside."""
    polys = _polys(T)
    s = np.zeros(len(T), int)
    s[shapely.within(polys, bnd)] = 1
    s[shapely.disjoint(polys, bnd)] = -1
    return s


def _clip_boundary(T, bnd):
    """Clip triangles to a (prepared) polygon; returns (triangles, index into the input)."""
    inter = shapely.intersection(_polys(T), bnd)
    tri = shapely.constrained_delaunay_triangles(inter)
    parts, idx = shapely.get_parts(tri, return_index=True)
    keep = shapely.get_type_id(parts) == 3                # polygons only
    parts, idx = parts[keep], idx[keep]
    if not len(parts):
        return np.zeros((0, 3, 2)), np.zeros(0, int)
    C =shapely.get_coordinates(parts).reshape(-1, 4, 2)[:, :3].copy()
    a = _area(C)
    C[a < 0] = C[a < 0][:, [0, 2, 1]]
    ok = np.abs(a) > MIN_AREA
    return C[ok], idx[ok]


# ------------------------------------------------------------------ volumes

def _pos_integral(T, A, c):
    """Sum over triangles of the integral of max(plane - c, 0)."""
    if not len(T):
        return 0.0
    s = _at(A, T) - c
    area = _area(T)
    full = (s >= 0).all(axis=1)
    tot = float((area[full] * s[full].mean(axis=1)).sum())
    mixed = np.flatnonzero(~full & (s > 0).any(axis=1))
    if len(mixed):
        T2, k = clip(T[mixed], s[mixed])
        A2 = A[mixed][k]
        tot += float((_area(T2) * (_centroid_value(A2, T2) - c)).sum())
    return tot


def _range_volume(T, Ahi, Alo, z1, z2):
    """Integral of the overlap of [lower surface, upper surface] with [z1, z2] over triangles where Ahi >= Alo.

    overlap = clip(hi, z1, z2) - clip(lo, z1, z2), and clip(z, z1, z2) = z1 + max(z - z1, 0) - max(z - z2, 0)."""
    return (_pos_integral(T, Ahi, z1) - _pos_integral(T, Ahi, z2)
            - _pos_integral(T, Alo, z1) + _pos_integral(T, Alo, z2))


def bin_edges(zmin, zmax, interval, origin=0.0):
    """Elevation-range limits like TBC's report: multiples of interval (offset by origin), first/last at zmin/zmax."""
    e = np.arange(np.floor((zmin - origin) / interval) * interval + origin, zmax + interval, interval)
    e = e[(e > zmin + 1e-6) & (e < zmax - 1e-6)]          # zmin on a limit (to rounding) must not add an empty range
    return np.r_[zmin, e, zmax] if zmax > zmin else np.array([zmin, zmax])


def isopach(initial, final, polygon=None, interval=None, origin=0.0, edges=None, zero_lines=False):
    """Exact volumes between two TINs, each (P, F): cut where `initial` is above `final`, fill where below.

    polygon: optional boundary (east, north vertices); only the part inside it counts.
    interval/origin or edges: elevation ranges for the volumes-by-range table (TBC's "Volumes by Depth Range").
    zero_lines: also return the plan segments where the depth is zero (the cut/fill line)."""
    (Pi, Fi), (Pf, Ff) = initial, final
    Pi, Pf = np.asarray(Pi, float), np.asarray(Pf, float)
    for name, F in (("initial", Fi), ("final", Ff)):
        if not len(F):
            raise ValueError(f"the {name} surface has no triangles (a TBC 'source data only' export? "
                             "triangulate it first: surfpatch rebuild-source)")
    off = np.array([Pi[:, 1].mean(), Pi[:, 0].mean()])
    Ti, Ai = _triangles(Pi, Fi, off)
    Tf, Af = _triangles(Pf, Ff, off)
    r = IsopachResult()

    bnd = None
    if polygon is not None:
        pg = np.asarray(polygon, float)[:, :2] - off
        bnd = shapely.make_valid(shapely.Polygon(pg))
        r.area_boundary = float(bnd.area)
        shapely.prepare(bnd)
        si, sf = _side(Ti, bnd), _side(Tf, bnd)
        ki, kf = np.flatnonzero(si >= 0), np.flatnonzero(sf >= 0)
        Ti, Ai, si = Ti[ki], Ai[ki], si[ki]
        Tf, Af, sf = Tf[kf], Af[kf], sf[kf]

    T, ii, jf = overlay(Ti, Tf)
    if bnd is not None:
        cross = np.flatnonzero((si[ii] == 0) & (sf[jf] == 0))   # inside if either parent is inside
        if len(cross):
            C, k = _clip_boundary(T[cross], bnd)
            keep = np.ones(len(T), bool); keep[cross] = False
            T = np.concatenate([T[keep], C])
            ii = np.concatenate([ii[keep], ii[cross][k]])
            jf = np.concatenate([jf[keep], jf[cross][k]])
    Ai, Af = Ai[ii], Af[jf]
    r.pieces = len(T)
    if not len(T):
        return r
    r.area_compared = float(_area(T).sum())

    D = Ai - Af                                  # depth plane: initial minus final
    dv = _at(D, T)
    Tc, kc = clip(T, dv)
    Tl, kl = clip(T, -dv)
    r.cut = float((_area(Tc) * _centroid_value(D[kc], Tc)).sum())
    r.fill = -float((_area(Tl) * _centroid_value(D[kl], Tl)).sum())
    r.area_cut, r.area_fill = float(_area(Tc).sum()), float(_area(Tl).sum())
    r.max_cut, r.max_fill = max(float(dv.max()), 0.0), max(-float(dv.min()), 0.0)
    zi, zf = _at(Ai, T), _at(Af, T)
    r.zmin, r.zmax = float(min(zi.min(), zf.min())), float(max(zi.max(), zf.max()))

    if edges is None and interval:
        edges = bin_edges(r.zmin, r.zmax, interval, origin)
    if edges is not None:
        for z1, z2 in zip(edges[:-1], edges[1:]):
            c = _range_volume(Tc, Ai[kc], Af[kc], z1, z2)
            f = _range_volume(Tl, Af[kl], Ai[kl], z1, z2)
            r.bins.append((float(z1), float(z2), c, f))

    if zero_lines:
        r.zero_lines = _zero_segments(T, dv) + off
    return r


def join_segments(seg, snap=1e-4):
    """Join (k, 2, 2) segments into polylines (lists of (east, north)); ends are snapped to `snap` m first."""
    if not len(seg):
        return []
    s = np.round(np.asarray(seg, float) / snap) * snap
    s = s[np.hypot(*(s[:, 1] - s[:, 0]).T) > 0]
    merged = shapely.line_merge(shapely.multilinestrings(shapely.linestrings(s)))
    return [shapely.get_coordinates(g) for g in shapely.get_parts(merged)]


def _zero_segments(T, s):
    """Plan segments where the linear function (vertex values s) is zero, one per triangle it crosses."""
    pos = s > 0
    k = pos.sum(axis=1)
    m = np.flatnonzero((k == 1) | (k == 2))
    if not len(m):
        return np.zeros((0, 2, 2))
    odd = np.argmax(np.where((k[m] == 1)[:, None], pos[m], ~pos[m]), axis=1)
    r = (odd[:, None] + np.arange(3)) % 3
    t = np.take_along_axis(T[m], r[:, :, None], axis=1)
    v = np.take_along_axis(s[m], r, axis=1)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    ab = a + (v[:, 0] / (v[:, 0] - v[:, 1]))[:, None] * (b - a)
    ac = a + (v[:, 0] / (v[:, 0] - v[:, 2]))[:, None] * (c - a)
    seg = np.stack([ab, ac], axis=1)
    return seg[np.hypot(*(seg[:, 1] - seg[:, 0]).T) > 1e-6]
