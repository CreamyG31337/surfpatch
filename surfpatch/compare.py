"""Checks and comparisons on TINs: elevation sampling, volume difference, spikes, sparse areas.

Points are rxlkit order (north, east, z); faces 0-based. Plan coordinates passed in are (east, north).
"""
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MPath
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .mesh import edge_keys


class Sampler:
    """Linear elevation on a TIN at plan points; NaN off the surface."""

    def __init__(self, P, F):
        P = np.asarray(P, float)
        self.off = np.array([P[:, 1].mean(), P[:, 0].mean()])
        tri = Triangulation(P[:, 1] - self.off[0], P[:, 0] - self.off[1], np.asarray(F, int))
        self._i = LinearTriInterpolator(tri, P[:, 2])

    def __call__(self, east, north):
        z = self._i(np.asarray(east, float) - self.off[0], np.asarray(north, float) - self.off[1])
        return np.ma.filled(z.astype(float), np.nan)


def inside(polygon, east, north):
    """Bool mask: plan points inside a closed polygon given as (k, 2) east/north vertices."""
    return MPath(np.asarray(polygon, float)).contains_points(np.c_[np.ravel(east), np.ravel(north)])


@dataclass
class DiffResult:
    step: float
    compared_area: float = 0.0      # m2 where both surfaces exist
    changed_area: float = 0.0       # m2 where |dz| > 1 mm
    raised: float = 0.0             # m3 where new is above old
    lowered: float = 0.0            # m3 where new is below old (positive number)
    max_raise: float = float("nan")
    max_drop: float = float("nan")
    grid: dict = field(default_factory=dict)   # x0, y0, step, dz (rows = north) when keep_grid

    @property
    def net(self):
        return self.raised - self.lowered


def diff(new, old, step=0.25, polygon=None, bbox=None, keep_grid=False):
    """Volume of `new` minus `old` (each (P, F)) on a grid of `step` m cells, over their common extent,
    optionally limited to a polygon (east/north vertices) or bbox (emin, nmin, emax, nmax)."""
    (Pn, Fn), (Po, Fo) = new, old
    e0 = max(Pn[:, 1].min(), Po[:, 1].min()); e1 = min(Pn[:, 1].max(), Po[:, 1].max())
    n0 = max(Pn[:, 0].min(), Po[:, 0].min()); n1 = min(Pn[:, 0].max(), Po[:, 0].max())
    if polygon is not None:
        pg = np.asarray(polygon, float)
        e0, n0 = max(e0, pg[:, 0].min()), max(n0, pg[:, 1].min())
        e1, n1 = min(e1, pg[:, 0].max()), min(n1, pg[:, 1].max())
    if bbox is not None:
        e0, n0, e1, n1 = max(e0, bbox[0]), max(n0, bbox[1]), min(e1, bbox[2]), min(n1, bbox[3])
    r = DiffResult(step)
    if e1 <= e0 or n1 <= n0:
        return r
    zn, zo = Sampler(Pn, Fn), Sampler(Po, Fo)
    xs = np.arange(e0 + step / 2, e1, step)
    rows, c = [], step * step
    hi, lo = -np.inf, np.inf
    for y in np.arange(n0 + step / 2, n1, step):
        yy = np.full_like(xs, y)
        dz = zn(xs, yy) - zo(xs, yy)
        if polygon is not None:
            dz[~inside(polygon, xs, yy)] = np.nan
        ok = ~np.isnan(dz)
        if ok.any():
            d = dz[ok]
            r.compared_area += ok.sum() * c
            r.changed_area += (np.abs(d) > 0.001).sum() * c
            r.raised += d[d > 0].sum() * c
            r.lowered -= d[d < 0].sum() * c
            hi, lo = max(hi, d.max()), min(lo, d.min())
        if keep_grid:
            rows.append(dz)
    if np.isfinite(hi):
        r.max_raise, r.max_drop = float(hi), float(lo)
    if keep_grid:
        r.grid = {"x0": float(xs[0]) if len(xs) else e0, "y0": n0 + step / 2, "step": step,
                  "dz": np.array(rows)}
    return r


def neighbours(F, n):
    """CSR-style (starts, ids): neighbours of vertex v are ids[starts[v]:starts[v+1]]."""
    e = np.unique(edge_keys(F), axis=0)
    a, b = np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]]
    order = np.argsort(a, kind="stable")
    starts = np.searchsorted(a[order], np.arange(n + 1))
    return starts, b[order]


def spikes(P, F, verts=None):
    """z minus the median z of each vertex's TIN neighbours, for `verts` (default all used vertices).

    Returns (vertex indices, dz). |dz| > ~0.75 m on a natural surface is worth a look."""
    P = np.asarray(P, float)
    starts, ids = neighbours(F, len(P))
    verts = np.unique(F) if verts is None else np.asarray(verts, int)
    dz = np.full(len(verts), np.nan)
    for k, v in enumerate(verts):
        nb = ids[starts[v]:starts[v + 1]]
        if len(nb):
            dz[k] = P[v, 2] - np.median(P[nb, 2])
    return verts, dz


def sparse_regions(P, F, max_edge=12.0, min_area=300.0):
    """Clusters (sharing a vertex) of faces with an edge longer than max_edge, larger than min_area m2.

    Returns dicts sorted by area: area, faces (indices), vertices, centre (e, n), z range, length x width."""
    P = np.asarray(P, float)
    F = np.asarray(F, int)
    en = P[:, [1, 0]]
    A, B, C = en[F[:, 0]], en[F[:, 1]], en[F[:, 2]]
    emax = np.max([np.hypot(*(A - B).T), np.hypot(*(B - C).T), np.hypot(*(C - A).T)], axis=0)
    area = 0.5 * np.abs((B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1]))
    big = np.flatnonzero(emax > max_edge)
    if not len(big):
        return []
    # faces linked through shared vertices: bipartite face-vertex graph
    nf = len(big)
    rows = np.repeat(np.arange(nf), 3)
    cols = nf + F[big].ravel()
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nf + len(P), nf + len(P)))
    _, lab = connected_components(g, directed=False)
    out = []
    for c in np.unique(lab[:nf]):
        ts = big[lab[:nf] == c]
        a = float(area[ts].sum())
        if a < min_area:
            continue
        vs = np.unique(F[ts])
        Q = en[vs]
        ctr = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - ctr, full_matrices=False)
        proj = (Q - ctr) @ vt.T
        out.append({"area": a, "faces": ts, "vertices": vs, "centre": ctr.tolist(),
                    "z": (float(P[vs, 2].min()), float(P[vs, 2].max())),
                    "size": (float(np.ptp(proj[:, 0])), float(np.ptp(proj[:, 1]) if proj.shape[1] > 1 else 0))})
    return sorted(out, key=lambda d: -d["area"])
