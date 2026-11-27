"""Patch a TIN from repeat drone flights: per grid cell, the lowest ground that two flights agree on.

Excavation only lowers rock and later fill only raises it, so where a surface sits above ground that two
separate flights both saw lower (within `confirm` m), the surface is out of date there. The patch is a grid
TIN of that confirmed-lowest ground, only where it is more than `min_drop` below the surface, never above it.

Steps (drone_patch):
1. Each flight's points binned to `step` m cells: median Z and spread (p90 - p10), cells with >= 4 points.
2. Per cell, sort the flights' medians; the lowest one that the next-lowest is within `confirm` of wins
   (their mean). One low flight on its own (a bad flight, a hole) is never used.
3. Cells that are walls (spread > max_spread or slope > max_slope deg) are not trusted, and neither are cells
   where an optional check cloud (e.g. a terrestrial scan) disagrees by more than check_tol.
4. Candidate cells (confirmed ground > min_drop below the surface, inside the optional mask) are closed,
   hole-filled, and clusters smaller than min_area m2 dropped. Untrusted cells inside get interpolated
   drone Z, except within edge_band m of the surface's outer edge.
5. Output Z = min(patch, surface): the patch only ever lowers.

Points are rxlkit order (north, east, z) for surfaces; clouds are (x=east, y=north, z) arrays.
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

from .compare import Sampler, inside


@dataclass
class Grid:
    x0: float
    y0: float
    step: float
    nx: int
    ny: int

    @classmethod
    def over(cls, emin, nmin, emax, nmax, step):
        x0, y0 = np.floor(emin), np.floor(nmin)
        return cls(float(x0), float(y0), step, int(np.ceil((emax - x0) / step)), int(np.ceil((nmax - y0) / step)))

    def centres(self):
        return np.meshgrid(self.x0 + (np.arange(self.nx) + 0.5) * self.step,
                           self.y0 + (np.arange(self.ny) + 0.5) * self.step)


def cell_stats(grid, x, y, z, min_count=4):
    """(median, spread p90 - p10) per cell as (ny, nx) arrays; NaN where fewer than min_count points."""
    ix = np.floor((np.asarray(x) - grid.x0) / grid.step).astype(np.int64)
    iy = np.floor((np.asarray(y) - grid.y0) / grid.step).astype(np.int64)
    k = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    key, z = iy[k] * grid.nx + ix[k], np.asarray(z, float)[k]
    o = np.lexsort((z, key))
    key, z = key[o], z[o]
    med = np.full(grid.nx * grid.ny, np.nan)
    spr = np.full(grid.nx * grid.ny, np.nan)
    if len(key):
        u, st, cnt = np.unique(key, return_index=True, return_counts=True)
        ok = cnt >= min_count
        u, st, cnt = u[ok], st[ok], cnt[ok]
        med[u] = z[st + cnt // 2]
        spr[u] = z[st + (cnt * 9) // 10] - z[st + cnt // 10]
    return med.reshape(grid.ny, grid.nx), spr.reshape(grid.ny, grid.nx)


def confirmed_lowest(medians, spreads, confirm=0.10):
    """medians/spreads: (flights, ny, nx). Returns (z, flight index or -1, spread) of the lowest flight
    that the next-lowest flight is within `confirm` of; z is the mean of the two."""
    M = np.asarray(medians, float)
    S = np.asarray(spreads, float)
    zd = np.full(M.shape[1:], np.nan)
    fl = np.full(M.shape[1:], -1)
    sp = np.full(M.shape[1:], np.nan)
    order = np.argsort(np.where(np.isnan(M), np.inf, M), axis=0)
    for r in range(len(M) - 1):
        a = np.take_along_axis(M, order[r:r + 1], 0)[0]
        b = np.take_along_axis(M, order[r + 1:r + 2], 0)[0]
        take = np.isnan(zd) & np.isfinite(a) & np.isfinite(b) & (b - a <= confirm)
        zd[take] = ((a + b) / 2)[take]
        fl[take] = order[r][take]
        sp[take] = np.take_along_axis(S, order[r:r + 1], 0)[0][take]
    return zd, fl, sp


def grid_tin(grid, zo, region):
    """Grid TIN over the region's cell centres: points (north, east, z), faces 0-based."""
    GX, GY = grid.centres()
    J, I = np.nonzero(region)
    P = np.c_[GY[J, I], GX[J, I], zo[J, I]]
    pid = -np.ones(region.shape, int)
    pid[J, I] = np.arange(len(J))
    a, b = pid[:-1, :-1], pid[:-1, 1:]
    c, d = pid[1:, 1:], pid[1:, :-1]
    quad = np.stack([a, b, c, d], -1).reshape(-1, 4)
    n_ok = (quad >= 0).sum(1)
    tris = [quad[n_ok == 4][:, [0, 1, 2]], quad[n_ok == 4][:, [0, 2, 3]]]
    for q in quad[n_ok == 3]:
        tris.append(q[q >= 0][None])
    F = np.concatenate(tris) if tris else np.zeros((0, 3), int)
    return P, F.astype(int)


def region_outline(grid, region):
    """Outline loops of the region's cells (east, north), as lists of corner points."""
    J, I = np.nonzero(region)
    edges = Counter()
    for j, i in zip(J, I):
        xa, ya = grid.x0 + i * grid.step, grid.y0 + j * grid.step
        xb, yb = xa + grid.step, ya + grid.step
        for e in (((xa, ya), (xb, ya)), ((xb, ya), (xb, yb)), ((xb, yb), (xa, yb)), ((xa, yb), (xa, ya))):
            rev = (e[1], e[0])
            if edges[rev]:
                edges[rev] -= 1
            else:
                edges[e] += 1
    nxt = {}
    for (a, b), c in edges.items():
        if c:
            nxt.setdefault(a, []).append(b)
    loops = []
    while nxt:
        s = next(iter(nxt))
        loop, cur = [s], s
        while True:
            nb = nxt[cur].pop()
            if not nxt[cur]:
                del nxt[cur]
            if nb == s:
                break
            loop.append(nb)
            cur = nb
        loops.append(loop)
    return loops


@dataclass
class DronePatch:
    grid: Grid
    points: np.ndarray            # patch TIN (north, east, z)
    faces: np.ndarray
    region: np.ndarray            # (ny, nx) bool
    z_patch: np.ndarray           # (ny, nx), NaN outside region
    z_surface: np.ndarray         # (ny, nx), NaN off the surface
    flight: np.ndarray            # (ny, nx) index of the flight used, -1 = interpolated / none
    stats: dict = field(default_factory=dict)


def drone_patch(P, F, flights, step=0.5, confirm=0.10, min_drop=0.10, min_area=4.0, max_spread=0.5,
                max_slope=45.0, mask_polygon=None, check=None, check_tol=0.25, edge_band=1.5):
    """Confirmed-lowest drone patch for surface (P, F).

    flights: list of (x, y, z) arrays (east, north, z), one per flight; at least two.
    mask_polygon: optional (k, 2) east/north polygon - patch only inside it.
    check: optional (x, y, z) of an independent cloud (e.g. TLS); cells where it disagrees with the drone
    ground by more than check_tol are not trusted."""
    if len(flights) < 2:
        raise ValueError("need at least two flights to confirm ground")
    P = np.asarray(P, float)
    grid = Grid.over(P[:, 1].min(), P[:, 0].min(), P[:, 1].max(), P[:, 0].max(), step)
    GX, GY = grid.centres()
    zc = Sampler(P, F)(GX.ravel(), GY.ravel()).reshape(GX.shape)
    on = np.isfinite(zc)
    allow = on.copy()
    if mask_polygon is not None:
        allow &= inside(mask_polygon, GX.ravel(), GY.ravel()).reshape(GX.shape)

    st = [cell_stats(grid, *f) for f in flights]
    zd, fl, sp = confirmed_lowest([s[0] for s in st], [s[1] for s in st], confirm)
    with np.errstate(invalid="ignore"):
        gy, gx = np.gradient(zd, step)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        wall = (sp > max_spread) | (slope > max_slope)
    bad = np.zeros_like(on)
    if check is not None:
        zt, _ = cell_stats(grid, *check)
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(zt) & (np.abs(zd - zt) > check_tol)

    edge_dist = ndimage.distance_transform_edt(on) * step
    with np.errstate(invalid="ignore"):
        cand = allow & np.isfinite(zd) & ~wall & ~bad & (zd < zc - min_drop)
    reg = ndimage.binary_closing(cand, iterations=2) & allow
    reg = ndimage.binary_fill_holes(reg) & allow
    reg &= ~((wall | bad) & (edge_dist < edge_band))   # never interpolate across an outer wall band
    lab, n = ndimage.label(reg)
    if n:
        sizes = ndimage.sum(reg, lab, range(1, n + 1))
        reg = np.isin(lab, 1 + np.flatnonzero(sizes * step * step >= min_area))

    good = reg & np.isfinite(zd) & ~wall & ~bad
    zo = zd.copy()
    fill = reg & ~good
    if fill.any() and good.sum() >= 3:
        zo[fill] = griddata(np.c_[GX[good], GY[good]], zd[good], np.c_[GX[fill], GY[fill]], method="linear")
        nn = fill & np.isnan(zo)
        if nn.any():
            zo[nn] = griddata(np.c_[GX[good], GY[good]], zd[good], np.c_[GX[nn], GY[nn]], method="nearest")
    reg &= np.isfinite(zo)
    zo = np.where(reg, np.fmin(zo, zc), np.nan)
    flight = np.where(good & reg, fl, -1)

    Pp, Fp = grid_tin(grid, zo, reg)
    A = step * step
    dz = (zo - zc)[reg]
    stats = {"surface_area": float(on.sum() * A), "patch_area": float(reg.sum() * A),
             "interpolated_area": float((fill & reg).sum() * A), "volume": float(dz.sum() * A),
             "median_dz": float(np.median(dz)) if len(dz) else float("nan"),
             "flight_area": {int(k): float(c * A) for k, c in Counter(flight[reg & (flight >= 0)].tolist()).items()}}
    if check is not None:
        k = reg & np.isfinite(zt)
        if k.any():
            stats["check_area"] = float(k.sum() * A)
            stats["check_median"] = float(np.median((zo - zt)[k]))
    return DronePatch(grid, Pp, Fp, reg, zo, zc, flight, stats)
