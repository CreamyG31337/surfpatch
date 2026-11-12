"""Local re-triangulation of a TIN: faces outside the rebuilt region are kept exactly as they were.

Points are rxlkit order (north, east, z); faces are 0-based index triples. Plan-only arrays named `en` are
(east, north).

The rebuilt region's outline (edges used by one region face) is held as constraint segments, so the patch meets
the kept faces edge-for-edge. Every result is checked with rxlkit.tin.quality and an area check, and a
PatchError is raised instead of returning a bad TIN.
"""
import numpy as np
import triangle
from matplotlib.tri import Triangulation
from rxlkit.tin import plan_areas, quality, quality_ok
from scipy.spatial import cKDTree


class PatchError(RuntimeError):
    pass


def en_of(P):
    return np.asarray(P, float)[:, [1, 0]]


def edge_keys(F):
    """(3m, 2) sorted vertex pairs; row k*m + i is edge k of face i."""
    F = np.asarray(F, int)
    return np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)


def outline_edges(F):
    """Edges used by exactly one face of F (the boundary of the face set)."""
    if not len(F):
        return np.zeros((0, 2), int)
    u, c = np.unique(edge_keys(F), axis=0, return_counts=True)
    return u[c == 1]


def locate(P, F, en):
    """Index into F of the face containing each plan point (east, north); -1 where no face does.

    F must be a valid 2.5D face set (no overlaps) - any subset of a TIN that passes rxlkit quality is."""
    F = np.asarray(F, int)
    q = np.atleast_2d(np.asarray(en, float))
    if not len(F):
        return np.full(len(q), -1)
    used = np.unique(F)
    remap = np.full(len(P), -1)
    remap[used] = np.arange(len(used))
    xy = en_of(np.asarray(P)[used])
    off = xy.mean(0)
    tri = Triangulation(xy[:, 0] - off[0], xy[:, 1] - off[1], remap[F])
    return np.asarray(tri.get_trifinder()(q[:, 0] - off[0], q[:, 1] - off[1]))


def faces_touching(F, verts):
    return np.isin(np.asarray(F, int), np.asarray(list(verts), int)).any(axis=1)


def grow(F, mask, rings=1):
    """Grow a face mask by `rings` rings of edge-neighbours."""
    F = np.asarray(F, int)
    mask = np.asarray(mask, bool).copy()
    m = len(F)
    fid = np.tile(np.arange(m), 3)
    _, inv = np.unique(edge_keys(F), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    for _ in range(rings):
        hot = np.zeros(inv.max() + 1, bool)
        hot[inv[mask[fid]]] = True
        add = np.zeros(m, bool)
        add[fid[hot[inv]]] = True
        mask |= add
    return mask


def _worse(q, q0, min_area=1e-6):
    return (q["nonmanifold_edges"] > q0["nonmanifold_edges"] or q["duplicate_xy"] > q0["duplicate_xy"]
            or min(q["ccw"], q["cw"]) > min(q0["ccw"], q0["cw"])
            or (q["min_plan_area"] <= min_area < q0["min_plan_area"]))


def retriangulate(P, F, region, new_points=None, drop=None, tol=0.01):
    """Rebuild the faces in `region` (bool mask over F) from their vertices minus `drop`, plus `new_points`.

    Returns (P2, F2, info). Unused vertices are removed and indices renumbered; kept faces keep their geometry.
    Area lost is allowed only where a dropped vertex was on the TIN's outer hull."""
    P = np.asarray(P, float)
    F = np.asarray(F, int)
    region = np.asarray(region, bool)
    new = np.zeros((0, 3)) if new_points is None else np.asarray(new_points, float).reshape(-1, 3)
    drop = np.unique(np.asarray([] if drop is None else list(drop), int))
    keepF, holeF = F[~region], F[region]
    if not len(holeF):
        raise PatchError("no faces in the region to rebuild")
    if len(drop) and np.isin(keepF, drop).any():
        raise PatchError("a removed vertex is still used by a face outside the region")
    if len(new) and (locate(P, holeF, en_of(new)) < 0).any():
        raise PatchError("new points lie outside the rebuilt region")

    seg = outline_edges(holeF)
    on_lost_hull = np.isin(seg, drop).any(axis=1)
    seg = seg[~on_lost_hull]
    hv = np.setdiff1d(np.unique(holeF), drop)
    V = np.r_[P[hv], new]
    gid = np.r_[hv, len(P) + np.arange(len(new))]
    loc = np.full(len(P), -1)
    loc[hv] = np.arange(len(hv))
    xy = en_of(V)
    off = xy.mean(0)
    args = {"vertices": xy - off}
    if len(seg):
        args["segments"] = loc[seg]
    t = triangle.triangulate(args, "pcQ")
    if len(t["vertices"]) != len(V):
        raise PatchError(f"Triangle added {len(t['vertices']) - len(V)} Steiner point(s): outline segments "
                         "cross, or a new point sits on the outline (grow the region)")
    T = np.asarray(t["triangles"], int)
    T = T[locate(P, holeF, xy[T].mean(axis=1)) >= 0]

    Pall = np.r_[P, new]
    G = gid[T]
    a_region = float(np.abs(plan_areas(P, holeF)).sum())
    a_patch = float(np.abs(plan_areas(Pall, G)).sum()) if len(G) else 0.0
    if a_patch > a_region + tol:
        raise PatchError(f"patch area {a_patch:.3f} m2 exceeds region area {a_region:.3f} m2")
    if not on_lost_hull.any() and a_patch < a_region - tol:
        raise PatchError(f"patch area {a_patch:.3f} m2 leaves a gap in region area {a_region:.3f} m2")

    s = np.sign(np.sign(plan_areas(P, keepF)).sum()) if len(keepF) else 1.0
    s = s or 1.0
    if len(G):
        flip = np.sign(plan_areas(Pall, G)) != s
        G[flip] = G[flip][:, [0, 2, 1]]
    unused_new = np.setdiff1d(gid[len(hv):], G)
    if len(unused_new):
        raise PatchError(f"{len(unused_new)} new point(s) did not make it into the TIN (duplicates?)")
    F2 = np.r_[keepF, G]
    used = np.unique(F2)
    remap = np.full(len(Pall), -1)
    remap[used] = np.arange(len(used))
    P2, F2 = Pall[used], remap[F2]
    q0, q = quality(P, F), quality(P2, F2)
    if not quality_ok(q) and (quality_ok(q0) or _worse(q, q0)):
        raise PatchError(f"result fails TIN checks: {q} (input: {q0})")
    info = {"region_faces": len(holeF), "patch_faces": len(G), "removed": len(drop), "added": len(new),
            "region_area": a_region, "patch_area": a_patch, "hull_area_lost": a_region - a_patch,
            "quality": q, "input_quality_ok": quality_ok(q0)}
    return P2, F2, info


def remove_vertices(P, F, verts, tol=0.01):
    """Remove vertices and re-triangulate only the faces that used them."""
    verts = np.unique(np.asarray(list(verts), int))
    if not len(verts):
        raise PatchError("no vertices to remove")
    return retriangulate(P, F, faces_touching(F, verts), drop=verts, tol=tol)


def insert_points(P, F, points, min_sep=0.10, rings=1, tol=0.01):
    """Insert points (north, east, z) into the TIN, re-triangulating only around them.

    Points within `min_sep` (plan) of an existing vertex or an earlier new point, or outside the TIN, are
    skipped. The rebuilt region is the faces containing a point grown by `rings` edge-rings."""
    P = np.asarray(P, float)
    F = np.asarray(F, int)
    pts = np.asarray(points, float).reshape(-1, 3)
    en = en_of(pts)
    near = cKDTree(en_of(P[np.unique(F)])).query(en)[0] < min_sep if len(pts) else np.zeros(0, bool)
    dup = np.zeros(len(pts), bool)
    if len(pts) > 1:
        for i, j in sorted(cKDTree(en).query_pairs(min_sep)):
            if not dup[i] and not near[i]:
                dup[j] = True
    face = locate(P, F, en) if len(pts) else np.zeros(0, int)
    outside = face < 0
    sel = ~near & ~dup & ~outside
    info0 = {"skipped_near_vertex": int(near.sum()), "skipped_duplicate": int((dup & ~near).sum()),
             "skipped_outside": int((outside & ~near & ~dup).sum())}
    if not sel.any():
        raise PatchError(f"no points to insert ({info0})")
    region = np.zeros(len(F), bool)
    region[face[sel]] = True
    P2, F2, info = retriangulate(P, F, grow(F, region, rings), new_points=pts[sel], tol=tol)
    info.update(info0)
    return P2, F2, info
