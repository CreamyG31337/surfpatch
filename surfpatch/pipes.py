"""3D pipe and conduit sweep: circular rings, longitudinal rails, 3D polyface DXF meshes,
LandXML <PlanFeature> export, and pairwise clearance checks.

All input/output point arrays use (north, east, z) to match rxlkit and surfpatch conventions.
When writing DXF files, coordinates are written in CAD (X=East, Y=North, Z=Elevation) order.
"""
import math
from dataclasses import dataclass

import numpy as np

from rxlkit.landxml import PlanFeature, sample_segments, write_plan_features


@dataclass
class Pipe:
    """A 3D pipe or conduit defined by its centerline and outer diameter (metres).

    ``segments`` (optional) are the analytical line/curve dicts of ``rxlkit.landxml.PlanFeature.segments``
    with 3D start/end points. When given, ``pipes_to_landxml`` writes the centreline and the rails as true
    ``<Line>``/``<Curve>`` elements (concentric arcs) instead of chorded polylines; build such a pipe with
    ``Pipe.from_segments``.
    """
    name: str
    centerline: np.ndarray  # (n, 3) north, east, z
    diameter: float
    layer: str = ""
    color: str = ""  # "R,G,B" for LandXML trimbleCADProperties
    segments: list | None = None

    @classmethod
    def from_segments(cls, name, segments, diameter, step=0.05, **kw):
        """Pipe from analytical line/curve segments; the centreline is densified every <= ``step`` m."""
        return cls(name, sample_segments(segments, step=step), diameter, segments=list(segments), **kw)


def offset_segments(segments, h_off, v_off=0.0):
    """Parallel 3D offset of line/curve segments: ``h_off`` m horizontally (+ right), ``v_off`` m up.

    Curves stay concentric: a right offset on a CW (right-turning) curve is toward the centre (R - h), on a
    CCW curve away from it (R + h). Rails of a sloped pipe offset this way lie within r*(1/cos(grade) - 1) of
    the true pipe surface (0.2 mm for a 103 mm pipe at 9 %).
    """
    out = []
    for seg in segments:
        kind = seg["kind"].lower()
        p0, p1 = np.asarray(seg["start"], float), np.asarray(seg["end"], float)
        if kind == "line":
            d = p1[:2] - p0[:2]
            L = math.hypot(*d)
            if L <= 1e-12:
                continue
            rn = np.array([-d[1], d[0]]) / L          # right normal in (N, E)
            q0, q1 = p0[:2] + h_off * rn, p1[:2] + h_off * rn
            out.append({"kind": "line", "start": (q0[0], q0[1], p0[2] + v_off),
                        "end": (q1[0], q1[1], p1[2] + v_off)})
        elif kind == "curve":
            pc = np.asarray(seg["center"], float)[:2]
            cw = seg.get("rot", "cw").lower() == "cw"
            r = math.hypot(*(p0[:2] - pc))
            r_new = r - h_off if cw else r + h_off
            if r_new <= 1e-9:
                raise ValueError(f"offset {h_off} m collapses a curve of radius {r:.3f} m")

            def off(p):
                u = (p[:2] - pc) / math.hypot(*(p[:2] - pc))
                return pc + r_new * u
            q0, q1 = off(p0), off(p1)
            out.append({"kind": "curve", "rot": "cw" if cw else "ccw", "radius": r_new,
                        "start": (q0[0], q0[1], p0[2] + v_off), "center": (pc[0], pc[1]),
                        "end": (q1[0], q1[1], p1[2] + v_off)})
        else:
            raise ValueError(f"unsupported segment kind: {kind!r}")
    return out


def _dedupe_polyline(pts, tol=1e-9):
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 2:
        raise ValueError("centerline must have shape (n >= 2, 3) in (north, east, z)")
    kept = [arr[0]]
    for p in arr[1:]:
        if np.linalg.norm(p - kept[-1]) > tol:
            kept.append(p)
    if len(kept) < 2:
        raise ValueError("centerline has fewer than 2 distinct points")
    return np.array(kept, dtype=float)


def _station_frames(cl):
    """Compute local right-normal (horizontal) and upward-binormal frames along centerline (N, E, Z).

    Returns ``(n_vecs, b_vecs, tangents, seg_u)``; frames are ``(n, 3)`` in ``(N, E, Z)`` order, ``tangents``
    are the bisector (miter-plane normal) at each vertex, ``seg_u`` the unit segment directions.
    """
    n = len(cl)
    seg_vecs = np.diff(cl, axis=0)
    seg_lens = np.linalg.norm(seg_vecs, axis=1, keepdims=True)
    seg_u = seg_vecs / np.maximum(seg_lens, 1e-12)

    tangents = np.empty_like(cl)
    tangents[0] = seg_u[0]
    tangents[-1] = seg_u[-1]
    for i in range(1, n - 1):
        t = seg_u[i - 1] + seg_u[i]
        nt = np.linalg.norm(t)
        tangents[i] = t / nt if nt > 1e-12 else seg_u[i]

    n_vecs = np.empty_like(cl)
    b_vecs = np.empty_like(cl)
    for i, t in enumerate(tangents):
        uN, uE, uZ = float(t[0]), float(t[1]), float(t[2])
        h_norm = math.hypot(uN, uE)
        if h_norm > 1e-9:
            # Horizontal unit vector pointing right of heading (N, E): (-uE, +uN, 0) / h_norm
            n_vec = np.array([-uE / h_norm, uN / h_norm, 0.0], dtype=float)
            # Upward normal perpendicular to tangent and n_vec
            b_vec = np.array([-uZ * uN / h_norm, -uZ * uE / h_norm, h_norm], dtype=float)
            b_vec /= np.linalg.norm(b_vec)
        else:
            # Vertical riser: fix n_vec East, b_vec North
            n_vec = np.array([0.0, 1.0, 0.0], dtype=float)
            b_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        n_vecs[i] = n_vec
        b_vecs[i] = b_vec
    return n_vecs, b_vecs, tangents, seg_u


def sweep_pipe(centerline, diameter, n_circ=16, n_rails=8):
    """Sweep a circular cross-section of outer ``diameter`` (m) along a 3D centerline ``(N, E, Z)``.

    Returns a dict with:
      - ``centerline``: deduplicated ``(n_sta, 3)`` array in ``(N, E, Z)``
      - ``rings``: ``(n_sta, n_circ, 3)`` circular cross-section rings in ``(N, E, Z)``
      - ``rails``: dict of ``n_rails`` longitudinal ``(n_sta, 3)`` polylines around the circumference
        (when ``n_rails == 8``: ``right``, ``top_right``, ``crown``, ``top_left``, ``left``,
        ``bot_left``, ``invert``, ``bot_right``)
      - ``mesh_vertices``: ``(n_sta * n_circ, 3)`` shared vertices in ``(N, E, Z)``
      - ``mesh_quads``: list of ``(v0, v1, v2, v3)`` 0-based quad vertex indices
    """
    if diameter <= 0:
        raise ValueError(f"pipe diameter must be positive, got {diameter}")
    if n_circ < 4:
        raise ValueError(f"n_circ must be >= 4, got {n_circ}")
    if n_rails < 1 or n_circ % n_rails != 0:
        raise ValueError(f"n_circ ({n_circ}) must be divisible by n_rails ({n_rails})")

    cl = _dedupe_polyline(centerline)
    rad = diameter / 2.0
    n_vecs, b_vecs, tangents, seg_u = _station_frames(cl)
    alphas = np.linspace(0.0, 2.0 * math.pi, n_circ, endpoint=False)
    cos_a = np.cos(alphas)
    sin_a = np.sin(alphas)

    n_sta = len(cl)
    rings = np.empty((n_sta, n_circ, 3), dtype=float)
    for i in range(n_sta):
        v = rad * (cos_a[:, None] * n_vecs[i] + sin_a[:, None] * b_vecs[i])
        if 0 < i < n_sta - 1:
            # Miter joint: the ring lies in the bisector plane, where the pipe section is an ellipse
            # stretched by 1/cos(half bend) along the bend direction. Without this a 90 deg vertex pinches
            # the pipe to 71 % of its diameter (and rails cut the corner).
            cos_half = float(np.dot(tangents[i], seg_u[i - 1]))
            dvec = seg_u[i] - seg_u[i - 1]
            dn = np.linalg.norm(dvec)
            if dn > 1e-9 and cos_half < 1.0 - 1e-12:
                dvec /= dn
                k = 1.0 / max(cos_half, 0.25) - 1.0        # cap at ~150 deg bends (hairpins)
                v = v + k * (v @ dvec)[:, None] * dvec
        rings[i] = cl[i] + v

    step = n_circ // n_rails
    names8 = ("right", "top_right", "crown", "top_left", "left", "bot_left", "invert", "bot_right")
    rails = {}
    for k in range(n_rails):
        key = names8[k] if n_rails == 8 else f"rail_{k}"
        rails[key] = rings[:, k * step, :].copy()

    mesh_vertices = rings.reshape(-1, 3)
    mesh_quads = []
    for i in range(n_sta - 1):
        b0 = i * n_circ
        b1 = (i + 1) * n_circ
        for j in range(n_circ):
            jn = (j + 1) % n_circ
            mesh_quads.append((b0 + j, b1 + j, b1 + jn, b0 + jn))

    return {
        "centerline": cl,
        "rings": rings,
        "rails": rails,
        "mesh_vertices": mesh_vertices,
        "mesh_quads": mesh_quads,
    }


def _seg_seg_closest(p1, q1, p2, q2):
    """Vectorised closest points between 3D segments p1-q1 and p2-q2 (arrays (m, 3)).

    Returns ``(dist, c1, c2)``. Standard clamped solution (Ericson, Real-Time Collision Detection 5.1.9).
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    den = a * e - b * b
    s = np.where(den > 1e-14 * np.maximum(a * e, 1e-30), np.clip((b * f - c * e) / np.where(den > 0, den, 1), 0, 1), 0.0)
    t = (b * s + f) / e
    lo, hi = t < 0, t > 1
    t = np.clip(t, 0, 1)
    s = np.where(lo, np.clip(-c / a, 0, 1), np.where(hi, np.clip((b - c) / a, 0, 1), s))
    c1 = p1 + s[:, None] * d1
    c2 = p2 + t[:, None] * d2
    return np.linalg.norm(c1 - c2, axis=1), c1, c2


def _polyline_distance(xa, xb):
    """Exact minimum distance between two 3D polylines, KD-tree pruned. Returns (dist, point_a, point_b)."""
    from scipy.spatial import cKDTree

    pa, qa = xa[:-1], xa[1:]
    pb, qb = xb[:-1], xb[1:]
    ma, mb = 0.5 * (pa + qa), 0.5 * (pb + qb)
    ha = 0.5 * np.linalg.norm(qa - pa, axis=1)
    hb = 0.5 * np.linalg.norm(qb - pb, axis=1)
    tree = cKDTree(mb)
    # Upper bound from nearest midpoints, then every segment pair whose midpoints could be closer than it
    dmid, _ = tree.query(ma)
    ub = float(np.min(dmid))                 # segment distance <= midpoint distance
    best = (math.inf, None, None)
    for i, cand in enumerate(tree.query_ball_point(ma, ub + ha + hb.max())):
        if not cand:
            continue
        cand = np.asarray(cand)
        d, c1, c2 = _seg_seg_closest(np.repeat(pa[i:i + 1], len(cand), 0), np.repeat(qa[i:i + 1], len(cand), 0),
                                     pb[cand], qb[cand])
        k = int(np.argmin(d))
        if d[k] < best[0]:
            best = (float(d[k]), c1[k], c2[k])
    return best


def min_pipe_clearance(pipes, step=None):
    """Minimum 3D centre-to-centre distance and clear OD gap between every pair of pipes.

    Exact segment-to-segment distance between the centreline polylines (densify curved pipes first, e.g.
    ``Pipe.from_segments``), so parallel pipes are not over-reported by the sample spacing. ``step`` is
    accepted for backward compatibility and ignored. Returns dicts
    ``{"pair", "center_dist", "clear_gap", "at_a", "at_b"}`` (closest points, N E Z) sorted by
    ``clear_gap`` ascending; negative ``clear_gap`` = clash/overlap.
    """
    lines = [(p, _dedupe_polyline(p.centerline)) for p in pipes]
    results = []
    for i in range(len(lines)):
        pa, xa = lines[i]
        for j in range(i + 1, len(lines)):
            pb, xb = lines[j]
            dmin, ca, cb = _polyline_distance(xa, xb)
            results.append({
                "pair": (pa.name, pb.name),
                "center_dist": dmin,
                "clear_gap": dmin - 0.5 * (pa.diameter + pb.diameter),
                "at_a": tuple(float(v) for v in ca),
                "at_b": tuple(float(v) for v in cb),
            })
    results.sort(key=lambda r: r["clear_gap"])
    return results


def pipes_to_dxf(path, pipes, n_circ=16, n_rails=8, include_centerlines=True,
                 include_rails=True, include_rings=True, include_mesh=True,
                 layer_prefix="PIPE"):
    """Write 3D pipe centrelines, longitudinal rails, cross-section rings, and 3D polyface meshes
    to a universal R12 (AC1009) DXF file (CAD X=East, Y=North, Z=Elevation)."""
    lay_cl = f"{layer_prefix}_CL"
    lay_rails = f"{layer_prefix}_RAILS"
    lay_rings = f"{layer_prefix}_RINGS"
    lay_mesh = f"{layer_prefix}_MESH"

    out = [
        "0", "SECTION", "2", "HEADER",
        "9", "$ACADVER", "1", "AC1009",
        "9", "$INSUNITS", "70", "6",
        "0", "ENDSEC",
        "0", "SECTION", "2", "TABLES",
        "0", "TABLE", "2", "LAYER", "70", "4",
        "0", "LAYER", "2", lay_cl, "70", "0", "62", "6", "6", "CONTINUOUS",
        "0", "LAYER", "2", lay_rails, "70", "0", "62", "2", "6", "CONTINUOUS",
        "0", "LAYER", "2", lay_rings, "70", "0", "62", "3", "6", "CONTINUOUS",
        "0", "LAYER", "2", lay_mesh, "70", "0", "62", "4", "6", "CONTINUOUS",
        "0", "ENDTAB",
        "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
    ]

    def add_poly3d(layer, color, pts_nez, closed=False):
        flag = "9" if closed else "8"
        out.extend(["0", "POLYLINE", "8", layer, "62", str(color), "66", "1",
                    "10", "0.0", "20", "0.0", "30", "0.0", "70", flag])
        for n, e, z in pts_nez:
            out.extend(["0", "VERTEX", "8", layer, "62", str(color),
                        "10", f"{e:.5f}", "20", f"{n:.5f}", "30", f"{z:.5f}", "70", "32"])
        out.extend(["0", "SEQEND", "8", layer])

    for pipe in pipes:
        sw = sweep_pipe(pipe.centerline, pipe.diameter, n_circ=n_circ, n_rails=n_rails)
        if include_centerlines:
            add_poly3d(pipe.layer or lay_cl, 6, sw["centerline"], closed=False)
        if include_rails:
            for rail in sw["rails"].values():
                add_poly3d(lay_rails, 2, rail, closed=False)
        if include_rings:
            for ring in sw["rings"]:
                r_closed = np.vstack([ring, ring[0]])
                add_poly3d(lay_rings, 3, r_closed, closed=True)
        if include_mesh:
            rings = sw["rings"]
            # R12 polyface meshes hold at most 32767 vertices/faces: split long pipes into overlapping runs
            per = max(2, 32000 // n_circ)
            for a in range(0, len(rings) - 1, per - 1):
                part = rings[a:a + per]
                verts = part.reshape(-1, 3)
                ns = len(part)
                out.extend(["0", "POLYLINE", "8", lay_mesh, "62", "4", "66", "1",
                            "10", "0.0", "20", "0.0", "30", "0.0",
                            "70", "64", "71", str(len(verts)), "72", str((ns - 1) * n_circ)])
                for n, e, z in verts:
                    out.extend(["0", "VERTEX", "8", lay_mesh, "62", "4",
                                "10", f"{e:.5f}", "20", f"{n:.5f}", "30", f"{z:.5f}", "70", "192"])
                for i in range(ns - 1):
                    for j in range(n_circ):
                        jn = (j + 1) % n_circ
                        v0, v1, v2, v3 = i * n_circ + j, (i + 1) * n_circ + j, (i + 1) * n_circ + jn, i * n_circ + jn
                        out.extend(["0", "VERTEX", "8", lay_mesh, "62", "4",
                                    "10", "0.0", "20", "0.0", "30", "0.0", "70", "128",
                                    "71", str(v0 + 1), "72", str(v1 + 1),
                                    "73", str(v2 + 1), "74", str(v3 + 1)])
                out.extend(["0", "SEQEND", "8", lay_mesh])

    out.extend(["0", "ENDSEC", "0", "EOF"])
    with open(path, "w", encoding="ascii", newline="\r\n") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def pipes_to_landxml(path, pipes, n_circ=16, n_rails=8, include_centerlines=True,
                     include_rails=True, include_rings=False, project_name=""):
    """Write 3D pipe centrelines, rails, and optional rings to a TBC-ready LandXML 1.2 file.

    Pipes with ``segments`` (``Pipe.from_segments``) get analytical ``<Line>``/``<Curve>`` centrelines and
    rails (concentric arcs, exact radii R -/+ offset); others get the swept polylines.
    """
    features = []
    for pipe in pipes:
        sw = sweep_pipe(pipe.centerline, pipe.diameter, n_circ=n_circ, n_rails=n_rails)
        base_layer = pipe.layer or "PIPE_CL"
        if include_centerlines:
            features.append(PlanFeature(
                name=pipe.name, points=sw["centerline"],
                layer=base_layer, color=pipe.color or "255,0,255", segments=pipe.segments,
            ))
        if include_rails:
            rad = pipe.diameter / 2.0
            for k, (rname, rpts) in enumerate(sw["rails"].items()):
                segs = None
                if pipe.segments:
                    a = 2.0 * math.pi * k / n_rails      # same angles as sweep_pipe: 0 = right, 90 = crown
                    segs = offset_segments(pipe.segments, rad * math.cos(a), rad * math.sin(a))
                features.append(PlanFeature(
                    name=f"{pipe.name}_{rname}", points=rpts,
                    layer="PIPE_RAILS", color="255,255,0", segments=segs,
                ))
        if include_rings:
            for idx, ring in enumerate(sw["rings"]):
                features.append(PlanFeature(
                    name=f"{pipe.name}_ring_{idx}", points=ring,
                    layer="PIPE_RINGS", color="0,255,0", closed=True,
                ))
    write_plan_features(path, features, application="surfpatch", project_name=project_name)
    return path
