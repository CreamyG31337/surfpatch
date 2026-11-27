"""3D pipe and conduit sweep: circular rings, longitudinal rails, 3D polyface DXF meshes,
LandXML <PlanFeature> export, and pairwise clearance checks.

All input/output point arrays use (north, east, z) to match rxlkit and surfpatch conventions.
When writing DXF files, coordinates are written in CAD (X=East, Y=North, Z=Elevation) order.
"""
import math
from dataclasses import dataclass

import numpy as np

from rxlkit.landxml import PlanFeature, write_plan_features


@dataclass
class Pipe:
    """A 3D pipe or conduit defined by its centerline and outer diameter (metres)."""
    name: str
    centerline: np.ndarray  # (n, 3) north, east, z
    diameter: float
    layer: str = ""
    color: str = ""  # "R,G,B" for LandXML trimbleCADProperties


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

    Returns ``(n_vecs, b_vecs)`` of shape ``(n, 3)`` in ``(N, E, Z)`` order.
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
    return n_vecs, b_vecs


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
    n_vecs, b_vecs = _station_frames(cl)
    alphas = np.linspace(0.0, 2.0 * math.pi, n_circ, endpoint=False)
    cos_a = np.cos(alphas)
    sin_a = np.sin(alphas)

    n_sta = len(cl)
    rings = np.empty((n_sta, n_circ, 3), dtype=float)
    for i in range(n_sta):
        rings[i] = cl[i] + rad * (cos_a[:, None] * n_vecs[i] + sin_a[:, None] * b_vecs[i])

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


def _resample_polyline(pts, step=0.1):
    cl = _dedupe_polyline(pts)
    segs = np.diff(cl, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.r_[0.0, np.cumsum(lens)]
    total = float(cum[-1])
    if total <= 0:
        return cl
    npts = max(2, int(math.ceil(total / step)) + 1)
    s = np.linspace(0.0, total, npts)
    return np.column_stack([np.interp(s, cum, cl[:, k]) for k in range(3)])


def min_pipe_clearance(pipes, step=0.1):
    """Compute minimum 3D centre-to-centre distance and clear OD gap between every pair of pipes.

    Returns a list of dicts ``{"pair": (a, b), "center_dist": float, "clear_gap": float}``
    sorted by ``clear_gap`` ascending (negative ``clear_gap`` = clash/overlap).
    """
    sampled = [(p, _resample_polyline(p.centerline, step=step)) for p in pipes]
    results = []
    for i in range(len(sampled)):
        pa, xa = sampled[i]
        for j in range(i + 1, len(sampled)):
            pb, xb = sampled[j]
            diff = xa[:, None, :] - xb[None, :, :]
            dmin = float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))
            gap = dmin - 0.5 * (pa.diameter + pb.diameter)
            results.append({
                "pair": (pa.name, pb.name),
                "center_dist": dmin,
                "clear_gap": gap,
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
            verts = sw["mesh_vertices"]
            quads = sw["mesh_quads"]
            out.extend(["0", "POLYLINE", "8", lay_mesh, "62", "4", "66", "1",
                        "10", "0.0", "20", "0.0", "30", "0.0",
                        "70", "64", "71", str(len(verts)), "72", str(len(quads))])
            for n, e, z in verts:
                out.extend(["0", "VERTEX", "8", lay_mesh, "62", "4",
                            "10", f"{e:.5f}", "20", f"{n:.5f}", "30", f"{z:.5f}", "70", "192"])
            for v0, v1, v2, v3 in quads:
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
    """Write 3D pipe centrelines, rails, and optional rings to a TBC-ready LandXML 1.2 file."""
    features = []
    for pipe in pipes:
        sw = sweep_pipe(pipe.centerline, pipe.diameter, n_circ=n_circ, n_rails=n_rails)
        base_layer = pipe.layer or "PIPE_CL"
        if include_centerlines:
            features.append(PlanFeature(
                name=pipe.name, points=sw["centerline"],
                layer=base_layer, color=pipe.color or "255,0,255",
            ))
        if include_rails:
            for rname, rpts in sw["rails"].items():
                features.append(PlanFeature(
                    name=f"{pipe.name}_{rname}", points=rpts,
                    layer="PIPE_RAILS", color="255,255,0",
                ))
        if include_rings:
            for idx, ring in enumerate(sw["rings"]):
                features.append(PlanFeature(
                    name=f"{pipe.name}_ring_{idx}", points=ring,
                    layer="PIPE_RINGS", color="0,255,0", closed=True,
                ))
    write_plan_features(path, features, application="surfpatch", project_name=project_name)
    return path
