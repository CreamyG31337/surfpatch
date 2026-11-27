"""Streaming vertical bias and planar tilt checks for drone/LiDAR LAS/LAZ clouds.

Two checks, both streaming the cloud in chunks so file size doesn't matter:

- ``check_laz_bias``: cloud vs survey control/check points. Around each point the cloud is fitted with a
  local plane (robust: outliers such as vegetation, people, equipment and noise are rejected by MAD on the
  plane residuals) and evaluated at the control point, so control on a slope or at a curb edge is not biased
  by an uneven spread of cloud points. The per-point residuals ``dz = Z_cloud - Z_control`` are summarised and
  fitted with a tilt plane ``dz = a + b*(E - E0) + c*(N - N0)`` (a 1D trend along the line when the control
  is collinear, e.g. all along one road).
- ``check_laz_pair``: flight vs flight. Both clouds are gridded at ``cell`` m keeping the lowest point per
  cell, differenced, and the difference is fitted with a tilt plane that iteratively drops real change
  (``|resid| > change_tol``). A constant offset = datum/geoid problem, a trend = bad tilt/georeferencing,
  scattered = real change.
"""
import math
import os

import numpy as np

from . import io


def _load_control_points(control, order="PNEZD"):
    """Normalize control points into a list of ``(name, north, east, z, code)``."""
    if isinstance(control, (str, os.PathLike)):
        shots = io.read_shots(str(control), order=order)
        return [(str(nm), float(n), float(e), float(z), str(cd))
                for nm, (n, e, z), cd in zip(shots.name, shots.points, shots.code)]
    if hasattr(control, "points") and hasattr(control, "name"):
        codes = getattr(control, "code", [""] * len(control.name))
        return [(str(nm), float(n), float(e), float(z), str(cd))
                for nm, (n, e, z), cd in zip(control.name, control.points, codes)]
    out = []
    for idx, row in enumerate(control):
        if len(row) == 3:
            n, e, z = row
            out.append((f"P{idx + 1}", float(n), float(e), float(z), ""))
        elif len(row) == 4:
            nm, n, e, z = row
            out.append((str(nm), float(n), float(e), float(z), ""))
        else:
            nm, n, e, z, cd = row[:5]
            out.append((str(nm), float(n), float(e), float(z), str(cd)))
    return out


def _chunks(laz_path, chunk_size, classes=None):
    """Yield (x, y, z) float arrays per chunk, optionally keeping only LAS ``classes``."""
    import laspy

    with laspy.open(str(laz_path)) as f:
        for ch in f.chunk_iterator(chunk_size):
            x = np.asarray(ch.x, dtype=float)
            y = np.asarray(ch.y, dtype=float)
            z = np.asarray(ch.z, dtype=float)
            if classes is not None:
                keep = np.isin(np.asarray(ch.classification), list(classes))
                x, y, z = x[keep], y[keep], z[keep]
            if len(x):
                yield x, y, z


def _local_surface_z(dx, dy, z, stat="plane", k_mad=3.0, floor=0.01):
    """Robust cloud elevation at the control point (dx = dy = 0) from nearby points.

    ``stat``: ``"plane"`` (robust local plane, default), ``"median"`` (robust median) or ``"low"``
    (5th percentile of all points: unclassified clouds with vegetation over bare ground; low noise biases it).
    Returns ``(z, n_used, spread)`` where spread = robust sigma (1.4826 * MAD) of the inlier residuals.
    """
    if stat == "low":
        # No MAD clipping: when vegetation is the majority, clipping would keep the canopy, not the ground
        return float(np.percentile(z, 5.0)), int(len(z)), 1.4826 * float(np.median(np.abs(z - np.median(z))))
    if stat not in ("plane", "median"):
        raise ValueError(f"stat must be 'plane', 'median' or 'low', got {stat!r}")
    keep = np.ones(len(z), bool)
    use_plane = stat == "plane" and len(z) >= 6
    A = np.c_[np.ones(len(z)), dx, dy]
    for _ in range(4):
        if use_plane and keep.sum() >= 6 and np.linalg.matrix_rank(A[keep]) == 3:
            coef = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
            model = A @ coef
            z0 = float(coef[0])
        else:
            z0 = float(np.median(z[keep]))
            model = np.full(len(z), z0)
        r = z - model
        mr = float(np.median(r[keep]))      # centre on the majority (ground), not on a fit pulled up by shrubs
        s = 1.4826 * float(np.median(np.abs(r[keep] - mr)))
        new = np.abs(r - mr) <= max(k_mad * s, floor)
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    return z0, int(keep.sum()), s


def _tilt_fit(E, N, dz):
    """Least-squares tilt of dz over plan positions; handles collinear control (1D trend only)."""
    e0, n0 = float(E.mean()), float(N.mean())
    X = np.c_[E - e0, N - n0]
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    rms_along = sv[0] / math.sqrt(len(E))
    rms_across = (sv[1] if len(sv) > 1 else 0.0) / math.sqrt(len(E))
    collinear = rms_across < max(2.0, 0.05 * rms_along)
    if collinear:
        u = X @ vt[0]
        A = np.c_[np.ones(len(E)), u]
        (a, b), *_ = np.linalg.lstsq(A, dz, rcond=None)
        be, bn = b * vt[0][0], b * vt[0][1]
        resid = dz - A @ np.array([a, b])
        dof = len(E) - 2
    else:
        A = np.c_[np.ones(len(E)), X]
        (a, be, bn), *_ = np.linalg.lstsq(A, dz, rcond=None)
        resid = dz - A @ np.array([a, be, bn])
        dof = len(E) - 3
    return {
        "center_e": e0,
        "center_n": n0,
        "offset_at_center": float(a),
        "slope_e_per_100m": float(be) * 100.0,
        "slope_n_per_100m": float(bn) * 100.0,
        "mag_per_100m": float(math.hypot(be, bn) * 100.0),
        "rise_azimuth_deg": float((math.degrees(math.atan2(be, bn)) + 360.0) % 360.0),
        "plane_rms": float(np.sqrt(np.mean(resid * resid))),
        "collinear": bool(collinear),
        "dof": int(dof),
    }


def check_laz_bias(laz_path, control, *, radius=0.5, min_points=5, max_bias=0.05,
                   max_tilt_per_100m=0.05, order="PNEZD", chunk_size=5_000_000, classes=None, stat="plane"):
    """Stream ``laz_path`` and check vertical residuals and planar tilt against ``control`` points.

    Parameters
    ----------
    laz_path : str or Path
        LAS or LAZ point cloud (X = Easting, Y = Northing, Z = Elevation).
    control : str, Path, Shots, or iterable of (name, N, E, Z[, code])
        Survey control or check points.
    radius : float
        Horizontal search radius around each control point (metres).
    min_points : int
        Minimum cloud points (after outlier rejection) required to score a control point.
    max_bias, max_tilt_per_100m : float
        Warning thresholds for ``|median dz|`` (m) and fitted tilt (m per 100 m).
    classes : iterable of int, optional
        Keep only these LAS classifications (e.g. ``(2,)`` = ground) when the cloud is classified.
    stat : "plane" | "median" | "low"
        Cloud elevation estimator at each point (see ``_local_surface_z``).
    """
    ctrl = _load_control_points(control, order=order)
    if not ctrl:
        raise ValueError("no control points supplied")
    import laspy

    with laspy.open(str(laz_path)) as f:
        total_pts = int(f.header.point_count)

    cn_all = np.array([c[1] for c in ctrl])
    ce_all = np.array([c[2] for c in ctrl])
    r2 = float(radius) ** 2
    hits = [[] for _ in ctrl]
    for x, y, z in _chunks(laz_path, chunk_size, classes):
        near = ((ce_all >= x.min() - radius) & (ce_all <= x.max() + radius)
                & (cn_all >= y.min() - radius) & (cn_all <= y.max() + radius))
        if not near.any():
            continue
        order_x = np.argsort(x, kind="stable")       # one sort per chunk, then a tiny slice per control
        xs = x[order_x]
        for i in np.nonzero(near)[0]:
            lo, hi = np.searchsorted(xs, [ce_all[i] - radius, ce_all[i] + radius])
            if hi <= lo:
                continue
            idx = order_x[lo:hi]
            dx, dy = x[idx] - ce_all[i], y[idx] - cn_all[i]
            m = dx * dx + dy * dy <= r2
            if m.any():
                hits[i].append(np.c_[dx[m], dy[m], z[idx][m]])

    matched, unmatched = [], []
    for i, (nm, cn, ce, cz, cd) in enumerate(ctrl):
        pts = np.concatenate(hits[i]) if hits[i] else np.zeros((0, 3))
        if len(pts) < min_points:
            unmatched.append(nm)
            continue
        zc, n_used, spread = _local_surface_z(pts[:, 0], pts[:, 1], pts[:, 2], stat=stat)
        if n_used < min_points:
            unmatched.append(nm)
            continue
        q25, q75 = np.percentile(pts[:, 2], [25.0, 75.0])
        matched.append({
            "name": nm, "code": cd, "north": cn, "east": ce, "ctrl_z": cz,
            "cloud_z": zc, "dz": zc - cz, "iqr": float(q75 - q25), "spread": spread,
            "n_pts": int(len(pts)), "n_used": n_used,
        })

    dzs = np.array([m["dz"] for m in matched], dtype=float)
    nan = float("nan")
    summary = {
        "matched": len(matched),
        "total_control": len(ctrl),
        "median_dz": float(np.median(dzs)) if len(dzs) else nan,
        "mean_dz": float(np.mean(dzs)) if len(dzs) else nan,
        "rmse_dz": float(np.sqrt(np.mean(dzs * dzs))) if len(dzs) else nan,
        "min_dz": float(np.min(dzs)) if len(dzs) else nan,
        "max_dz": float(np.max(dzs)) if len(dzs) else nan,
    }

    tilt = None
    if len(matched) >= 3:
        tilt = _tilt_fit(np.array([m["east"] for m in matched]), np.array([m["north"] for m in matched]), dzs)
    elif len(matched) == 2:
        p, q = matched
        run = math.hypot(q["east"] - p["east"], q["north"] - p["north"])
        if run > 1.0:
            tilt = _tilt_fit(np.array([p["east"], q["east"]]), np.array([p["north"], q["north"]]), dzs)

    warnings = []
    if not matched:
        warnings.append("no control points had enough cloud points within search radius")
    else:
        if abs(summary["median_dz"]) > max_bias:
            warnings.append(f"median vertical bias {summary['median_dz']:+.3f} m exceeds +/-{max_bias:.3f} m")
        if tilt is not None and tilt["mag_per_100m"] > max_tilt_per_100m:
            how = " (along the control line only: collinear control)" if tilt["collinear"] else ""
            warnings.append(f"fitted cloud tilt {tilt['mag_per_100m']:.3f} m/100m exceeds "
                            f"{max_tilt_per_100m:.3f} m/100m{how}")

    return {
        "laz": str(laz_path), "total_points": total_pts, "radius": float(radius),
        "min_points": int(min_points), "stat": stat, "points": matched, "unmatched": unmatched,
        "summary": summary, "tilt": tilt, "status": "OK" if not warnings else "WARN", "warnings": warnings,
    }


def _grid_lowest(laz_path, e0, n0, w, h, cell, chunk_size, classes):
    g = np.full(h * w, np.inf)
    for x, y, z in _chunks(laz_path, chunk_size, classes):
        c = np.floor((x - e0) / cell).astype(np.int64)
        r = np.floor((y - n0) / cell).astype(np.int64)
        ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        np.minimum.at(g, r[ok] * w + c[ok], z[ok])
    g[np.isinf(g)] = np.nan
    return g.reshape(h, w)


def check_laz_pair(laz_a, laz_b, *, cell=1.0, block=100.0, change_tol=0.5, max_bias=0.05,
                   max_tilt_per_100m=0.05, classes=None, chunk_size=5_000_000, max_cells=25_000_000):
    """Compare two flights: lowest point per ``cell`` m grid of A minus B, tilt plane and block medians.

    The plane fit iteratively drops cells more than ``change_tol`` m from the fit (real earthwork change),
    so what's left is the systematic offset/tilt between the two clouds. ``cell`` is enlarged automatically
    if the overlap would need more than ``max_cells`` cells.
    """
    import laspy

    ext = []
    for p in (laz_a, laz_b):
        with laspy.open(str(p)) as f:
            hd = f.header
            ext.append((float(hd.mins[0]), float(hd.mins[1]), float(hd.maxs[0]), float(hd.maxs[1]),
                        int(hd.point_count)))
    e0, n0 = max(ext[0][0], ext[1][0]), max(ext[0][1], ext[1][1])
    e1, n1 = min(ext[0][2], ext[1][2]), min(ext[0][3], ext[1][3])
    if e1 <= e0 or n1 <= n0:
        raise ValueError("the two clouds do not overlap in plan")
    while math.ceil((e1 - e0) / cell) * math.ceil((n1 - n0) / cell) > max_cells:
        cell *= 2.0
    w, h = int(math.ceil((e1 - e0) / cell)), int(math.ceil((n1 - n0) / cell))
    ga = _grid_lowest(laz_a, e0, n0, w, h, cell, chunk_size, classes)
    gb = _grid_lowest(laz_b, e0, n0, w, h, cell, chunk_size, classes)
    d = ga - gb
    rr, cc = np.nonzero(np.isfinite(d))
    if len(rr) < 3:
        raise ValueError("fewer than 3 grid cells have points in both clouds")
    E = e0 + (cc + 0.5) * cell
    N = n0 + (rr + 0.5) * cell
    z = d[rr, cc]
    keep = np.ones(len(z), bool)
    for _ in range(5):
        t = _tilt_fit(E[keep], N[keep], z[keep])
        model = t["offset_at_center"] + (t["slope_e_per_100m"] * (E - t["center_e"])
                                         + t["slope_n_per_100m"] * (N - t["center_n"])) / 100.0
        new = np.abs(z - model) < change_tol
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    tilt = _tilt_fit(E[keep], N[keep], z[keep])

    nb = max(1, int(round(block / cell)))
    blocks = []
    min_cells = max(10, nb * nb // 20)
    for r0 in range(0, h, nb):
        for c0 in range(0, w, nb):
            b = d[r0:r0 + nb, c0:c0 + nb]
            k = np.isfinite(b)
            if k.sum() >= min_cells:
                blocks.append({"e": e0 + c0 * cell, "n": n0 + r0 * cell, "median_dz": float(np.median(b[k])),
                               "cells": int(k.sum())})

    med = float(np.median(z))
    warnings = []
    if abs(med) > max_bias:
        warnings.append(f"median A - B {med:+.3f} m exceeds +/-{max_bias:.3f} m")
    if tilt["mag_per_100m"] > max_tilt_per_100m:
        warnings.append(f"relative tilt {tilt['mag_per_100m']:.3f} m/100m exceeds {max_tilt_per_100m:.3f} m/100m")
    return {
        "laz_a": str(laz_a), "laz_b": str(laz_b), "points_a": ext[0][4], "points_b": ext[1][4],
        "cell": cell, "block": nb * cell, "origin": (e0, n0), "cells_both": int(len(z)),
        "cells_fit": int(keep.sum()), "median_dz": med, "tilt": tilt, "blocks": blocks, "grid_dz": d,
        "status": "OK" if not warnings else "WARN", "warnings": warnings,
    }


def format_pair_report(result):
    """Plain-text summary of ``check_laz_pair``: offset, tilt and a block-median table (north at top)."""
    t = result["tilt"]
    lines = [
        f"LAZ pair check: A = {os.path.basename(result['laz_a'])}  minus  B = {os.path.basename(result['laz_b'])}",
        f"Status: {result['status']}  |  {result['cells_both']:,} cells of {result['cell']:g} m in both, "
        f"{result['cells_fit']:,} used for the plane (rest = real change)",
        f"Median A - B: {result['median_dz']:+.3f} m",
        f"Tilt plane: {t['offset_at_center']:+.3f} m at centre, {t['mag_per_100m']:.3f} m/100m rising toward az "
        f"{t['rise_azimuth_deg']:.0f} deg (dE {t['slope_e_per_100m']:+.3f}, dN {t['slope_n_per_100m']:+.3f} m/100m; "
        f"RMS {t['plane_rms']:.3f} m)",
    ]
    lines += [f"WARNING: {w}" for w in result["warnings"]]
    if result["blocks"]:
        es = sorted({b["e"] for b in result["blocks"]})
        ns = sorted({b["n"] for b in result["blocks"]}, reverse=True)
        by = {(b["e"], b["n"]): b["median_dz"] for b in result["blocks"]}
        lines += ["", f"Median A - B by {result['block']:g} m block (row = block south edge northing):"]
        lines.append(" " * 11 + " ".join(f"E{e:>8.0f}" for e in es))
        for n in ns:
            lines.append(f"N {n:8.0f} " + " ".join(f"{by[(e, n)]:+9.2f}" if (e, n) in by else " " * 9 for e in es))
    return "\n".join(lines)


def format_bias_report(result):
    """Format ``check_laz_bias`` result as a clean plain-text survey summary."""
    s = result["summary"]
    lines = [
        f"LAZ bias check: {os.path.basename(result['laz'])} ({result['total_points']:,} pts, r={result['radius']:.2f} m)",
        f"Status: {result['status']}  |  matched {s['matched']}/{s['total_control']} control points",
    ]
    if s["matched"]:
        lines.append(
            f"Vertical dz (cloud - ctrl): median {s['median_dz']:+.3f} m, mean {s['mean_dz']:+.3f} m, "
            f"RMSE {s['rmse_dz']:.3f} m, range [{s['min_dz']:+.3f}, {s['max_dz']:+.3f}] m"
        )
    if result["tilt"] is not None:
        t = result["tilt"]
        note = ", along the control line only (collinear)" if t["collinear"] else ""
        note += ", exact fit: no redundancy" if t["dof"] <= 0 else ""
        lines.append(
            f"Tilt plane: {t['mag_per_100m']:.3f} m/100m rising toward az {t['rise_azimuth_deg']:.0f} deg "
            f"(dE {t['slope_e_per_100m']:+.3f}, dN {t['slope_n_per_100m']:+.3f} m/100m; plane RMS {t['plane_rms']:.3f} m"
            f"{note})"
        )
    for w in result["warnings"]:
        lines.append(f"WARNING: {w}")
    if result.get("unmatched"):
        lines.append(f"No cloud coverage (skipped): {', '.join(result['unmatched'])}")
    if result["points"]:
        lines.append("")
        lines.append(f"  {'Point':12s} {'Northing':>11s} {'Easting':>10s} {'Ctrl Z':>8s} {'Cloud Z':>8s} "
                     f"{'dz':>8s} {'sigma':>6s} {'used':>9s}")
        for p in result["points"]:
            lines.append(
                f"  {p['name']:12s} {p['north']:11.3f} {p['east']:10.3f} {p['ctrl_z']:8.3f} "
                f"{p['cloud_z']:8.3f} {p['dz']:+8.3f} {p['spread']:6.3f} {p['n_used']:4d}/{p['n_pts']:<4d}"
            )
    return "\n".join(lines)
