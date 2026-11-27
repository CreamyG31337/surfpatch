"""Streaming vertical bias and planar tilt check for drone/LiDAR LAS/LAZ clouds vs survey control.

Streams arbitrarily large LAS/LAZ clouds in chunks without loading the whole file into memory,
measures median vertical residuals ``dz = Z_cloud - Z_control`` around each control/check point,
and fits a 2D tilt plane ``dz = a + b*(E - E0) + c*(N - N0)`` to catch vertical shifts or tilted
drone processing before a basemap is used for volumes or design checks.
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


def check_laz_bias(laz_path, control, *, radius=0.5, min_points=5, max_bias=0.05,
                   max_tilt_per_100m=0.05, order="PNEZD", chunk_size=5_000_000):
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
        Minimum cloud points inside ``radius`` required to score a control point.
    max_bias : float
        Warning threshold for absolute median vertical bias ``|median_dz|`` (metres).
    max_tilt_per_100m : float
        Warning threshold for fitted planar tilt magnitude (metres per 100 m horizontal run).
    """
    import laspy

    ctrl = _load_control_points(control, order=order)
    if not ctrl:
        raise ValueError("no control points supplied")

    r2 = float(radius) * float(radius)
    hits = [[] for _ in range(len(ctrl))]
    total_pts = 0

    with laspy.open(str(laz_path)) as f:
        total_pts = int(f.header.point_count)
        for ch in f.chunk_iterator(chunk_size):
            x = np.asarray(ch.x, dtype=float)
            y = np.asarray(ch.y, dtype=float)
            z = np.asarray(ch.z, dtype=float)
            if len(x) == 0:
                continue
            xmin, xmax = float(x.min()), float(x.max())
            ymin, ymax = float(y.min()), float(y.max())
            for i, (_, cn, ce, _, _) in enumerate(ctrl):
                if ce < xmin - radius or ce > xmax + radius or cn < ymin - radius or cn > ymax + radius:
                    continue
                dx = x - ce
                dy = y - cn
                m = (dx * dx + dy * dy) <= r2
                if m.any():
                    hits[i].append(z[m])

    matched = []
    for i, (nm, cn, ce, cz, cd) in enumerate(ctrl):
        if not hits[i]:
            continue
        zs = np.concatenate(hits[i])
        if len(zs) < min_points:
            continue
        med_z = float(np.median(zs))
        q25, q75 = np.percentile(zs, [25.0, 75.0])
        matched.append({
            "name": nm,
            "code": cd,
            "north": cn,
            "east": ce,
            "ctrl_z": cz,
            "cloud_z": med_z,
            "dz": med_z - cz,
            "iqr": float(q75 - q25),
            "n_pts": int(len(zs)),
        })

    dzs = np.array([m["dz"] for m in matched], dtype=float)
    summary = {
        "matched": len(matched),
        "total_control": len(ctrl),
        "median_dz": float(np.median(dzs)) if len(dzs) else float("nan"),
        "mean_dz": float(np.mean(dzs)) if len(dzs) else float("nan"),
        "rmse_dz": float(np.sqrt(np.mean(dzs * dzs))) if len(dzs) else float("nan"),
        "min_dz": float(np.min(dzs)) if len(dzs) else float("nan"),
        "max_dz": float(np.max(dzs)) if len(dzs) else float("nan"),
    }

    tilt = None
    if len(matched) >= 3:
        E = np.array([m["east"] for m in matched], dtype=float)
        N = np.array([m["north"] for m in matched], dtype=float)
        e0, n0 = float(E.mean()), float(N.mean())
        A = np.c_[np.ones(len(matched)), E - e0, N - n0]
        if np.linalg.matrix_rank(A) == 3:
            coef, _, _, _ = np.linalg.lstsq(A, dzs, rcond=None)
            a, be, bn = float(coef[0]), float(coef[1]), float(coef[2])
            resid = dzs - (A @ coef)
            mag_100 = float(math.hypot(be, bn) * 100.0)
            az_deg = float((math.degrees(math.atan2(be, bn)) + 360.0) % 360.0)
            tilt = {
                "center_e": e0,
                "center_n": n0,
                "offset_at_center": a,
                "slope_e_per_100m": be * 100.0,
                "slope_n_per_100m": bn * 100.0,
                "mag_per_100m": mag_100,
                "rise_azimuth_deg": az_deg,
                "plane_rms": float(np.sqrt(np.mean(resid * resid))),
            }

    warnings = []
    if len(matched) == 0:
        warnings.append("no control points had enough cloud points within search radius")
    else:
        if abs(summary["median_dz"]) > max_bias:
            warnings.append(f"median vertical bias {summary['median_dz']:+.3f} m exceeds +/-{max_bias:.3f} m")
        if tilt is not None and tilt["mag_per_100m"] > max_tilt_per_100m:
            warnings.append(
                f"fitted cloud tilt {tilt['mag_per_100m']:.3f} m/100m exceeds {max_tilt_per_100m:.3f} m/100m"
            )

    return {
        "laz": str(laz_path),
        "total_points": total_pts,
        "radius": float(radius),
        "min_points": int(min_points),
        "points": matched,
        "summary": summary,
        "tilt": tilt,
        "status": "OK" if not warnings else "WARN",
        "warnings": warnings,
    }


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
        lines.append(
            f"Tilt plane: {t['mag_per_100m']:.3f} m/100m rising toward az {t['rise_azimuth_deg']:.0f} deg "
            f"(dE {t['slope_e_per_100m']:+.3f}, dN {t['slope_n_per_100m']:+.3f} m/100m; plane RMS {t['plane_rms']:.3f} m)"
        )
    for w in result["warnings"]:
        lines.append(f"WARNING: {w}")
    if result["points"]:
        lines.append("")
        lines.append(f"  {'Point':12s} {'Northing':>11s} {'Easting':>10s} {'Ctrl Z':>8s} {'Cloud Z':>8s} {'dz':>8s} {'IQR':>6s} {'n':>5s}")
        for p in result["points"]:
            lines.append(
                f"  {p['name']:12s} {p['north']:11.3f} {p['east']:10.3f} {p['ctrl_z']:8.3f} "
                f"{p['cloud_z']:8.3f} {p['dz']:+8.3f} {p['iqr']:6.3f} {p['n_pts']:5d}"
            )
    return "\n".join(lines)
