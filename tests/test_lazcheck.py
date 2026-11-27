"""Synthetic tests for surfpatch.lazcheck (streaming LAZ vertical bias & planar tilt check)."""
import laspy
import numpy as np
import pytest

from surfpatch.lazcheck import check_laz_bias, format_bias_report


def _write_synthetic_las(path, ctrl, bias_z=0.0, slope_e=0.0, slope_n=0.0):
    rng = np.random.default_rng(42)
    e0 = np.mean([c[2] for c in ctrl])
    n0 = np.mean([c[1] for c in ctrl])
    xs, ys, zs = [], [], []
    for _, cn, ce, cz in ctrl:
        # 25 points around each control point inside 0.3 m radius
        dx = rng.uniform(-0.25, 0.25, 25)
        dy = rng.uniform(-0.25, 0.25, 25)
        dz_true = bias_z + slope_e * (ce - e0) + slope_n * (cn - n0)
        xs.append(ce + dx)
        ys.append(cn + dy)
        zs.append(cz + dz_true + rng.normal(0.0, 0.004, 25))
    hdr = laspy.LasHeader(point_format=0, version="1.2")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([2000.0, 1000.0, 100.0])
    las = laspy.LasData(hdr)
    las.x = np.concatenate(xs)
    las.y = np.concatenate(ys)
    las.z = np.concatenate(zs)
    las.write(str(path))


def test_lazcheck_ok_and_tilted(tmp_path):
    # Synthetic control grid around N 1000..1200, E 2000..2200
    ctrl = [
        ("CP1", 1000.0, 2000.0, 100.000),
        ("CP2", 1000.0, 2200.0, 101.500),
        ("CP3", 1200.0, 2000.0, 102.000),
        ("CP4", 1200.0, 2200.0, 103.500),
        ("CP5", 1100.0, 2100.0, 101.750),
    ]
    csv_path = tmp_path / "control.csv"
    csv_path.write_text(
        "\n".join(f"{nm},{n:.3f},{e:.3f},{z:.3f},CTRL" for nm, n, e, z in ctrl) + "\n",
        encoding="utf-8",
    )

    # 1. Clean flight (within 1 cm)
    good_las = tmp_path / "good.las"
    _write_synthetic_las(good_las, ctrl, bias_z=0.008, slope_e=0.0, slope_n=0.0)
    r_ok = check_laz_bias(good_las, csv_path, radius=0.5, min_points=5)
    assert r_ok["status"] == "OK"
    assert r_ok["summary"]["matched"] == 5
    assert r_ok["summary"]["median_dz"] == pytest.approx(0.008, abs=0.005)
    assert r_ok["tilt"]["mag_per_100m"] < 0.01

    # 2. Tilted & biased flight (-1.20 m bias, -0.80 m/100m north tilt)
    bad_las = tmp_path / "tilted.las"
    _write_synthetic_las(bad_las, ctrl, bias_z=-1.20, slope_e=0.0, slope_n=-0.008)
    r_bad = check_laz_bias(bad_las, csv_path, radius=0.5, min_points=5)
    assert r_bad["status"] == "WARN"
    assert len(r_bad["warnings"]) == 2
    assert r_bad["summary"]["median_dz"] == pytest.approx(-1.20, abs=0.02)
    assert r_bad["tilt"]["slope_n_per_100m"] == pytest.approx(-0.80, abs=0.02)
    assert r_bad["tilt"]["mag_per_100m"] == pytest.approx(0.80, abs=0.02)
    txt = format_bias_report(r_bad)
    assert "WARN" in txt and "CP1" in txt
