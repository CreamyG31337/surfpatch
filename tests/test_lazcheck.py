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


def _write_las(path, x, y, z, cls=None):
    hdr = laspy.LasHeader(point_format=0, version="1.2")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([2000.0, 1000.0, 100.0])
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = x, y, z
    if cls is not None:
        las.classification = cls
    las.write(str(path))


def test_lazcheck_vegetation_slope_and_collinear(tmp_path):
    rng = np.random.default_rng(3)
    # Collinear control along an east-west road (N 1000), ground rising 1:2 to the north at each point,
    # cloud only covers the uphill half of each search circle (control at a curb/edge), 30 % shrubs +1.5 m.
    ctrl = [(f"R{i}", 1000.0, 2000.0 + 50.0 * i, 100.0) for i in range(5)]
    xs, ys, zs, cs = [], [], [], []
    for _, cn, ce, cz in ctrl:
        dx = rng.uniform(-0.45, 0.45, 200)
        dy = rng.uniform(0.0, 0.45, 200)                       # uphill side only
        z = cz + 0.5 * dy + 0.02 + 0.0004 * (ce - 2000.0)       # +2 cm bias, 0.04 m/100 m rise to the east
        veg = rng.random(200) < 0.3
        z = z + np.where(veg, rng.uniform(1.0, 2.0, 200), rng.normal(0, 0.003, 200))
        xs.append(ce + dx); ys.append(cn + dy); zs.append(z); cs.append(np.where(veg, 3, 2))
    las = tmp_path / "road.las"
    _write_las(las, np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), np.concatenate(cs))

    r = check_laz_bias(las, ctrl, radius=0.5, max_tilt_per_100m=0.02)
    assert r["summary"]["median_dz"] == pytest.approx(0.02 + 0.0004 * 100, abs=0.006)
    for p in r["points"]:  # each point: bias + east trend, unaffected by slope or shrubs
        assert p["dz"] == pytest.approx(0.02 + 0.0004 * (p["east"] - 2000.0), abs=0.006)
    t = r["tilt"]
    assert t["collinear"] is True
    assert t["slope_e_per_100m"] == pytest.approx(0.04, abs=0.01)
    assert abs(t["slope_n_per_100m"]) < 1e-9                # no invented cross-line tilt
    assert any("collinear" in w for w in r["warnings"])

    # Plain median would be biased by the slope + shrubs; ground class filter + plane agrees
    r_med = check_laz_bias(las, ctrl, radius=0.5, stat="median")
    assert r_med["summary"]["median_dz"] > 0.12            # ~8 cm slope bias from one-sided coverage
    r_cls = check_laz_bias(las, ctrl, radius=0.5, classes=(2,))
    assert r_cls["summary"]["median_dz"] == pytest.approx(r["summary"]["median_dz"], abs=0.004)


def test_lazcheck_unmatched_and_two_points(tmp_path):
    x = np.r_[2000.0 + np.random.default_rng(1).uniform(-0.3, 0.3, 30), 2100.0 + np.zeros(30)]
    y = np.r_[1000.0 + np.random.default_rng(2).uniform(-0.3, 0.3, 30), 1000.0 + np.zeros(30)]
    z = np.full(60, 100.01)
    las = tmp_path / "two.las"
    _write_las(las, x, y, z)
    r = check_laz_bias(las, [("A", 1000.0, 2000.0, 100.0), ("B", 1000.0, 2100.0, 100.0),
                             ("FAR", 1500.0, 2500.0, 100.0)])
    assert r["unmatched"] == ["FAR"]
    assert r["summary"]["matched"] == 2
    assert r["tilt"]["dof"] == 0
    assert "FAR" in format_bias_report(r)


def test_laz_pair_offset_tilt_and_real_change(tmp_path):
    from surfpatch.lazcheck import check_laz_pair, format_pair_report
    rng = np.random.default_rng(5)
    n = 60000
    x = rng.uniform(2000.0, 2200.0, n)
    y = rng.uniform(1000.0, 1200.0, n)
    ground = 100.0 + 0.01 * (x - 2000.0)
    _write_las(tmp_path / "old.las", x, y, ground)
    # New flight: +0.10 m datum shift, 0.05 m/100 m tilt rising north, plus a 3 m stockpile (real change)
    z_new = ground + 0.10 + 0.0005 * (y - 1100.0)
    pile = (x - 2050.0) ** 2 + (y - 1050.0) ** 2 < 20.0 ** 2
    z_new[pile] += 3.0
    _write_las(tmp_path / "new.las", x, y, z_new)
    r = check_laz_pair(tmp_path / "new.las", tmp_path / "old.las", cell=2.0, block=100.0, max_tilt_per_100m=0.03)
    assert r["tilt"]["offset_at_center"] == pytest.approx(0.10, abs=0.01)
    assert r["tilt"]["slope_n_per_100m"] == pytest.approx(0.05, abs=0.01)
    assert abs(r["tilt"]["slope_e_per_100m"]) < 0.01
    assert r["cells_fit"] < r["cells_both"]                 # stockpile cells dropped as real change
    assert r["status"] == "WARN"
    assert len(r["blocks"]) == 4
    assert "Median A - B by 100 m block" in format_pair_report(r)
