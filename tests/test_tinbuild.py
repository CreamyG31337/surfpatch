"""Tests for surfpatch.tinbuild (constrained TINs, depression method, overlap boundaries)."""
import numpy as np
import pytest

from rxlkit.builder import build_rxl
from rxlkit.roadxml import Corridor
from surfpatch import isopach, tinbuild
from surfpatch.mesh import edge_keys


def _edge_set(F):
    return {tuple(e) for e in edge_keys(F)}


def test_depression_rectangular_trench_exact_volume():
    # Outer rim: 10 m x 4 m at z = 102; inner flat bottom: 8 m x 2 m at z = 100 (local N 1000 / E 2000).
    # Exact prismoidal volume: (h / 6) * (A_bot + 4 * A_mid + A_top) = (2 / 6) * (16 + 4*27 + 40) = 164 / 3.
    N0, E0 = 1000.0, 2000.0
    rim = np.array([
        [N0 + 0.0, E0 + 0.0, 102.0],
        [N0 + 0.0, E0 + 4.0, 102.0],
        [N0 + 10.0, E0 + 4.0, 102.0],
        [N0 + 10.0, E0 + 0.0, 102.0],
    ])
    bot = np.array([
        [N0 + 1.0, E0 + 1.0, 100.0],
        [N0 + 1.0, E0 + 3.0, 100.0],
        [N0 + 9.0, E0 + 3.0, 100.0],
        [N0 + 9.0, E0 + 1.0, 100.0],
    ])
    P_all = np.vstack([rim, bot])
    bottom, lid = tinbuild.depression(P_all, [rim, bot], rim)
    r = isopach.isopach(bottom, lid, polygon=rim[:, [1, 0]])
    expected_vol = 164.0 / 3.0
    assert r.fill == pytest.approx(expected_vol, rel=1e-9)
    assert r.cut == pytest.approx(0.0, abs=1e-9)
    assert r.area_compared == pytest.approx(40.0, rel=1e-9)
    assert r.max_fill == pytest.approx(2.0, rel=1e-9)


def test_tin_from_points_keeps_breakline_and_drops_outside_concave_boundary():
    # L-shaped concave boundary + a long diagonal breakline that plain Delaunay would flip.
    N0, E0 = 1000.0, 2000.0
    bnd = np.array([
        [N0 + 0.0, E0 + 0.0, 10.0],
        [N0 + 0.0, E0 + 10.0, 10.0],
        [N0 + 2.0, E0 + 10.0, 10.0],
        [N0 + 2.0, E0 + 2.0, 10.0],
        [N0 + 10.0, E0 + 2.0, 10.0],
        [N0 + 10.0, E0 + 0.0, 10.0],
    ])
    # Breakline across the bottom arm (0,0) -> (2,10): long diagonal forced as an edge
    diag = np.array([[N0 + 0.0, E0 + 0.0, 10.0], [N0 + 2.0, E0 + 10.0, 10.0]])
    closed_bnd = np.vstack([bnd, bnd[:1]])
    P, F = tinbuild.tin_from_points(bnd, breaklines=[closed_bnd, diag], boundary=bnd[:, [1, 0]])
    edges = _edge_set(F)
    # Breakline (0, 2) must be present as a TIN edge
    assert (0, 2) in edges
    # Every closed boundary edge must be present
    for i in range(len(bnd)):
        a, b = i, (i + 1) % len(bnd)
        assert (min(a, b), max(a, b)) in edges
    # Area of L-shape = 10*2 + 8*2 = 36 m2 (concavity 8x8 = 64 m2 dropped)
    from rxlkit.tin import plan_areas
    assert float(np.abs(plan_areas(P, F)).sum()) == pytest.approx(36.0, rel=1e-9)


def test_tin_from_points_crossing_breaklines_steiner():
    N0, E0 = 1000.0, 2000.0
    corners = np.array([
        [N0 + 0.0, E0 + 0.0, 100.0],
        [N0 + 0.0, E0 + 4.0, 104.0],
        [N0 + 4.0, E0 + 4.0, 108.0],
        [N0 + 4.0, E0 + 0.0, 104.0],
    ])
    bl1 = np.array([corners[0], corners[2]])  # z goes 100 -> 108, midpoint z = 104
    bl2 = np.array([corners[1], corners[3]])  # z = 104 constant, midpoint z = 104
    with pytest.raises(ValueError, match="breaklines cross"):
        tinbuild.tin_from_points(corners, breaklines=[bl1, bl2], allow_steiner=False)
    P, F = tinbuild.tin_from_points(corners, breaklines=[bl1, bl2], allow_steiner=True)
    assert len(P) == 5
    # Steiner point at centre (N0+2, E0+2) has linearly interpolated z = 104.0
    assert P[4] == pytest.approx([N0 + 2.0, E0 + 2.0, 104.0], abs=1e-9)
    assert len(F) == 4


def test_overlap_boundary_and_parallel_planes_volume(tmp_path):
    N0, E0 = 1000.0, 2000.0
    rxl_path = tmp_path / "line.rxl"
    rxl_path.write_text(build_rxl({
        "name": "TEST",
        "horizontal": {"pis": [{"north": N0, "east": E0}, {"north": N0 + 50.0, "east": E0}]},
        "vertical": {"vpis": [{"station": 0.0, "elevation": 100.0}, {"station": 50.0, "elevation": 100.0}]},
    }), encoding="utf-8")
    cor = Corridor.load(rxl_path)

    # Bottom plane covers N: 1000..1040, E: 1994..2006 at z = 100.0
    # Top plane covers    N: 1010..1050, E: 1996..2004 at z = 100.35
    # Station band:       s = 15..35, +/- 3 m (E: 1997..2003)
    # Overlap = band (20 m x 6 m = 120 m2), thickness = 0.35 m -> volume = 42.0 m3
    def _plane(n0, n1, e0, e1, z):
        ns, es = np.meshgrid(np.linspace(n0, n1, 5), np.linspace(e0, e1, 5))
        P = np.c_[ns.ravel(), es.ravel(), np.full(ns.size, z)]
        return tinbuild.tin_from_points(P)

    bot = _plane(N0, N0 + 40, E0 - 6, E0 + 6, 100.0)
    top = _plane(N0 + 10, N0 + 50, E0 - 4, E0 + 4, 100.35)
    band = tinbuild.station_band(cor.alignment, 15.0, 35.0, half_width=3.0)
    bnd = tinbuild.overlap_boundary([bot, top], band=band)
    r = isopach.isopach(bot, top, polygon=bnd)
    assert r.area_compared == pytest.approx(120.0, rel=1e-9)
    assert r.fill == pytest.approx(42.0, rel=1e-9)
    assert r.cut == pytest.approx(0.0, abs=1e-9)
