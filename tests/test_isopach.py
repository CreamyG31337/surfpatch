"""Exact isopach volumes against shapes with known answers. Made-up local coordinates (N 1000 / E 2000)."""
import numpy as np
import pytest
from scipy.spatial import Delaunay

from surfpatch import compare
from surfpatch.isopach import isopach

E0, N0 = 2000.0, 1000.0


def tin(zfun, n=11, jitter=0.0, seed=0, size=10.0):
    """TIN over a size x size square at (E0, N0); points (north, east, z)."""
    g = np.linspace(0, size, n)
    E, N = (a.ravel() for a in np.meshgrid(g, g))
    if jitter:
        rng = np.random.default_rng(seed)
        inner = (E > 0) & (E < size) & (N > 0) & (N < size)
        E = E + inner * rng.uniform(-jitter, jitter, E.size)
        N = N + inner * rng.uniform(-jitter, jitter, N.size)
    F = Delaunay(np.c_[E, N]).simplices
    return np.c_[N + N0, E + E0, zfun(E, N)], F


def test_flat_layers_different_triangulations():
    r = isopach(tin(lambda e, n: 10 + 0 * e, 11), tin(lambda e, n: 8 + 0 * e, 7, jitter=0.6))
    assert r.cut == pytest.approx(200.0, abs=1e-6)
    assert r.fill == pytest.approx(0.0, abs=1e-9)
    assert r.area_compared == pytest.approx(100.0, abs=1e-6)
    assert r.max_cut == pytest.approx(2.0)


def test_cut_and_fill_split_at_zero_line():
    r = isopach(tin(lambda e, n: e, 6, jitter=0.5), tin(lambda e, n: 5 + 0 * e, 9, jitter=0.4, seed=1),
                zero_lines=True)
    assert r.cut == pytest.approx(125.0, abs=1e-6)        # integral of (x - 5) over x 5..10, 10 m wide
    assert r.fill == pytest.approx(125.0, abs=1e-6)
    assert r.area_cut == pytest.approx(50.0, abs=1e-6)
    assert r.area_fill == pytest.approx(50.0, abs=1e-6)
    assert np.allclose(r.zero_lines[:, :, 0], E0 + 5.0)  # zero line at x = 5
    assert np.hypot(*(r.zero_lines[:, 1] - r.zero_lines[:, 0]).T).sum() == pytest.approx(10.0, abs=1e-6)


def test_boundary_clip_concave():
    # L-shaped boundary, area 3 x 8 + 5 x 3 = 39 m2, crossing triangles everywhere
    L = np.array([[1, 1], [9, 1], [9, 4], [4, 4], [4, 9], [1, 9]], float) + [E0, N0]
    r = isopach(tin(lambda e, n: 10 + 0 * e, 8, jitter=0.5), tin(lambda e, n: 7 + 0 * e, 5), polygon=L)
    assert r.area_boundary == pytest.approx(39.0)
    assert r.area_compared == pytest.approx(39.0, abs=1e-6)
    assert r.cut == pytest.approx(117.0, abs=1e-6)


def test_partial_overlap_counts_common_area_only():
    a = tin(lambda e, n: 10 + 0 * e, 6)
    P, F = tin(lambda e, n: 9 + 0 * e, 6)
    P = P + [0, 4, 0]                                      # shifted 4 m east: 6 x 10 m in common
    r = isopach(a, (P, F))
    assert r.area_compared == pytest.approx(60.0, abs=1e-6)
    assert r.cut == pytest.approx(60.0, abs=1e-6)


def test_elevation_ranges():
    # initial z = x (0..10), final flat 2: cut where x > 2, fill where x < 2
    r = isopach(tin(lambda e, n: e, 11, jitter=0.3), tin(lambda e, n: 2 + 0 * e, 5), interval=4, origin=1)
    assert np.allclose([b[:2] for b in r.bins], [(0, 1), (1, 5), (5, 9), (9, 10)], atol=1e-9)
    cut = {(round(b[0]), round(b[1])): b[2] for b in r.bins}
    fill = {(round(b[0]), round(b[1])): b[3] for b in r.bins}
    # cut column between z = 2 and z = x, per unit width 10 m
    # 1..5: x from 2..10, overlap of [2, x] with [1, 5] = min(x, 5) - 2 -> integral 2..5 of (x-2) + 5*3 = 4.5 + 15
    assert cut[(1, 5)] == pytest.approx(10 * (4.5 + 15.0), abs=1e-6)
    assert cut[(5, 9)] == pytest.approx(10 * (8.0 + 4.0 * 1.0), abs=1e-6)   # x 5..9: (x-5) -> 8, x 9..10: 4
    assert cut[(9, 10)] == pytest.approx(10 * 0.5, abs=1e-6)
    assert cut[(0, 1)] == pytest.approx(0.0, abs=1e-9)
    # fill: column [x, 2] for x < 2; in 0..1: x 0..1 -> (1 - x) = 0.5; in 1..5: x 0..1 -> 1, x 1..2 -> 2 - x = 0.5
    assert fill[(0, 1)] == pytest.approx(10 * 0.5, abs=1e-6)
    assert fill[(1, 5)] == pytest.approx(10 * 1.5, abs=1e-6)
    assert sum(b[2] for b in r.bins) == pytest.approx(r.cut, abs=1e-6)
    assert sum(b[3] for b in r.bins) == pytest.approx(r.fill, abs=1e-6)


def test_zero_line_joined_and_cli(tmp_path):
    from rxlkit.landxml import Surface, write_surfaces
    from surfpatch import io
    from surfpatch.cli import main
    a, b = tin(lambda e, n: e, 6, jitter=0.5), tin(lambda e, n: 5 + 0 * e, 9, jitter=0.4, seed=1)
    for name, (P, F) in (("ini", a), ("fin", b)):
        write_surfaces(tmp_path / f"{name}.xml", [Surface(name, P, F)])
    io.write_dxf_polylines(tmp_path / "b.dxf", [[(E0 + 1, N0 + 1), (E0 + 9, N0 + 1), (E0 + 9, N0 + 9), (E0 + 1, N0 + 9)]])
    main(["isopach", str(tmp_path / "ini.xml"), str(tmp_path / "fin.xml"), "--boundary", str(tmp_path / "b.dxf"),
          "--interval", "2", "--zero-dxf", str(tmp_path / "zero.dxf")])
    r = isopach(a, b, polygon=io.read_boundary(tmp_path / "b.dxf"), zero_lines=True)
    assert r.cut == pytest.approx(8 * 8, abs=1e-6)                     # x 5..9: integral of (x - 5) = 8, 8 m wide
    assert r.fill == pytest.approx(8 * 8, abs=1e-6)
    from surfpatch.isopach import join_segments
    lines = join_segments(r.zero_lines)
    assert len(lines) == 1 and np.allclose(lines[0][:, 0], E0 + 5)
    assert (tmp_path / "zero.dxf").exists()


def test_empty_surface_message():
    with pytest.raises(ValueError, match="rebuild-source"):
        isopach((np.zeros((0, 3)), np.zeros((0, 3), int)), tin(lambda e, n: e))


def test_matches_fine_grid_on_curved_surfaces():
    a = tin(lambda e, n: 5 + np.sin(e / 2) + 0.1 * n, 15, jitter=0.25)
    b = tin(lambda e, n: 5 + np.cos(n / 3), 12, jitter=0.3, seed=3)
    r = isopach(a, b)
    g = compare.diff(b, a, step=0.02)                     # new - old: lowered = cut
    assert r.cut == pytest.approx(g.lowered, rel=2e-3)
    assert r.fill == pytest.approx(g.raised, rel=2e-3)
