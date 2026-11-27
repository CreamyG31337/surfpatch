"""Synthetic tests for the drone patch: flat 20 x 20 m surface, a 6 x 6 m pit two flights both saw."""
import numpy as np
import pytest
from rxlkit.landxml import Surface, read_surface, write_surfaces
from rxlkit.tin import quality, quality_ok

from surfpatch import cli, drone, io
from surfpatch.io import polygon_area

E0, N0 = 314000.0, 5541000.0
PIT = (5.0, 5.0, 11.0, 11.0)          # local box where flights A and B see ground 0.5 m lower
SPIKE = (14.0, 14.0, 18.0, 18.0)      # local box where only flight D is low (a bad flight)


def surface(n=21):
    P = np.array([(N0 + j, E0 + i, 100.0) for j in range(n) for i in range(n)])
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            F += [(a, b, d), (a, d, c)]
    return P, np.array(F)


def flight(seed, base, pit_z=None, spike_z=None, offset=0.0, density=80):
    # 80 pts/m2 = ~20 per 0.5 m cell (cells need >= 4); real flights here are denser still
    rng = np.random.default_rng(seed)
    k = 400 * density
    x, y = rng.uniform(0, 20, k), rng.uniform(0, 20, k)
    z = np.full(k, base) + rng.normal(0, 0.01, k) + offset

    def box(b):
        return (x >= b[0]) & (x < b[2]) & (y >= b[1]) & (y < b[3])
    if pit_z is not None:
        z[box(PIT)] = pit_z + offset + rng.normal(0, 0.01, box(PIT).sum())
    if spike_z is not None:
        z[box(SPIKE)] = spike_z
    return x + E0, y + N0, z


def flights():
    return [flight(1, 100.0, pit_z=99.5),                 # A
            flight(2, 100.0, pit_z=99.5, offset=0.03),    # B agrees with A within 0.10
            flight(3, 100.4),                             # C: later fill, higher everywhere
            flight(4, 100.0, spike_z=99.0)]               # D: alone in being low at SPIKE


def test_patch_only_where_two_flights_confirm_lower_ground():
    P, F = surface()
    r = drone.drone_patch(P, F, flights())
    assert r.stats["patch_area"] == pytest.approx(36.0)
    assert r.stats["volume"] == pytest.approx(-36 * 0.485, rel=0.03)
    assert quality_ok(quality(r.points, r.faces))
    en = r.points[:, [1, 0]] - (E0, N0)
    assert en.min() > PIT[0] and en.max() < PIT[2]       # nothing at the single-flight spike
    assert set(r.stats["flight_area"]) <= {0, 1}
    assert np.nanmax(r.z_patch) <= 100.0                  # never raises the surface


def test_mask_polygon_limits_patch():
    P, F = surface()
    left = np.array([(E0, N0), (E0 + 8, N0), (E0 + 8, N0 + 20), (E0, N0 + 20)])
    r = drone.drone_patch(P, F, flights(), mask_polygon=left)
    assert r.stats["patch_area"] == pytest.approx(18.0)


def test_check_cloud_that_disagrees_blocks_patch():
    P, F = surface()
    tls = flight(9, 100.0)                                # TLS saw no pit
    r = drone.drone_patch(P, F, flights(), check=tls)
    assert r.stats["patch_area"] == 0 and len(r.faces) == 0


def test_needs_two_flights():
    P, F = surface()
    with pytest.raises(ValueError):
        drone.drone_patch(P, F, flights()[:1])


def test_region_outline_and_grid_tin():
    g = drone.Grid(0.0, 0.0, 0.5, 10, 10)
    reg = np.zeros((10, 10), bool)
    reg[2:6, 3:7] = True
    loops = drone.region_outline(g, reg)
    assert len(loops) == 1 and polygon_area(loops[0]) == pytest.approx(4.0)
    P, F = drone.grid_tin(g, np.where(reg, 1.0, np.nan), reg)
    assert len(P) == 16 and len(F) == 18 and quality_ok(quality(P, F))


def test_cli_drone(tmp_path):
    laspy = pytest.importorskip("laspy")
    P, F = surface()
    write_surfaces(tmp_path / "og.xml", [Surface("OG", P, F)])
    paths = []
    for k, (x, y, z) in enumerate(flights()):
        h = laspy.LasHeader(point_format=3, version="1.2")
        h.scales, h.offsets = [0.001] * 3, [E0, N0, 0]
        las = laspy.LasData(h)
        las.x, las.y, las.z = x, y, z
        p = tmp_path / f"flight{k}.las"
        las.write(p)
        paths.append(str(p))
    cli.main(["drone", str(tmp_path / "og.xml"), str(tmp_path / "out"), "--laz", str(tmp_path / "flight*.las"),
              "--name", "PATCH"])
    s = read_surface(tmp_path / "out" / "PATCH.xml")
    assert len(s.points) == 144 and quality_ok(quality(s.points, s.faces))
    assert (tmp_path / "out" / "PATCH.csv").read_text().count("\n") == 144
    (b,) = io.read_polygons(tmp_path / "out" / "PATCH BNDY.dxf")
    assert polygon_area(b) == pytest.approx(36.0)
