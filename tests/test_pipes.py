"""Synthetic tests for surfpatch.pipes (3D sweep, clearance check, DXF & LandXML export)."""
import ezdxf
import numpy as np
import pytest

from rxlkit.landxml import read_plan_features
from surfpatch.pipes import Pipe, min_pipe_clearance, pipes_to_dxf, pipes_to_landxml, sweep_pipe


def test_sweep_pipe_geometry_and_rails():
    # Straight pipe heading North from (N 1000, E 2000, Z 100) to (N 1010, E 2000, Z 100)
    cl = np.array([[1000.0, 2000.0, 100.0], [1005.0, 2000.0, 100.0], [1010.0, 2000.0, 100.0]])
    diam = 0.200
    sw = sweep_pipe(cl, diameter=diam, n_circ=16, n_rails=8)

    assert sw["rings"].shape == (3, 16, 3)
    assert len(sw["mesh_vertices"]) == 48
    assert len(sw["mesh_quads"]) == 32

    # Every ring vertex sits exactly R = 0.100 m from its centerline point
    for i in range(3):
        dists = np.linalg.norm(sw["rings"][i] - cl[i], axis=1)
        assert dists == pytest.approx(0.100, abs=1e-6)

    # Crown is at +0.100 Z, invert at -0.100 Z, right springline at +0.100 East
    assert sw["rails"]["crown"][:, 2] == pytest.approx(100.100, abs=1e-6)
    assert sw["rails"]["invert"][:, 2] == pytest.approx(99.900, abs=1e-6)
    assert sw["rails"]["right"][:, 1] == pytest.approx(2000.100, abs=1e-6)
    assert sw["rails"]["left"][:, 1] == pytest.approx(1999.900, abs=1e-6)


def test_min_pipe_clearance():
    # Two parallel 100 mm pipes spaced 190 mm C-C -> 90 mm clear gap
    p1 = Pipe("PF1", np.array([[1000.0, 2000.00, 100.0], [1010.0, 2000.00, 99.9]]), diameter=0.100)
    p2 = Pipe("PF2", np.array([[1000.0, 2000.19, 100.0], [1010.0, 2000.19, 99.9]]), diameter=0.100)
    # Third pipe clashing with PF2 (only 80 mm C-C -> -20 mm gap)
    p3 = Pipe("PF3", np.array([[1000.0, 2000.27, 100.0], [1010.0, 2000.27, 99.9]]), diameter=0.100)

    res = min_pipe_clearance([p1, p2, p3], step=0.25)
    assert res[0]["pair"] == ("PF2", "PF3")
    assert res[0]["center_dist"] == pytest.approx(0.080, abs=1e-3)
    assert res[0]["clear_gap"] == pytest.approx(-0.020, abs=1e-3)
    pair_12 = next(r for r in res if r["pair"] == ("PF1", "PF2"))
    assert pair_12["clear_gap"] == pytest.approx(0.090, abs=1e-3)


def test_pipes_to_dxf_and_landxml(tmp_path):
    pipes = [
        Pipe("PF1", np.array([[1000.0, 2000.0, 100.0], [1005.0, 2000.0, 99.95], [1005.0, 2005.0, 99.90]]),
             diameter=0.103),
        Pipe("PF2", np.array([[1000.0, 2000.2, 100.0], [1005.2, 2000.2, 99.95], [1005.2, 2005.0, 99.90]]),
             diameter=0.103),
    ]
    dxf_path = tmp_path / "pipes.dxf"
    pipes_to_dxf(dxf_path, pipes, n_circ=16, n_rails=8)
    doc = ezdxf.readfile(dxf_path)
    layers = {e.dxf.layer for e in doc.modelspace()}
    assert {"PIPE_CL", "PIPE_RAILS", "PIPE_RINGS", "PIPE_MESH"}.issubset(layers)

    xml_path = tmp_path / "pipes.xml"
    pipes_to_landxml(xml_path, pipes, n_circ=16, n_rails=8, include_rings=True, project_name="SYNTH")
    feats = read_plan_features(xml_path)
    names = {f.name for f in feats}
    assert "PF1" in names and "PF1_crown" in names and "PF1_invert" in names and "PF1_ring_0" in names


def _dist_to_line(p, a, u):
    v = p - a
    return np.linalg.norm(v - np.outer(v @ u, u), axis=1)


def test_sweep_miter_keeps_full_diameter_at_bend():
    # Sharp 90 deg plan bend at (N 1005, E 2000): ring must sit R from BOTH straight pipe axes
    cl = np.array([[1000.0, 2000.0, 100.0], [1005.0, 2000.0, 100.0], [1005.0, 2005.0, 100.0]])
    sw = sweep_pipe(cl, diameter=0.2, n_circ=16, n_rails=8)
    ring = sw["rings"][1]
    for a, u in ((cl[0], np.array([1.0, 0, 0])), (cl[1], np.array([0, 1.0, 0]))):
        assert _dist_to_line(ring, a, u) == pytest.approx(0.1, abs=1e-9)
    # Outside rail (left of a right turn) is at the true outer corner, not cut by 29 %
    assert sw["rails"]["left"][1][:2] == pytest.approx([1005.1, 1999.9], abs=1e-9)


def test_clearance_is_exact_for_skew_and_parallel_pipes():
    # Skew crossing: PF1 runs north at Z 100, PF2 runs east at Z 100.30 crossing above it -> 0.30 C-C exactly,
    # with no vertex near the crossing point.
    p1 = Pipe("PF1", np.array([[990.0, 2000.0, 100.0], [1013.7, 2000.0, 100.0]]), 0.1)
    p2 = Pipe("PF2", np.array([[1003.3, 1987.0, 100.3], [1003.3, 2011.1, 100.3]]), 0.1)
    r = min_pipe_clearance([p1, p2])[0]
    assert r["center_dist"] == pytest.approx(0.30, abs=1e-9)
    assert r["at_a"] == pytest.approx((1003.3, 2000.0, 100.0), abs=1e-9)
    # Parallel pipes with staggered vertices: sampling would over-report, exact gives 0.090 clear
    p3 = Pipe("PF3", np.array([[1000.0, 2000.0, 100.0], [1000.33, 2000.0, 100.0], [1010.0, 2000.0, 100.0]]), 0.1)
    p4 = Pipe("PF4", np.array([[1000.17, 2000.19, 100.0], [1010.0, 2000.19, 100.0]]), 0.1)
    assert min_pipe_clearance([p3, p4])[0]["clear_gap"] == pytest.approx(0.090, abs=1e-9)


def test_from_segments_writes_concentric_analytical_rails(tmp_path):
    from surfpatch.pipes import offset_segments
    # Straight north 2 m, then 90 deg CW (right) curve R = 1.016 to head east, grade -0.5 %
    R = 1.016
    L_arc = R * np.pi / 2
    segs = [
        {"kind": "line", "start": (1000.0, 2000.0, 100.0), "end": (1002.0, 2000.0, 99.99)},
        {"kind": "curve", "rot": "cw", "start": (1002.0, 2000.0, 99.99), "center": (1002.0, 2000.0 + R),
         "end": (1002.0 + R, 2000.0 + R, 99.99 - 0.005 * L_arc)},
    ]
    pipe = Pipe.from_segments("PF1", segs, diameter=0.103, layer="DUCT")
    assert pipe.centerline[-1] == pytest.approx([1002.0 + R, 2000.0 + R, 99.99 - 0.005 * L_arc], abs=1e-9)
    # Right springline on a right-hand curve is the inside: radius R - r; crown is R, r higher
    right = offset_segments(segs, 0.0515, 0.0)
    assert right[1]["radius"] == pytest.approx(R - 0.0515)
    assert right[0]["start"] == pytest.approx((1000.0, 2000.0515, 100.0))
    xml = tmp_path / "pf.xml"
    pipes_to_landxml(xml, [pipe], n_circ=16, n_rails=8)
    feats = {f.name: f for f in read_plan_features(xml)}
    left = feats["PF1_left"]
    assert [s["kind"] for s in left.segments] == ["line", "curve"]
    assert left.segments[1]["radius"] == pytest.approx(R + 0.0515, abs=1e-5)
    crown = feats["PF1_crown"]
    assert crown.segments[1]["radius"] == pytest.approx(R, abs=1e-5)
    assert crown.points[0][2] == pytest.approx(100.0515, abs=1e-5)


def test_long_pipe_dxf_mesh_splits_under_r12_limit(tmp_path):
    n = 2500  # 2500 rings x 16 = 40000 vertices > 32767
    cl = np.c_[1000.0 + np.arange(n) * 0.1, np.full(n, 2000.0), np.full(n, 100.0)]
    path = tmp_path / "long.dxf"
    pipes_to_dxf(path, [Pipe("P", cl, 0.1)], include_rails=False, include_rings=False)
    doc = ezdxf.readfile(path)
    meshes = [e for e in doc.modelspace() if e.dxftype() == "POLYLINE" and e.dxf.layer == "PIPE_MESH"]
    assert len(meshes) == 2
    # group 71 = vertex count, 72 = face count; each must fit a signed 16-bit int in R12
    assert all(m.dxf.m_count <= 32767 and m.dxf.n_count <= 32767 for m in meshes)
    assert sum(m.dxf.m_count for m in meshes) == (n + 1) * 16  # one shared ring at the split
