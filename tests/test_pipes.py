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
