"""Synthetic tests: grid TINs at site-like coordinates, no site data needed.  Run: python -m pytest"""
import numpy as np
import pytest
from rxlkit.landxml import Surface, read_surface, write_surfaces
from rxlkit.tin import plan_areas, quality, quality_ok

from surfpatch import cli, compare, io, mesh

E0, N0 = 314000.0, 5541000.0


def grid_tin(n=11, step=1.0, z=lambda e, n: 200 + 0.05 * e):
    """(P, F) of an n x n vertex grid, two faces per cell, P = (north, east, z)."""
    P = np.array([(N0 + j * step, E0 + i * step, z(i * step, j * step)) for j in range(n) for i in range(n)])
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            F += [(a, b, d), (a, d, c)]
    return P, np.array(F)


def area(P, F):
    return np.abs(plan_areas(P, F)).sum()


def face_set(P, F):
    return {tuple(sorted(map(tuple, np.round(P[f], 4)))) for f in F}


def test_remove_interior_vertex_keeps_other_faces():
    P, F = grid_tin()
    v = 5 * 11 + 5
    P2, F2, info = mesh.remove_vertices(P, F, [v])
    assert len(P2) == len(P) - 1
    assert quality_ok(quality(P2, F2))
    assert area(P2, F2) == pytest.approx(area(P, F))
    untouched = face_set(P, F[~mesh.faces_touching(F, [v])])
    assert untouched <= face_set(P2, F2)
    assert info["hull_area_lost"] == pytest.approx(0, abs=1e-6)


def test_remove_hull_corner_loses_only_its_triangle():
    P, F = grid_tin()
    P2, F2, info = mesh.remove_vertices(P, F, [0])
    assert quality_ok(quality(P2, F2))
    assert info["hull_area_lost"] == pytest.approx(0.5)


def test_dropped_vertex_used_outside_region_is_refused():
    P, F = grid_tin()
    with pytest.raises(mesh.PatchError):
        mesh.retriangulate(P, F, np.arange(len(F)) == 0, drop=[60])


def test_insert_points_all_used_and_skips():
    P, F = grid_tin()
    pts = np.array([(N0 + 3.3, E0 + 4.6, 205.0), (N0 + 7.25, E0 + 2.5, 199.0),
                    (N0 + 5.0, E0 + 5.5, 201.0),          # on a grid edge
                    (N0 + 2.0, E0 + 2.02, 1.0),           # 2 cm from a vertex -> skipped
                    (N0 + 50, E0 + 50, 1.0)])             # outside -> skipped
    P2, F2, info = mesh.insert_points(P, F, pts)
    assert quality_ok(quality(P2, F2))
    assert area(P2, F2) == pytest.approx(area(P, F))
    assert info["added"] == 3 and info["skipped_near_vertex"] == 1 and info["skipped_outside"] == 1
    for p in pts[:3]:
        assert (np.abs(P2 - p).max(axis=1) < 1e-9).any()


def test_diff_volume_and_polygon():
    P, F = grid_tin()
    Q = P.copy(); Q[:, 2] += 1.0
    r = compare.diff((Q, F), (P, F), step=0.1)
    assert r.net == pytest.approx(100.0, rel=1e-6) and r.lowered == 0
    half = np.array([(E0, N0), (E0 + 5, N0), (E0 + 5, N0 + 10), (E0, N0 + 10)])
    assert compare.diff((Q, F), (P, F), step=0.1, polygon=half).net == pytest.approx(50.0, rel=1e-3)


def test_sampler_matches_plane():
    P, F = grid_tin()
    z = compare.Sampler(P, F)([E0 + 2.3, E0 + 99], [N0 + 4.1, N0])
    assert z[0] == pytest.approx(200 + 0.05 * 2.3) and np.isnan(z[1])


def test_spikes_finds_raised_vertex():
    P, F = grid_tin(z=lambda e, n: 100.0)
    P[60, 2] += 2.0
    v, dz = compare.spikes(P, F)
    assert v[np.argmax(dz)] == 60 and dz.max() == pytest.approx(2.0)


def test_sparse_region_after_removing_a_block():
    P, F = grid_tin()
    block = [j * 11 + i for j in range(3, 8) for i in range(3, 8)]
    P2, F2, _ = mesh.remove_vertices(P, F, block)
    regs = compare.sparse_regions(P2, F2, max_edge=3.0, min_area=10)
    # hole is 6 x 6 m; corner triangles of the refill have short edges, so the cluster is most of it
    assert len(regs) == 1 and 20 < regs[0]["area"] <= 36.0
    assert regs[0]["centre"] == pytest.approx([E0 + 5, N0 + 5])


def test_dxf_mesh_roundtrip(tmp_path):
    import ezdxf
    P, F = grid_tin(4)
    io.write_dxf_mesh(tmp_path / "m.dxf", [Surface("A B/C", P, F)])
    doc = ezdxf.readfile(tmp_path / "m.dxf")
    (pf,) = list(doc.modelspace())
    assert pf.is_poly_face_mesh and pf.dxf.layer == "A_B_C" and len(list(pf.faces())) == len(F)


def test_read_polygons_minimal_lwpolyline(tmp_path):
    # LWPOLYLINE without AcDb subclass markers (as older scripts wrote): ezdxf rejects it, fallback reads it
    tags = ["0", "SECTION", "2", "ENTITIES", "0", "LWPOLYLINE", "8", "B", "90", "4", "70", "1"]
    for x, y in [(E0, N0), (E0 + 2, N0), (E0 + 2, N0 + 3), (E0, N0 + 3)]:
        tags += ["10", f"{x:.3f}", "20", f"{y:.3f}"]
    (tmp_path / "b.dxf").write_text("\n".join(tags + ["0", "ENDSEC", "0", "EOF"]) + "\n")
    (pg,) = io.read_polygons(tmp_path / "b.dxf")
    assert io.polygon_area(pg) == pytest.approx(6.0)


def test_rebuild_source_snaps_to_written_precision(tmp_path):
    # two points 0.04 mm apart straddling a 1 mm rounding boundary must become one vertex
    pts = [(N0, E0, 1), (N0, E0 + 10, 1), (N0 + 10, E0, 1), (N0 + 5, E0 + 5.00049, 1), (N0 + 5, E0 + 5.00053, 1)]
    dp = " ".join(f"{n} {e} {z}" for n, e, z in pts)
    (tmp_path / "s.xml").write_text(
        '<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2"><Surfaces><Surface name="S"><SourceData>'
        f"<DataPoints><PntList3D>{dp}</PntList3D></DataPoints></SourceData></Surface></Surfaces></LandXML>")
    s = io.read_source_only(tmp_path / "s.xml")
    assert len(s.points) == 4 and quality(s.points, s.faces)["duplicate_xy"] == 0


def test_read_shots_header_and_plain(tmp_path):
    (tmp_path / "h.csv").write_text("Point,Easting,Northing,Elevation,Code\n1,10,20,30,TOR\n")
    (tmp_path / "p.csv").write_text("1,20,10,30,TOR\n2,21,11,31,OG\n")
    h, p = io.read_shots(tmp_path / "h.csv"), io.read_shots(tmp_path / "p.csv")
    assert h.points.tolist() == [[20, 10, 30]] and h.code == ["TOR"]
    assert p.points[0].tolist() == [20, 10, 30] and p.name == ["1", "2"]


def test_rebuild_source_only(tmp_path):
    P, F = grid_tin(5)
    edges = np.unique(mesh.edge_keys(F), axis=0)
    bl = "".join(f"<Breakline><PntList3D>{' '.join(f'{v:.4f}' for v in P[a])} {' '.join(f'{v:.4f}' for v in P[b])}"
                 f"</PntList3D></Breakline>" for a, b in edges)
    (tmp_path / "s.xml").write_text(
        '<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2"><Surfaces><Surface name="SRC">'
        f"<SourceData><Breaklines>{bl}</Breaklines></SourceData></Surface></Surfaces></LandXML>")
    s = io.read_source_only(tmp_path / "s.xml")
    assert s.name == "SRC" and len(s.points) == 25 and len(s.faces) == len(F)
    assert quality_ok(quality(s.points, s.faces))


def test_cli_cut_boundary(tmp_path):
    P, F = grid_tin()
    write_surfaces(tmp_path / "in.xml", [Surface("OG", P, F)])
    import ezdxf
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline([(E0 + 4.5, N0 + 4.5), (E0 + 6.5, N0 + 4.5), (E0 + 6.5, N0 + 6.5),
                                     (E0 + 4.5, N0 + 6.5)], close=True)
    doc.saveas(tmp_path / "b.dxf")
    cli.main(["cut", str(tmp_path / "in.xml"), str(tmp_path / "out.xml"), "--boundary", str(tmp_path / "b.dxf"),
              "--name", "OG REV1"])
    s = read_surface(tmp_path / "out.xml")
    assert s.name == "OG REV1" and len(s.points) == len(P) - 4
    assert quality_ok(quality(s.points, s.faces))
