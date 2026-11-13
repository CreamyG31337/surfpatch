"""cfmap and the CloudCompare wrapper on a made-up straight channel at site-scale coordinates."""
import csv
import json

import numpy as np
import pytest

from surfpatch import cc, cfmap

N0, E0, Z0 = 5541000.0, 314900.0, 200.0
HALF, SLOPE, TOP = 5.0, 8.0, 235.0           # invert half width, wall V:H, wall top


def rxl_text():
    v = TOP - Z0
    seg = lambda code, h, dv: (f"<DistanceAndVerticalDistance><Code>{code}</Code><HorizontalDistance>{h}"
                               f"</HorizontalDistance><VerticalDistance>{dv}</VerticalDistance>"
                               f"</DistanceAndVerticalDistance>")
    tpl = f"<TemplateRecord><Name>T</Name>{seg('INVERT', HALF, 0)}{seg('WALL', v / SLOPE, v)}</TemplateRecord>"
    side = "<Type>UserDefined</Type><Name>T</Name>"
    apply = "".join(f"<ApplyTemplates><Station>{s}</Station><LeftSide>{side}</LeftSide>"
                    f"<RightSide>{side}</RightSide></ApplyTemplates>" for s in (0, 100))
    vp = "".join(f"<VerticalPoint><KeyinMethod>VPI</KeyinMethod><IntersectionPoint><Station>{s}</Station>"
                 f"<Elevation>{Z0}</Elevation></IntersectionPoint></VerticalPoint>" for s in (0, 100))
    return (f'<?xml version="1.0"?><TrimbleRoad version="6.4"><Name>TEST</Name>'
            f'<HorizontalAlignment><StartStation>0</StartStation><Straight><StartStation>0</StartStation>'
            f'<StartAzimuth>0</StartAzimuth><StartCoordinate><North>{N0}</North><East>{E0}</East></StartCoordinate>'
            f'<Length>100</Length><EndCoordinate><North>{N0 + 100}</North><East>{E0}</East></EndCoordinate>'
            f'</Straight></HorizontalAlignment><VerticalAlignment>{vp}</VerticalAlignment>{tpl}'
            f'<TemplatePositioning>{apply}</TemplatePositioning></TrimbleRoad>')


def wall_point(sta, z, side, delta):
    """(E, N, Z) of a point `delta` m off the design wall, + toward the centreline."""
    sgn = -1 if side == "L" else 1
    off = sgn * (HALF + (z - Z0) / SLOPE)
    t = np.array([sgn / SLOPE, 1.0]); t /= np.linalg.norm(t)        # up the wall in (offset, z)
    n = np.array([t[1], -t[0]])
    if np.sign(n[0]) == sgn:                                          # point it at the centreline
        n = -n
    o, zz = np.array([off, z]) + delta * n
    return E0 + o, N0 + sta, zz


@pytest.fixture
def site(tmp_path):
    rxl = tmp_path / "test.rxl"
    rxl.write_text(rxl_text())
    rng = np.random.default_rng(1)
    rows, truth = [], {}
    k = 1
    for sta in np.arange(20, 41, 1.5):
        for z in np.arange(205, 216, 1.0):
            for side in "LR":
                dlt = round(float(rng.uniform(-0.4, 0.6)), 3)
                e, n, zz = wall_point(sta, z, side, dlt)
                rows.append((k, n, e, zz, "SCAN"))
                truth[str(k)] = dlt
                k += 1
    rows.append((k, N0 + 30, E0, Z0, "FLOOR"))                         # on the invert: dropped
    rows.append((k + 1, N0 + 30, E0 + 2.0, 210.0, "FAR"))              # 4+ m off the wall: dropped
    scan = tmp_path / "261113-TEST.csv"
    with open(scan, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    cfg = {"alignment": str(rxl), "wall_names": {"L": "WEST", "R": "EAST"},
           "grid": {"major_el": [212], "minor_el": [208]}, "sheet": {"project": "TEST"}}
    return cfg, scan, truth


def read_dist(path):
    return {r["P"]: r for r in csv.DictReader(open(path))}


def test_read_tcf_drops_saved_min_max_and_duplicates(tmp_path):
    vals = [(-0.71, (255, 0, 0)), (-0.3, (255, 165, 0)), (-0.1, (255, 255, 0)), (-0.1, (255, 255, 0)),
            (0.1, (0, 172, 0)), (0.3, (0, 0, 255)), (0.71, (128, 0, 128))]
    body = "".join(f"<value><elevation>{e}</elevation><Red>{r}</Red><Green>{g}</Green><Blue>{b}</Blue></value>"
                   for e, (r, g, b) in vals)
    p = tmp_path / "c.tcf"
    p.write_text(f"<ColorMapping><version>3</version><smooth>false</smooth><elevations>{body}</elevations></ColorMapping>")
    breaks, colours = cfmap.read_tcf(p)
    assert breaks == [-0.3, -0.1, 0.1, 0.3]
    assert len(colours) == 5 and colours[0] == (1, 0, 0) and colours[-1] == (0, 0, 1)   # purple top unused


def test_section_engine(site, tmp_path):
    cfg, scan, truth = site
    cfg = dict(cfg, engine="section")
    r = cfmap.run(cfg, [scan], tmp_path / "out", build_sheet=False)
    assert r["engine"] == "section" and [w["wall"] for w in r["walls"]] == ["WEST", "EAST"]
    d = read_dist(tmp_path / "out" / "261113-TEST distances.csv")
    got = {k: float(v["dist"]) for k, v in d.items() if v["used"] == "1"}
    assert set(got) == set(truth)                                       # floor and far points dropped
    assert max(abs(got[k] - truth[k]) for k in truth) < 2e-3
    job = json.loads((tmp_path / "out" / "261113-TEST C-F MAP - sheet.json").read_text())
    assert job["pages"][0]["wall_name"] == "WEST (LEFT) WALL" and job["drawing_no"] == "261113-CF"
    bar = job["pages"][0]["legend"][0]
    assert bar["kind"] == "colorbar" and len(bar["breaks"]) == len(bar["bands"]) - 1
    assert (tmp_path / "out" / "261113-TEST C-F MAP - views.pdf").stat().st_size > 1000


def test_windows_split_long_walls_on_round_stations():
    s = np.linspace(-63.4, 110.2, 500)
    w = cfmap.windows(s, 40, 10)
    assert w[0] == (-63.4, -30.0) and w[1] == (-30.0, 10.0) and w[-1][1] == 110.2
    assert all(b - a <= 40 + 1e-9 for a, b in w[:-1])
    assert cfmap.windows(np.array([80.6, 91.7]), 40, 10) == [(80.6, 91.7)]
    # a short tail is merged into the previous window
    assert cfmap.windows(np.linspace(0, 82, 50), 40, 10) == [(0.0, 40.0), (40.0, 82.0)]


def test_mesh_faces_point_up_into_the_channel(site):
    cfg, _, _ = site
    V, F = cfmap.Design(cfmap.load_config(cfg)).mesh()
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(B - A, C - A)
    assert (n[:, 2] > 0).all()
    cen = (A + B + C) / 3                                  # horizontal part points at the CL (E0)
    assert (np.sign(n[:, 0]) == np.sign(E0 - cen[:, 0])).all()


def test_thin_and_label_mask():
    s = np.array([0.01, 0.02, 0.5, 1.2]); z = np.array([0.01, 0.03, 0.02, 0.0]); d = np.array([1., 3., 5., 7.])
    ts, tz, td = cfmap.thin(s, z, d, 1.0)
    assert sorted(td) == [3.0, 7.0]
    m = cfmap.label_mask(s, z, 1.0)
    assert m.sum() == 2 and m[2] and m[3]                                # nearest to each cell centre


needs_cc = pytest.mark.skipif(cc.find_exe() is None, reason="CloudCompare not installed")


@needs_cc
def test_cc_c2m_signed_plane(tmp_path):
    # vertical plane x = 314900 facing +x, at UTM scale
    V = np.array([[E0, N0, 0], [E0, N0 + 10, 0], [E0, N0, 10], [E0, N0 + 10, 10]], float)
    F = np.array([[0, 1, 2], [1, 3, 2]])                                 # normal +x
    P = np.array([[E0 + 0.25, N0 + 2, 3], [E0 - 0.4, N0 + 5, 5], [E0 + 1.234, N0 + 8, 7]])
    d = cc.c2m(P, V, F)
    assert np.allclose(d, [0.25, -0.4, 1.234], atol=1e-3)


@needs_cc
def test_cloudcompare_engine_matches_truth(site, tmp_path):
    cfg, scan, truth = site
    r = cfmap.run(dict(cfg, engine="cloudcompare"), [scan], tmp_path / "out", build_sheet=False)
    assert r["engine"] == "cloudcompare" and r["check_max_diff"] < 2e-3
    d = read_dist(tmp_path / "out" / "261113-TEST distances.csv")
    assert max(abs(float(d[k]["dist"]) - truth[k]) for k in truth) < 2e-3
