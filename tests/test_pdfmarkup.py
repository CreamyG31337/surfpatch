"""Synthetic tests for surfpatch.pdfmarkup (affine registration, TIN/DXF/CSV footprints, Bluebeam OCG layers)."""
import ezdxf
import numpy as np
import pymupdf
import pytest

from rxlkit.landxml import Surface
from surfpatch.pdfmarkup import (
    dxf_footprint,
    en_to_pdf,
    fit_affine,
    overlay_boundaries,
    pdf_to_en,
    points_footprint,
    tin_footprint,
)


def test_affine_fit_and_inverse():
    # Synthetic ground grid around N 1000..1100, E 2000..2100 mapped with scale + rotation
    en = np.array([
        [2000.0, 1000.0],
        [2100.0, 1000.0],
        [2100.0, 1100.0],
        [2000.0, 1100.0],
    ])
    # Map to PDF points (72 pt/in) with slight rotation and offset
    xy = np.array([
        [100.0, 500.0],
        [400.0, 490.0],
        [390.0, 190.0],
        [90.0, 200.0],
    ])
    cx, cy, res = fit_affine(en, xy)
    assert res["max_m"] < 1e-6
    px, py = en_to_pdf(2050.0, 1050.0, cx, cy)
    e_back, n_back = pdf_to_en(px, py, cx, cy)
    assert (e_back, n_back) == pytest.approx((2050.0, 1050.0), abs=1e-6)


def test_overlay_boundaries_creates_ocgs_and_annotations(tmp_path):
    # Create a blank synthetic PDF sheet (11x17 landscape: 1224 x 792 pt)
    src_pdf = tmp_path / "sheet.pdf"
    doc = pymupdf.open()
    doc.new_page(width=1224, height=792)
    doc.save(src_pdf)
    doc.close()

    # Build a synthetic LandXML TIN surface (N 1000..1040, E 2000..2040)
    pts_nez = np.array([
        [1000.0, 2000.0, 100.0],
        [1000.0, 2040.0, 100.0],
        [1040.0, 2040.0, 100.0],
        [1040.0, 2000.0, 100.0],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    surf = Surface("TEST-SURF", pts_nez, faces)
    geom_tin = tin_footprint(surf)
    assert geom_tin.area == pytest.approx(1600.0, abs=1.0)

    # Synthetic DXF polyline
    dxf_path = tmp_path / "bnd.dxf"
    dxf = ezdxf.new("R2010")
    dxf.modelspace().add_lwpolyline(
        [(2010.0, 1010.0), (2030.0, 1010.0), (2030.0, 1030.0), (2010.0, 1030.0)],
        close=True,
    )
    dxf.saveas(dxf_path)
    geom_dxf = dxf_footprint(dxf_path)
    assert geom_dxf.area == pytest.approx(400.0, abs=1.0)

    # Synthetic CSV alpha shape
    rng = np.random.default_rng(7)
    pts_en = np.column_stack([rng.uniform(2005, 2035, 30), rng.uniform(1005, 1035, 30)])
    geom_pts = points_footprint(pts_en, max_edge=25.0)
    assert not geom_pts.is_empty

    pages_georef = [{
        "page": 0,
        "en": np.array([[2000.0, 1000.0], [2040.0, 1000.0], [2040.0, 1040.0], [2000.0, 1040.0]]),
        "xy": np.array([[100.0, 700.0], [900.0, 700.0], [900.0, 100.0], [100.0, 100.0]]),
        "viewport": (50.0, 50.0, 1150.0, 750.0),
    }]
    out_pdf = tmp_path / "marked.pdf"
    res = overlay_boundaries(src_pdf, out_pdf, pages_georef, [
        {"name": "01. TIN Footprint", "geom_en": geom_tin, "rgb": (0.85, 0.1, 0.1), "fill_opacity": 0.15},
        {"name": "02. DXF Limit", "geom_en": geom_dxf, "rgb": (0.1, 0.3, 0.85), "dashes": [4, 2]},
        {"name": "03. Shot Area", "geom_en": geom_pts, "rgb": (0.1, 0.65, 0.2), "on": False},
    ])

    assert res["residuals"][0]["max_m"] < 1e-6
    assert res["annotations"]["01. TIN Footprint"] == 1
    assert res["annotations"]["02. DXF Limit"] == 1
    assert res["annotations"]["03. Shot Area"] >= 1

    # Re-open PDF and confirm OCG layers and polygon annotations exist
    out_doc = pymupdf.open(out_pdf)
    try:
        ocg_names = {info["name"] for info in out_doc.get_ocgs().values()}
        assert {"01. TIN Footprint", "02. DXF Limit", "03. Shot Area"}.issubset(ocg_names)
        annots = list(out_doc[0].annots())
        assert len(annots) >= 3
    finally:
        out_doc.close()


def _donut_surface():
    # 40 x 40 m TIN (N 1000, E 2000) with a 10 x 10 m hole: 8 quads around the centre square
    xs = [2000.0, 2015.0, 2025.0, 2040.0]
    ys = [1000.0, 1015.0, 1025.0, 1040.0]
    pts = np.array([[n, e, 100.0] for n in ys for e in xs])
    faces = []
    for r in range(3):
        for c in range(3):
            if (r, c) == (1, 1):
                continue
            a, b, d, e = r * 4 + c, r * 4 + c + 1, (r + 1) * 4 + c, (r + 1) * 4 + c + 1
            faces += [[a, b, e], [a, e, d]]
    return Surface("DONUT", pts, np.array(faces))


def test_tin_hole_fill_legend_and_residual_flags(tmp_path):
    geom = tin_footprint(_donut_surface(), simplify=0)
    assert geom.area == pytest.approx(1600.0 - 100.0, abs=0.5)
    assert len(geom.interiors) == 1

    src = tmp_path / "sheet.pdf"
    d = pymupdf.open(); d.new_page(width=1224, height=792); d.save(src); d.close()
    en = np.array([[2000.0, 1000.0], [2040.0, 1000.0], [2040.0, 1040.0], [2000.0, 1040.0]])
    xy = np.array([[100.0, 700.0], [900.0, 700.0], [900.0, 100.0], [100.0, 100.0]])
    out = tmp_path / "marked.pdf"
    res = overlay_boundaries(src, out, [{"page": 0, "en": en, "xy": xy, "legend_rect": (950, 600, 1200, 760)}],
                             [{"name": "01. Donut", "geom_en": geom, "rgb": (1, 0, 0), "fill_opacity": 0.4,
                               "legend": "01. Donut TIN (with hole)"}])
    r = res["residuals"][0]
    assert r["redundant"] is True and r["n_points"] == 4
    assert r["scale_pt_per_m"] == pytest.approx((20.0, 15.0))
    assert res["annotations"]["01. Donut"] == 2            # outer + inner gap polygon annotations
    doc = pymupdf.open(out)
    try:
        page = doc[0]
        subjects = sorted(a.info["subject"] for a in page.annots())
        assert subjects == ["01. Donut", "01. Donut (inner gap)"]
        assert "00. Markup Legend" in {v["name"] for v in doc.get_ocgs().values()}
        assert "Donut TIN (with hole)" in page.get_text()
        # Hole stays unfilled (even-odd): centre of the hole is white, the ring is tinted
        pix = page.get_pixmap(dpi=72)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        hx, hy = en_to_pdf(2020.0, 1020.0, *fit_affine(en, xy)[:2])
        rx, ry = en_to_pdf(2005.0, 1005.0, *fit_affine(en, xy)[:2])
        assert tuple(a[int(hy), int(hx), :3]) == (255, 255, 255)
        assert a[int(ry), int(rx), 1] < 200
    finally:
        doc.close()


def test_fit_affine_three_points_not_redundant_and_bad_point_shows():
    en = np.array([[2000.0, 1000.0], [2100.0, 1000.0], [2000.0, 1100.0]])
    xy = np.array([[0.0, 500.0], [300.0, 500.0], [0.0, 200.0]])
    assert fit_affine(en, xy)[2]["redundant"] is False
    en4 = np.vstack([en, [2100.0, 1100.0]])
    xy4 = np.vstack([xy, [300.0 + 3.0, 200.0]])               # 1 m pick error on the 4th cross
    res = fit_affine(en4, xy4)[2]
    assert res["max_m"] > 0.2
    with pytest.raises(ValueError, match="collinear"):
        fit_affine(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))


def test_visible_xy_on_rotated_sheet(tmp_path):
    src = tmp_path / "rot.pdf"
    d = pymupdf.open(); p = d.new_page(width=600, height=400); p.set_rotation(90); d.save(src); d.close()
    # Crosses measured on the displayed (rotated, 400 x 600) sheet: square appears at x 50..150, y 50..150
    en = np.array([[2000.0, 1000.0], [2010.0, 1000.0], [2010.0, 1010.0], [2000.0, 1010.0]])
    xy_vis = np.array([[50.0, 150.0], [150.0, 150.0], [150.0, 50.0], [50.0, 50.0]])
    from shapely.geometry import box as sbox
    out = tmp_path / "rot_marked.pdf"
    overlay_boundaries(src, out, [{"page": 0, "en": en, "xy": xy_vis, "xy_space": "visible"}],
                       [{"name": "SQ", "geom_en": sbox(2002, 1002, 2008, 1008), "rgb": (1, 0, 0),
                         "fill_opacity": 0.6}])
    doc = pymupdf.open(out)
    try:
        pix = doc[0].get_pixmap(dpi=72)                      # rendered as displayed
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        ys, xs = np.nonzero((a[:, :, 0] > 200) & (a[:, :, 1] < 200))
        assert (xs.min(), xs.max()) == pytest.approx((70, 130), abs=3)
        assert (ys.min(), ys.max()) == pytest.approx((70, 130), abs=3)
    finally:
        doc.close()
