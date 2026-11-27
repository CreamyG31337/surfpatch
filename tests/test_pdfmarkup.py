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
