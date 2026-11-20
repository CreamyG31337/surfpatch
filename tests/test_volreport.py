"""Tests for surfpatch.volreport, ortho_crop, deductions, and end-to-end 3-page PDF generation."""
import json

import numpy as np
import pymupdf
import pytest
from PIL import Image

from rxlkit.builder import build_rxl
from rxlkit.roadxml import Corridor
from surfpatch import cli, io, volreport


def _write_r12_3d_dxf(path, named_rings):
    """Write minimal R12 DXF with closed 3D polylines [(layer, (k, 3) n/e/z), ...]."""
    out = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009",
           "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for layer, pts in named_rings:
        out += ["0", "POLYLINE", "8", layer, "66", "1", "10", "0", "20", "0", "30", "0", "70", "9"]
        for n, e, z in pts:
            out += ["0", "VERTEX", "8", layer, "10", f"{e:.4f}", "20", f"{n:.4f}", "30", f"{z:.4f}", "70", "32"]
        out += ["0", "SEQEND", "8", layer]
    out += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\r\n".join(out) + "\r\n", encoding="ascii")


def _make_ortho(path, e0, n_top, w, h, px=0.1):
    """Create a small JPEG orthophoto + .jgw world file."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(40, 200, w, dtype=np.uint8)[None, :]
    arr[:, :, 1] = np.linspace(60, 180, h, dtype=np.uint8)[:, None]
    arr[:, :, 2] = 100
    Image.fromarray(arr).save(path, format="JPEG", quality=92)
    jgw = path.with_suffix(".jgw")
    jgw.write_text(f"{px}\n0.0\n0.0\n{-px}\n{e0 + px / 2}\n{n_top - px / 2}\n", encoding="ascii")


def test_ortho_crop_extent_half_pixel(tmp_path):
    # Ortho covering E: 1950..2050, N: 950..1050 at 0.5 m/px (200x200)
    jpg = tmp_path / "ortho.jpg"
    _make_ortho(jpg, e0=1950.0, n_top=1050.0, w=200, h=200, px=0.5)
    # Full-extent check: outer edges match world file +/- half-pixel exactly for both downsample=1 and 2
    img_full, ext_full = io.ortho_crop(jpg, (1950.0, 950.0, 2050.0, 1050.0), downsample=2)
    assert ext_full == pytest.approx((1950.0, 2050.0, 950.0, 1050.0), abs=1e-9)

    req = (1980.2, 980.3, 2020.4, 1020.7)
    for ds in (1, 2):
        cache = tmp_path / f"crop_{ds}.jpg"
        img, ext = io.ortho_crop(jpg, req, cache=cache, downsample=ds)
        eff_px = (ext[1] - ext[0]) / img.size[0]
        eff_py = (ext[3] - ext[2]) / img.size[1]
        assert eff_px == pytest.approx(0.5 * (200 / (200 // ds)), rel=1e-9)
        # Requested bbox lies inside cropped extent, padded by < 1 effective pixel on each edge
        assert 0.0 <= req[0] - ext[0] < eff_px + 1e-9
        assert 0.0 <= ext[1] - req[2] < eff_px + 1e-9
        assert 0.0 <= req[1] - ext[2] < eff_py + 1e-9
        assert 0.0 <= ext[3] - req[3] < eff_py + 1e-9
        # Second call hits the cache and returns identical extent
        img2, ext2 = io.ortho_crop(jpg, req, cache=cache, downsample=ds)
        assert ext2 == ext
        assert img2.size == img.size


def test_deduction_slope_length_on_sloped_profile(tmp_path):
    N0, E0 = 1000.0, 2000.0
    rxl = tmp_path / "sloped.rxl"
    # 3-4-5 slope: 30 m rise over 40 m run -> slope length = 50 m
    rxl.write_text(build_rxl({
        "name": "SLOPED",
        "horizontal": {"pis": [{"north": N0, "east": E0}, {"north": N0 + 40.0, "east": E0}]},
        "vertical": {"vpis": [{"station": 0.0, "elevation": 100.0}, {"station": 40.0, "elevation": 130.0}]},
    }), encoding="utf-8")
    cor = Corridor.load(rxl)
    ded = {
        "label": "PIPE",
        "area_m2": 0.01,
        "count": 6,
        "stations": [0.0, 40.0],
        "format": "LESS PIPE ({count} x {area_m2} x {slope:.1f} m) = {volume:.2f} m3",
    }
    vol, slope, text = volreport.deduction_volume(ded, cor)
    assert slope == pytest.approx(50.0, rel=1e-9)
    assert vol == pytest.approx(3.0, rel=1e-9)
    assert text == "LESS PIPE (6 x 0.01 x 50.0 m) = 3.00 m3"


def _build_synthetic_job(tmp_path, logo_value):
    N0, E0 = 1000.0, 2000.0
    rxl = tmp_path / "trench.rxl"
    rxl.write_text(build_rxl({
        "name": "SYNTH-TRENCH",
        "horizontal": {"pis": [{"north": N0, "east": E0}, {"north": N0 + 60.0, "east": E0}]},
        "vertical": {"vpis": [{"station": 0.0, "elevation": 100.5}, {"station": 60.0, "elevation": 103.5}]},
    }), encoding="utf-8")

    # Depression report: stations 10..30, flat bottom z=100, rim z=102 -> exact volume 164/3? Here 20m long:
    # Outer rim: N 1010..1030 (20 m), E 1998..2002 (4 m), z=102; inner: N 1011..1029 (18 m), E 1999..2001 (2 m), z=100
    # A_bot = 36, A_mid = 19*3 = 57, A_top = 80 -> V = (2/6)*(36 + 4*57 + 80) = 344 / 3 = 114.6666...
    rim = np.array([
        [N0 + 10.0, E0 - 2.0, 102.0],
        [N0 + 10.0, E0 + 2.0, 102.0],
        [N0 + 30.0, E0 + 2.0, 102.0],
        [N0 + 30.0, E0 - 2.0, 102.0],
    ])
    bot = np.array([
        [N0 + 11.0, E0 - 1.0, 100.0],
        [N0 + 11.0, E0 + 1.0, 100.0],
        [N0 + 29.0, E0 + 1.0, 100.0],
        [N0 + 29.0, E0 - 1.0, 100.0],
    ])
    top_pts = np.array([
        [N0 + 15.0, E0 - 1.0, 102.0],
        [N0 + 15.0, E0 + 0.0, 102.0],
        [N0 + 15.0, E0 + 1.0, 102.0],
        [N0 + 20.0, E0 - 1.0, 102.0],
        [N0 + 20.0, E0 + 0.0, 102.0],
        [N0 + 20.0, E0 + 1.0, 102.0],
        [N0 + 25.0, E0 - 1.0, 102.0],
        [N0 + 25.0, E0 + 0.0, 102.0],
        [N0 + 25.0, E0 + 1.0, 102.0],
    ])
    rep1_name = "26-299 TRENCH LIFT1"
    io.write_pnezd(tmp_path / f"{rep1_name}.csv", np.vstack([bot, rim]), code="DEPR")
    io.write_pnezd(tmp_path / f"{rep1_name} TOP.csv", top_pts, code="TOP")
    _write_r12_3d_dxf(tmp_path / f"{rep1_name} BOUNDARY.dxf",
                      [("299 DEPR TOP", rim), ("299 DEPR BOTTOM", bot)])

    # Field CSVs for source table and surf2surf report (lift 2 from z=102 to z=102.5)
    fdir = tmp_path / "field" / "10-OCT" / "CREW"
    fdir.mkdir(parents=True)
    io.write_pnezd(fdir / "26-290-AB-TRENCH.csv", bot, code="BOT")
    ns, es = np.meshgrid(np.linspace(N0 + 8.0, N0 + 32.0, 7), np.linspace(E0 - 2.5, E0 + 2.5, 5))
    l1_pts = np.c_[ns.ravel(), es.ravel(), np.full(ns.size, 102.0)]
    l2_pts = np.c_[ns.ravel(), es.ravel(), np.full(ns.size, 102.5)]
    io.write_pnezd(fdir / "26-299-AB-TRENCH.csv", l1_pts, code="LIFT1")
    io.write_pnezd(fdir / "26-301-AB-TRENCH.csv", l2_pts, code="LIFT2")

    # Small orthophoto covering alignment
    ortho_path = tmp_path / "site_ortho.jpg"
    _make_ortho(ortho_path, e0=E0 - 60.0, n_top=N0 + 120.0, w=240, h=360, px=0.5)

    job = {
        "alignment": "trench.rxl",
        "year": 2026,
        "out_dir": "out",
        "project": "9999 - SYNTHETIC TRENCH PROJECT",
        "title": "Volume Report",
        "subtitle": "9999 Test - Trench",
        "crs_note": "LOCAL GRID",
        "logo": logo_value,
        "base_label": "trench bottom",
        "ortho": {
            "image": "site_ortho.jpg",
            "cache": "ortho_cache.jpg",
            "date": "Oct 30 2026",
            "downsample": 1,
        },
        "field": {
            "glob": "field/*/CREW/{yy}-{doy:03d}-*{filter}*.csv",
            "filename_date": r"(?P<yy>\d\d)-(?P<doy>\d{3})-",
            "exclude": ["ORIG"],
            "filter": "TRENCH",
            "max_offset": 5.0,
        },
        "design": [
            {"from": 0, "to": 60, "shape": "box", "w": 0.8, "h": 0.5, "top": 0.0, "label": "encasement (design)"},
            {"from": 0, "to": 60, "shape": "circles", "d": 0.1, "x": [-0.2, 0.0, 0.2], "z": [-0.15, -0.35],
             "label": "ducts (design)"},
        ],
        "reports": [
            {
                "name": rep1_name,
                "kind": "depression",
                "bottom_csv": f"{rep1_name}.csv",
                "boundary_dxf": f"{rep1_name} BOUNDARY.dxf",
                "top_csv": f"{rep1_name} TOP.csv",
                "bottom_desc": "Trench bottom",
                "top_desc": "26-299 LIFT1",
                "sources": {"bottom": [[290, "BOT"]], "top": [[299, "LIFT1"]]},
                "base": "self",
                "deduction": {
                    "label": "DUCT VOL",
                    "area_m2": 0.01,
                    "count": 6,
                    "stations": [10.0, 30.0],
                },
            },
            {
                "name": "26-301 LIFT2",
                "kind": "surf2surf",
                "bottom_specs": [[299, "LIFT1"]],
                "top_specs": [[301, "LIFT2"]],
                "stations": [10.0, 30.0],
                "bottom_desc": "26-299 LIFT1",
                "top_desc": "26-301 LIFT2",
                "sources": {"bottom": [[299, "LIFT1"]], "top": [[301, "LIFT2"]]},
                "base": [{"report": rep1_name}],
            },
        ],
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job_path


def test_end_to_end_pdf_without_logo(tmp_path):
    job_path = _build_synthetic_job(tmp_path, logo_value="${NONEXISTENT_VOLREPORT_LOGO}")
    out_dir = tmp_path / "pdfs"
    results = volreport.run(job_path, out_dir=out_dir)
    assert len(results) == 2

    # Report 1: exact depression volume 344 / 3 m3, area 80 m2
    r1 = results[0]
    assert r1["fill"] == pytest.approx(344.0 / 3.0, rel=1e-9)
    assert r1["area_compared"] == pytest.approx(80.0, rel=1e-9)
    # Deduction: 6 * 0.01 * hypot(20, 1.0)
    expected_ded = 6 * 0.01 * np.hypot(20.0, 1.0)
    assert r1["deduction"] == pytest.approx(expected_ded, rel=1e-9)
    assert r1["net"] == pytest.approx(r1["fill"] - expected_ded, rel=1e-9)

    # Report 2: surf2surf parallel planes 20 m x 5 m = 100 m2, dz = 0.5 -> 50 m3
    r2 = results[1]
    assert r2["area_compared"] == pytest.approx(100.0, rel=1e-9)
    assert r2["fill"] == pytest.approx(50.0, rel=1e-9)

    for res in results:
        doc = pymupdf.open(res["pdf"])
        try:
            assert len(doc) == 3
            p1_text = doc[0].get_text()
            assert "NET FILL =" in p1_text
            assert "9999 - SYNTHETIC TRENCH PROJECT" in p1_text
            assert "Generated " in p1_text
            assert "Page 1 of 3" in p1_text
            # 26-299 in top/bottom description converted to Oct 26
            assert "Oct 26 LIFT1" in p1_text
            assert "Page 2 of 3" in doc[1].get_text()
            assert "Page 3 of 3" in doc[2].get_text()
        finally:
            doc.close()


def test_end_to_end_with_svg_logo_and_cli(tmp_path, capsys):
    svg = tmp_path / "logo.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="150" height="30">'
                   '<rect width="150" height="30" fill="#123456"/></svg>', encoding="utf-8")
    job_path = _build_synthetic_job(tmp_path, logo_value="logo.svg")
    out_dir = tmp_path / "cli_out"
    cli.main(["volreport", str(job_path), "26-301 LIFT2", "--out", str(out_dir)])
    out_text = capsys.readouterr().out
    assert "26-301 LIFT2: fill 50.0" in out_text
    pdf_path = out_dir / "26-301 LIFT2 Volume Report.pdf"
    assert pdf_path.is_file()
    doc = pymupdf.open(str(pdf_path))
    try:
        assert len(doc) == 3
        # Vector logo was placed as a Form XObject on page 1
        assert len(doc[0].get_xobjects()) >= 1
    finally:
        doc.close()


def test_svg_logo_css_colours_kept(tmp_path):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="150" height="30"><defs><style>'
           '.a{fill:#00376a;}.b{fill:#fdbd00;}</style></defs>'
           '<rect class="a" width="100" height="30"/><rect class="b" x="100" width="50" height="30"/></svg>')
    assert 'style="fill:#00376a"' in volreport.svg_inline_css(svg)
    (tmp_path / "logo.svg").write_text(svg, encoding="utf-8")
    pdf = tmp_path / "p.pdf"
    doc = pymupdf.open(); doc.new_page(width=612, height=792); doc.save(str(pdf)); doc.close()
    volreport.add_header_footer(str(pdf), logo_path=str(tmp_path / "logo.svg"))
    pix = pymupdf.open(str(pdf))[0].get_pixmap(dpi=72, clip=pymupdf.Rect(612 - 186, 24, 612 - 36, 54))
    cols = {tuple(pix.pixel(x, pix.height // 2)) for x in range(0, pix.width, 5)}
    assert (0, 55, 106) in cols and (253, 189, 0) in cols      # navy and yellow, not black


def test_section_station_override(tmp_path):
    job = volreport.load_job(_build_synthetic_job(tmp_path, logo_value=""))
    auto = volreport.run(job, names=["26-301 LIFT2"], out_dir=tmp_path / "a")[0]["section_station"]
    rep = next(r for r in job["reports"] if r["name"] == "26-301 LIFT2")
    rep["section_station"] = 12.5
    res = volreport.run(job, names=["26-301 LIFT2"], out_dir=tmp_path / "b")[0]
    assert res["section_station"] == 12.5 != auto
    assert "Section A-A at 0+012.5" in pymupdf.open(res["pdf"])[2].get_text()

    rep["section_station"] = [12.5, 25.0]              # list: one section page each, A-A then B-B
    res = volreport.run(job, names=["26-301 LIFT2"], out_dir=tmp_path / "c")[0]
    assert res["section_stations"] == [12.5, 25.0] and res["section_station"] == 12.5
    doc = pymupdf.open(res["pdf"])
    assert len(doc) == 4
    assert "Section A-A at 0+012.5" in doc[2].get_text() and "Section B-B at 0+025.0" in doc[3].get_text()
    assert "Page 4 of 4" in doc[3].get_text()


def test_existing_pdfs_kept_unless_overwrite(tmp_path):
    job_path = _build_synthetic_job(tmp_path, logo_value="")
    out_dir = tmp_path / "sent"
    volreport.run(job_path, names=["26-301 LIFT2"], out_dir=out_dir)
    pdf = out_dir / "26-301 LIFT2 Volume Report.pdf"
    before = pdf.read_bytes()
    with pytest.raises(FileExistsError, match="26-301 LIFT2"):
        volreport.run(job_path, out_dir=out_dir)        # checked before anything is built
    assert pdf.read_bytes() == before
    assert sorted(p.name for p in out_dir.iterdir()) == [pdf.name]
    pdf.write_bytes(b"old report")
    volreport.run(job_path, names=["26-301 LIFT2"], out_dir=out_dir, overwrite=True)
    assert pdf.read_bytes().startswith(b"%PDF")


def test_required_path_with_unset_env_var_is_clear(tmp_path):
    job = volreport.load_job(_build_synthetic_job(tmp_path, logo_value=""))
    job["alignment"] = "${NONEXISTENT_VOLREPORT_RXL}/cl.rxl"
    with pytest.raises(ValueError, match="'alignment'.*not set"):
        volreport.run(job, out_dir=tmp_path / "x")
