"""Georeferenced Bluebeam/Acrobat layered PDF boundary markup overlays.

Fits a 2D affine transform between ground coordinates (Easting, Northing) and PDF sheet coordinates
using >= 3 grid crosses or known work points, converts LandXML TINs / DXFs / CSV shots to Shapely
geometries, clips to the sheet viewport, and writes editable PDF Polygon/Polyline markup annotations
and optional hole-aware translucent fills bound to toggleable PDF OCG layers.
"""
import ezdxf
import numpy as np
import pymupdf
from scipy.spatial import Delaunay
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union


def fit_affine(en, xy):
    """Fit a 2D affine transform mapping ``(E, N) -> (pdf_x, pdf_y)`` from >= 3 point pairs.

    Returns ``(cx, cy, residuals)`` where ``cx`` and ``cy`` are length-3 arrays
    (``x = cx[0]*E + cx[1]*N + cx[2]``) and ``residuals`` is a dict with ``max_m``, ``rms_m``,
    ``per_point_m`` (ground residuals via the inverse transform), ``n_points``, ``redundant`` (False with
    exactly 3 points: the fit is exact and the residuals prove nothing; use 4+ grid crosses),
    ``scale_pt_per_m`` (x, y) and ``skew_deg`` (non-orthogonality; a few tenths at most for a true plan sheet).
    """
    en = np.asarray(en, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if en.ndim != 2 or en.shape[1] != 2 or xy.shape != en.shape or len(en) < 3:
        raise ValueError("need at least 3 matching (E, N) and (pdf_x, pdf_y) pairs")
    A = np.c_[en[:, 0] - en[:, 0].mean(), en[:, 1] - en[:, 1].mean(), np.ones(len(en))]
    if np.linalg.matrix_rank(A, tol=1e-6 * max(1.0, float(np.abs(A[:, :2]).max()))) < 3:
        raise ValueError("control points are collinear: affine fit is singular")
    A[:, 0] += en[:, 0].mean()
    A[:, 1] += en[:, 1].mean()
    cx, _, _, _ = np.linalg.lstsq(A, xy[:, 0], rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, xy[:, 1], rcond=None)

    # Invert M = [[cx0, cx1], [cy0, cy1]] to express fit error in ground metres
    M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]], dtype=float)
    t = np.array([cx[2], cy[2]], dtype=float)
    en_back = np.linalg.solve(M, (xy - t).T).T
    errs = np.hypot(en_back[:, 0] - en[:, 0], en_back[:, 1] - en[:, 1])
    ce, cn = M[:, 0], M[:, 1]  # sheet vectors of +1 m east and +1 m north
    skew = abs(90.0 - np.degrees(np.arccos(np.clip(ce @ cn / (np.linalg.norm(ce) * np.linalg.norm(cn)), -1, 1))))
    residuals = {
        "max_m": float(np.max(errs)),
        "rms_m": float(np.sqrt(np.mean(errs * errs))),
        "per_point_m": [float(v) for v in errs],
        "n_points": int(len(en)),
        "redundant": bool(len(en) > 3),
        "scale_pt_per_m": (float(np.linalg.norm(ce)), float(np.linalg.norm(cn))),
        "skew_deg": float(skew),
    }
    return cx, cy, residuals


def en_to_pdf(e, n, cx, cy):
    """Transform ground ``(E, N)`` to PDF sheet coordinates ``(x, y)``."""
    return float(cx[0] * e + cx[1] * n + cx[2]), float(cy[0] * e + cy[1] * n + cy[2])


def pdf_to_en(x, y, cx, cy):
    """Inverse-transform PDF sheet coordinates ``(x, y)`` back to ground ``(E, N)``."""
    M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]], dtype=float)
    e, n = np.linalg.solve(M, np.array([x - cx[2], y - cy[2]], dtype=float))
    return float(e), float(n)


def tin_footprint(surface, max_edge=None, simplify=0.05):
    """Extract the ground ``(E, N)`` Shapely footprint polygon(s) of an ``rxlkit.landxml.Surface``.

    If ``max_edge`` is given, triangles with any plan edge longer than ``max_edge`` m are dropped
    before unioning (useful to exclude long bridging triangles).
    """
    pts_en = np.asarray(surface.points, dtype=float)[:, [1, 0]]
    faces = np.asarray(surface.faces, dtype=int)
    if len(faces) == 0:
        return MultiPolygon()
    return _union_triangles(pts_en[faces], max_edge=max_edge, min_area=1e-4, simplify=simplify)


def _union_triangles(tri_xy, max_edge=None, min_area=0.0, simplify=0.0):
    """Union of (m, 3, 2) plan triangles, vectorised. TIN/Delaunay triangles share edges exactly, so the
    fast coverage union is used; overlapping input falls back to a general union."""
    import shapely

    t = np.asarray(tri_xy, dtype=float)
    keep = np.ones(len(t), bool)
    if max_edge is not None:
        edges = np.linalg.norm(t - np.roll(t, -1, axis=1), axis=2)
        keep &= edges.max(axis=1) <= max_edge
    area = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                        - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
    keep &= area > min_area
    if not keep.any():
        return MultiPolygon()
    polys = shapely.polygons(np.concatenate([t[keep], t[keep][:, :1]], axis=1))
    try:
        geom = shapely.coverage_union_all(polys)
        if not geom.is_valid:
            raise ValueError
    except Exception:
        geom = unary_union(polys)
    geom = geom.buffer(0.01).buffer(-0.01)
    if simplify and simplify > 0:
        geom = geom.simplify(simplify, preserve_topology=True)
    return geom


def points_footprint(points_en, max_edge=25.0, simplify=0.08):
    """Build tight alpha-shape polygon(s) from ``(E, N)`` points by keeping Delaunay triangles
    whose longest edge is ``<= max_edge`` m."""
    en = np.unique(np.asarray(points_en, dtype=float), axis=0)
    if len(en) < 3:
        return MultiPolygon()
    tri = Delaunay(en)
    return _union_triangles(en[tri.simplices], max_edge=max_edge, min_area=0.05, simplify=simplify)


def dxf_footprint(path, as_lines=False):
    """Read DXF LWPOLYLINE/POLYLINE/LINE entities into a combined Shapely ``(E, N)`` geometry."""
    doc = ezdxf.readfile(path)
    geoms = []
    for e in doc.modelspace():
        et = e.dxftype()
        pts, closed = None, False
        if et == "LWPOLYLINE":
            pts = [(pt[0], pt[1]) for pt in e.get_points()]
            closed = bool(e.closed)
        elif et == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            closed = bool(e.is_closed)
        elif et == "LINE":
            pts = [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
        if not pts or len(pts) < 2:
            continue
        if (not as_lines and len(pts) >= 3
                and (closed or np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1.0)):
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                geoms.append(poly)
        else:
            geoms.append(LineString(pts))
    return unary_union(geoms) if geoms else MultiPolygon()


def project_geom_to_pdf(geom_en, cx, cy, clip_rect=None, min_poly_area=10.0, min_line_len=3.0):
    """Project Shapely ``(E, N)`` polygons/lines to PDF points and optionally clip to ``clip_rect``."""
    if geom_en is None or geom_en.is_empty:
        return [], []
    clip_box = box(*clip_rect) if clip_rect is not None else None

    def xform(coords):
        return [en_to_pdf(e, n, cx, cy) for e, n in coords]

    polys_pdf, lines_pdf = [], []
    items = list(geom_en.geoms) if hasattr(geom_en, "geoms") else [geom_en]
    for it in items:
        if isinstance(it, Polygon) and not it.is_empty:
            pp = Polygon(xform(it.exterior.coords), [xform(r.coords) for r in it.interiors])
            if not pp.is_valid:
                pp = pp.buffer(0)
            cl = pp.intersection(clip_box) if clip_box is not None else pp
            if cl.is_empty:
                continue
            for g in (list(cl.geoms) if hasattr(cl, "geoms") else [cl]):
                if isinstance(g, Polygon) and g.area >= min_poly_area:
                    polys_pdf.append(g)
        elif isinstance(it, LineString) and not it.is_empty:
            ll = LineString(xform(it.coords))
            cl = ll.intersection(clip_box) if clip_box is not None else ll
            if cl.is_empty:
                continue
            for g in (list(cl.geoms) if hasattr(cl, "geoms") else [cl]):
                if isinstance(g, LineString) and g.length >= min_line_len:
                    lines_pdf.append(g)
    return polys_pdf, lines_pdf


def draw_layer_on_page(page, polys_pdf, lines_pdf, *, ocg, rgb=(0.85, 0.1, 0.1), lw=1.5,
                       dashes=None, fill_opacity=0.0, subject="Boundary", comment="",
                       author="Survey Markup", stroke_opacity=0.92):
    """Draw hole-aware translucent fill + Bluebeam/Acrobat editable markup annotations onto ``page``,
    all bound to PDF OCG layer ``ocg``. Returns the number of annotations added."""
    if not polys_pdf and not lines_pdf:
        return 0

    if fill_opacity > 0 and polys_pdf:
        for poly in polys_pdf:
            sh = page.new_shape()
            sh.draw_polyline([pymupdf.Point(x, y) for x, y in list(poly.exterior.coords)])
            for hole in poly.interiors:
                sh.draw_polyline([pymupdf.Point(x, y) for x, y in list(hole.coords)])
            sh.finish(color=None, fill=rgb, fill_opacity=fill_opacity,
                      even_odd=True, closePath=True, oc=ocg)
            sh.commit(overlay=True)

    count = 0
    for poly in polys_pdf:
        rings = [("outer", poly.exterior)] + [("hole", h) for h in poly.interiors]
        for rkind, ring in rings:
            coords = list(ring.coords)[:-1]
            if len(coords) < 3:
                continue
            ann = page.add_polygon_annot([pymupdf.Point(x, y) for x, y in coords])
            ann.set_colors(stroke=rgb)
            bcfg = {"width": lw}
            if dashes:
                bcfg["dashes"] = dashes
            ann.set_border(**bcfg)
            ann.set_opacity(stroke_opacity)
            ann.set_oc(ocg)
            lbl = subject if rkind == "outer" else f"{subject} (inner gap)"
            ann.set_info(title=author, subject=lbl, content=comment)
            ann.update()
            count += 1

    for ln in lines_pdf:
        coords = list(ln.coords)
        if len(coords) < 2:
            continue
        ann = page.add_polyline_annot([pymupdf.Point(x, y) for x, y in coords])
        ann.set_colors(stroke=rgb)
        bcfg = {"width": lw}
        if dashes:
            bcfg["dashes"] = dashes
        ann.set_border(**bcfg)
        ann.set_opacity(stroke_opacity)
        ann.set_oc(ocg)
        ann.set_info(title=author, subject=subject, content=comment)
        ann.update()
        count += 1

    return count


def _pdf_dashes(dashes):
    return f"[{' '.join(f'{d:g}' for d in dashes)}] 0" if dashes else None


def draw_legend(page, rect, layers, *, ocg=None, title="BOUNDARY LAYERS", note="", columns=None,
                row_h=15.2, fontsize=7.6):
    """Draw a boxed legend (title bar + one swatch per layer, 1 or 2 columns) on ``page`` in ``rect``
    ``(x0, y0, x1, y1)`` PDF points, bound to OCG ``ocg`` so it toggles like the layers.

    ``layers`` are the same dicts as ``overlay_boundaries`` (``name``, ``rgb``, optional ``dashes``,
    ``fill_opacity``, and ``legend`` text, default ``name``). ``columns`` defaults to 2 when the layers
    don't fit one column. Returns the number of rows that did not fit (0 = all shown). Text is upright in
    unrotated page space, so on a /Rotate 90 sheet it reads sideways.
    """
    x0, y0, x1, y1 = rect
    oc = ocg if ocg is not None else 0
    dark = (0.15, 0.15, 0.15)
    sh = page.new_shape()
    sh.draw_rect(pymupdf.Rect(x0, y0, x1, y1))
    sh.finish(color=dark, fill=(1, 1, 1), fill_opacity=1.0, width=1.4, oc=oc)
    sh.draw_rect(pymupdf.Rect(x0, y0, x1, y0 + 21))
    sh.finish(color=dark, fill=(0.12, 0.31, 0.47), width=1.0, oc=oc)
    sh.commit(overlay=True)
    page.insert_text(pymupdf.Point(x0 + 8, y0 + 14.5), title, fontsize=9.2, fontname="hebo",
                     color=(1, 1, 1), overlay=True, oc=oc)

    body_h = (y1 - (12 if note else 4)) - (y0 + 35) + row_h
    fit_rows = max(1, int(body_h // row_h))
    n = len(layers)
    cols = columns or (1 if n <= fit_rows else 2)
    per_col = (n + cols - 1) // cols
    col_w = (x1 - x0) / cols
    dropped = 0
    for idx, lyr in enumerate(layers):
        col, row = divmod(idx, per_col)
        if row >= fit_rows:
            dropped += 1
            continue
        cx = x0 + 8 + col * col_w
        cy = y0 + 35 + row * row_h
        rgb = lyr.get("rgb", (0.85, 0.1, 0.1))
        fo = min(0.32, lyr.get("fill_opacity", 0.0) * 2.5)
        sw = page.new_shape()
        sw.draw_rect(pymupdf.Rect(cx, cy - 7, cx + 22, cy + 2))
        sw.finish(color=rgb, fill=rgb if fo > 0 else None, fill_opacity=fo if fo > 0 else 1.0, width=1.8,
                  dashes=_pdf_dashes(lyr.get("dashes")), oc=oc)
        sw.commit(overlay=True)
        page.insert_text(pymupdf.Point(cx + 27, cy + 0.5), lyr.get("legend", lyr["name"]), fontsize=fontsize,
                         fontname="helv", color=(0.1, 0.1, 0.1), overlay=True, oc=oc)
    if note:
        page.insert_text(pymupdf.Point(x0 + 8, y1 - 6), note, fontsize=7.0, fontname="heit",
                         color=(0.3, 0.3, 0.3), overlay=True, oc=oc)
    return dropped


def overlay_boundaries(src_pdf, out_pdf, pages_georef, layers, legend_title="BOUNDARY LAYERS",
                       legend_note="Toggle layers in the Bluebeam/Acrobat Layers panel.",
                       legend_layer="00. Markup Legend"):
    """Overlay multiple georeferenced boundary layers onto ``src_pdf`` as toggleable Bluebeam/Acrobat
    PDF layers (OCGs) and save to ``out_pdf``.

    Parameters
    ----------
    pages_georef : list[dict]
        Each entry: ``{"page": int (0-based), "en": (n, 2), "xy": (n, 2), "viewport": (x0,y0,x1,y1)|None}``
        plus optional ``"legend_rect": (x0, y0, x1, y1)`` (draws ``draw_legend`` there on its own OCG) and
        ``"xy_space"``: ``"page"`` (default: unrotated page coordinates, as ``page.get_text`` /
        ``get_drawings`` report them) or ``"visible"`` (measured on the sheet as displayed, e.g. off a
        rendered PNG of a page with /Rotate 90; converted with the page's derotation matrix). The viewport
        and legend rect use the same space.
    layers : list[dict]
        Each entry: ``{"name": str, "geom_en": shapely geom, "rgb": (r,g,b), ...}`` plus optional
        ``lw``, ``dashes``, ``fill_opacity``, ``stroke_opacity``, ``subject``, ``comment``, ``author``,
        ``legend`` (legend text), ``on`` (bool).

    Returns a summary dict with ``residuals`` per page and ``annotations`` per layer.
    """
    doc = pymupdf.open(src_pdf)
    try:
        ocgs = []
        for lyr in layers:
            ocg = doc.add_ocg(lyr["name"], on=lyr.get("on", True))
            ocgs.append(ocg)

        def to_page(page, pts, space):
            if space != "visible" or page.rotation == 0:
                return np.asarray(pts, dtype=float)
            m = page.derotation_matrix
            return np.array([tuple(pymupdf.Point(x, y) * m) for x, y in pts], dtype=float)

        def rect_to_page(page, r, space):
            if r is None:
                return None
            q = to_page(page, [(r[0], r[1]), (r[2], r[3])], space)
            return (min(q[:, 0]), min(q[:, 1]), max(q[:, 0]), max(q[:, 1]))

        page_fits = []
        residuals_out = {}
        ocg_legend = None
        for pg in pages_georef:
            page = doc[pg["page"]]
            space = pg.get("xy_space", "page")
            cx, cy, res = fit_affine(pg["en"], to_page(page, pg["xy"], space))
            page_fits.append((pg["page"], cx, cy, rect_to_page(page, pg.get("viewport"), space)))
            residuals_out[pg["page"]] = res
            if pg.get("legend_rect") is not None:
                if ocg_legend is None:
                    ocg_legend = doc.add_ocg(legend_layer, on=True)
                draw_legend(page, rect_to_page(page, pg["legend_rect"], space), layers, ocg=ocg_legend,
                            title=legend_title, note=legend_note)

        counts = {}
        for lyr, ocg in zip(layers, ocgs):
            total = 0
            for pno, cx, cy, viewport in page_fits:
                page = doc[pno]
                polys_pdf, lines_pdf = project_geom_to_pdf(lyr["geom_en"], cx, cy, clip_rect=viewport)
                total += draw_layer_on_page(
                    page, polys_pdf, lines_pdf,
                    ocg=ocg,
                    rgb=lyr.get("rgb", (0.85, 0.1, 0.1)),
                    lw=lyr.get("lw", 1.5),
                    dashes=lyr.get("dashes"),
                    fill_opacity=lyr.get("fill_opacity", 0.0),
                    subject=lyr.get("subject", lyr["name"]),
                    comment=lyr.get("comment", ""),
                    author=lyr.get("author", "Survey Markup"),
                    stroke_opacity=lyr.get("stroke_opacity", 0.92),
                )
            counts[lyr["name"]] = total

        doc.save(out_pdf, garbage=3, deflate=True)
        return {"out_pdf": str(out_pdf), "residuals": residuals_out, "annotations": counts}
    finally:
        doc.close()
