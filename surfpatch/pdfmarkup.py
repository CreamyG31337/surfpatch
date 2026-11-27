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
    and ``per_point_m`` ground residuals obtained by inverting the fitted transform.
    """
    en = np.asarray(en, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if en.ndim != 2 or en.shape[1] != 2 or xy.shape != en.shape or len(en) < 3:
        raise ValueError("need at least 3 matching (E, N) and (pdf_x, pdf_y) pairs")
    A = np.c_[en[:, 0], en[:, 1], np.ones(len(en))]
    cx, _, _, _ = np.linalg.lstsq(A, xy[:, 0], rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, xy[:, 1], rcond=None)

    # Invert M = [[cx0, cx1], [cy0, cy1]] to express fit error in ground metres
    M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]], dtype=float)
    t = np.array([cx[2], cy[2]], dtype=float)
    en_back = np.linalg.solve(M, (xy - t).T).T
    errs = np.hypot(en_back[:, 0] - en[:, 0], en_back[:, 1] - en[:, 1])
    residuals = {
        "max_m": float(np.max(errs)),
        "rms_m": float(np.sqrt(np.mean(errs * errs))),
        "per_point_m": [float(v) for v in errs],
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
    tris = []
    for a, b, c in faces:
        p0, p1, p2 = pts_en[a], pts_en[b], pts_en[c]
        if max_edge is not None:
            e0 = float(np.hypot(*(p1 - p0)))
            e1 = float(np.hypot(*(p2 - p1)))
            e2 = float(np.hypot(*(p0 - p2)))
            if max(e0, e1, e2) > max_edge:
                continue
        poly = Polygon([p0, p1, p2])
        if poly.is_valid and poly.area > 1e-4:
            tris.append(poly)
    if not tris:
        return MultiPolygon()
    geom = unary_union(tris).buffer(0.01).buffer(-0.01)
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
    tris = []
    for f in tri.simplices:
        pts = en[f]
        e0 = float(np.hypot(*(pts[1] - pts[0])))
        e1 = float(np.hypot(*(pts[2] - pts[1])))
        e2 = float(np.hypot(*(pts[0] - pts[2])))
        if max(e0, e1, e2) <= max_edge:
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 0.05:
                tris.append(poly)
    if not tris:
        return MultiPolygon()
    geom = unary_union(tris).buffer(0.01).buffer(-0.01)
    if simplify and simplify > 0:
        geom = geom.simplify(simplify, preserve_topology=True)
    return geom


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


def overlay_boundaries(src_pdf, out_pdf, pages_georef, layers):
    """Overlay multiple georeferenced boundary layers onto ``src_pdf`` as toggleable Bluebeam/Acrobat
    PDF layers (OCGs) and save to ``out_pdf``.

    Parameters
    ----------
    pages_georef : list[dict]
        Each entry: ``{"page": int (0-based), "en": (n, 2), "xy": (n, 2), "viewport": (x0,y0,x1,y1)|None}``.
    layers : list[dict]
        Each entry: ``{"name": str, "geom_en": shapely geom, "rgb": (r,g,b), ...}`` plus optional
        ``lw``, ``dashes``, ``fill_opacity``, ``subject``, ``comment``, ``author``, ``on`` (bool).

    Returns a summary dict with ``residuals`` per page and ``annotations`` per layer.
    """
    doc = pymupdf.open(src_pdf)
    try:
        ocgs = []
        for lyr in layers:
            ocg = doc.add_ocg(lyr["name"], on=lyr.get("on", True))
            ocgs.append(ocg)

        page_fits = []
        residuals_out = {}
        for pg in pages_georef:
            cx, cy, res = fit_affine(pg["en"], pg["xy"])
            page_fits.append((pg["page"], cx, cy, pg.get("viewport")))
            residuals_out[pg["page"]] = res

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
                )
            counts[lyr["name"]] = total

        doc.save(out_pdf, garbage=3, deflate=True)
        return {"out_pdf": str(out_pdf), "residuals": residuals_out, "annotations": counts}
    finally:
        doc.close()
