"""File helpers: survey CSVs, DXF boundaries and meshes, TBC source-data-only LandXML, LAS/LAZ clipping, orthophotos.

LandXML TIN read/write itself is rxlkit.landxml (Surface: points north, east, z).
"""
import csv
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import triangle

from rxlkit.landxml import Surface

# Windows attributes of an OneDrive online-only (not downloaded) file
ONLINE_ONLY = 0x400000 | 0x40000 | 0x1000


def is_online_only(path):
    return bool(getattr(os.stat(path), "st_file_attributes", 0) & ONLINE_ONLY)


# ---------------------------------------------------------------- survey CSVs

_NAMES = {"p", "pt", "pnt", "point", "name", "id", "pointname", "point name", "point id"}
_NORTH = {"n", "north", "northing", "y"}
_EAST = {"e", "east", "easting", "x"}
_ELEV = {"z", "elev", "elevation", "h", "height", "el"}
_CODE = {"d", "code", "desc", "description", "feature", "feature code"}


@dataclass
class Shots:
    name: list
    points: np.ndarray   # (n, 3) north, east, z  (rxlkit order)
    code: list
    source: str = ""

    def __len__(self):
        return len(self.points)

    def select(self, mask):
        m = np.asarray(mask, bool)
        return Shots([x for x, k in zip(self.name, m) if k], self.points[m],
                     [x for x, k in zip(self.code, m) if k], self.source)


def _num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_shots(path, order="PNEZD"):
    """Points from a survey CSV. A header row is used if present (P/N/E/Z/D or Northing/Easting/... names);
    otherwise columns follow `order` (letters P N E Z D; site default PNEZD)."""
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        sample = fh.read(16384)
        fh.seek(0)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            delim = ","
        rows = [r for r in csv.reader(fh, delimiter=delim) if any(c.strip() for c in r)]
    idx = {}
    if rows and sum(_num(c) for c in rows[0]) <= 1 and len(rows) > 1 and sum(_num(c) for c in rows[1]) >= 3:
        for i, h in enumerate(c.strip().lower() for c in rows[0]):
            for key, names in (("P", _NAMES), ("N", _NORTH), ("E", _EAST), ("Z", _ELEV), ("D", _CODE)):
                if h in names and key not in idx:
                    idx[key] = i
        rows = rows[1:]
    if not {"N", "E", "Z"} <= idx.keys():
        idx = {k: i for i, k in enumerate(order.upper())}
    names, pts, codes = [], [], []
    for r in rows:
        try:
            pts.append((float(r[idx["N"]]), float(r[idx["E"]]), float(r[idx["Z"]])))
        except (ValueError, IndexError):
            continue
        names.append(r[idx["P"]].strip() if "P" in idx and idx["P"] < len(r) else "")
        codes.append(r[idx["D"]].strip() if "D" in idx and idx["D"] < len(r) else "")
    return Shots(names, np.array(pts, float).reshape(-1, 3), codes, str(path))


def read_shots_many(paths, order="PNEZD", code=None):
    """Concatenate shots from several CSVs, optionally keeping codes matching regex `code`."""
    out = Shots([], np.zeros((0, 3)), [], "")
    srcs = []
    rx = re.compile(code) if code else None
    for p in paths:
        s = read_shots(p, order)
        if rx:
            s = s.select([bool(rx.search(c)) for c in s.code])
        out = Shots(out.name + s.name, np.r_[out.points, s.points], out.code + s.code)
        srcs += [str(p)] * len(s)
    out.source = srcs
    return out


# ---------------------------------------------------------------- DXF

def _read_polygons_raw(path):
    """Plain group-code reader for (LW)POLYLINE vertices, for minimal DXFs ezdxf rejects (e.g. LWPOLYLINE
    without subclass markers, as older job scripts wrote)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh]
    polys, cur, ent, x = [], None, None, None
    for code, val in zip(lines[0::2], lines[1::2]):
        if code == "0":
            if val in ("POLYLINE", "LWPOLYLINE"):
                cur = []
                polys.append(cur)
            elif val != "VERTEX":
                cur = None
            ent, x = val, None
        elif cur is not None and ent in ("VERTEX", "LWPOLYLINE"):
            if code == "10":
                x = float(val)
            elif code == "20" and x is not None:
                cur.append((x, float(val)))
                x = None
    return [np.array(p, float) for p in polys if len(p) >= 3]


def read_polygons(path):
    """2D/3D polylines with >= 3 vertices from a DXF, as (k, 2) east/north arrays."""
    from ezdxf import recover
    from ezdxf.lldxf.const import DXFError
    try:
        doc, _ = recover.readfile(str(path))
    except DXFError:
        return _read_polygons_raw(path)
    out = []
    for ent in doc.modelspace():
        t = ent.dxftype()
        if t == "LWPOLYLINE":
            pts = np.array([(p[0], p[1]) for p in ent.get_points("xy")], float)
        elif t == "POLYLINE" and not (ent.is_poly_face_mesh or ent.is_polygon_mesh):
            pts = np.array([(v.dxf.location.x, v.dxf.location.y) for v in ent.vertices], float)
        else:
            continue
        if len(pts) >= 3:
            out.append(pts)
    return out


def polygon_area(pg):
    x, y = np.asarray(pg, float).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def read_boundary(path):
    """The largest-area polyline in a DXF (east/north)."""
    polys = read_polygons(path)
    if not polys:
        raise ValueError(f"{path}: no polylines with 3+ vertices")
    return max(polys, key=polygon_area)


def _read_polylines_3d_raw(path):
    """Plain group-code reader for 2D/3D (LW)POLYLINEs with layer and Z, returning [(layer, (k, 3) n/e/z)]."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh]
    out, cur, layer, ent, v, lw_z = [], None, "0", None, None, 0.0
    for code, val in zip(lines[0::2], lines[1::2]):
        if code == "0":
            if v is not None and cur is not None:
                cur.append([v[1], v[0], v[2]])
            v = None
            if val in ("POLYLINE", "LWPOLYLINE"):
                if cur is not None and len(cur) >= 2:
                    out.append((layer, np.array(cur, float)))
                cur, layer, lw_z = [], "0", 0.0
            elif val == "VERTEX" and cur is not None:
                v = [0.0, 0.0, 0.0]
            elif val == "SEQEND":
                if cur is not None and len(cur) >= 2:
                    out.append((layer, np.array(cur, float)))
                cur = None
            elif cur is not None and ent == "LWPOLYLINE":
                if len(cur) >= 2:
                    out.append((layer, np.array(cur, float)))
                cur = None
            ent = val
        elif cur is not None and ent == "POLYLINE" and code == "8":
            layer = val
        elif cur is not None and ent == "LWPOLYLINE":
            if code == "8":
                layer = val
            elif code == "38":
                lw_z = float(val)
            elif code == "10":
                v = [float(val), 0.0, lw_z]
            elif code == "20" and v is not None:
                v[1] = float(val)
                cur.append([v[1], v[0], v[2]])
                v = None
        elif v is not None and ent == "VERTEX" and code in ("10", "20", "30"):
            v[("10", "20", "30").index(code)] = float(val)
    if v is not None and cur is not None:
        cur.append([v[1], v[0], v[2]])
    if cur is not None and len(cur) >= 2:
        out.append((layer, np.array(cur, float)))
    return out


def read_polylines_3d(path):
    """2D/3D polylines with >= 2 vertices from a DXF, as [(layer, (k, 3) array of (north, east, z))]."""
    from ezdxf import recover
    from ezdxf.lldxf.const import DXFError
    try:
        doc, _ = recover.readfile(str(path))
    except DXFError:
        return _read_polylines_3d_raw(path)
    out = []
    for ent in doc.modelspace():
        t = ent.dxftype()
        layer = ent.dxf.get("layer", "0")
        if t == "LWPOLYLINE":
            z = float(ent.dxf.get("elevation", 0.0))
            pts = np.array([(p[1], p[0], z) for p in ent.get_points("xy")], float)
        elif t == "POLYLINE" and not (ent.is_poly_face_mesh or ent.is_polygon_mesh):
            pts = np.array([(v.dxf.location.y, v.dxf.location.x, v.dxf.location.z) for v in ent.vertices], float)
        else:
            continue
        if len(pts) >= 2:
            out.append((layer, pts))
    return out


def _layer(name):
    return re.sub(r"[^A-Za-z0-9_$-]+", "_", name).strip("_")[:31] or "MESH"


def write_dxf_mesh(path, surfaces):
    """R12 DXF with one polyface mesh per surface (layer = surface name). X = east, Y = north."""
    out = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "9", "$INSUNITS", "70", "6",
           "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for s in surfaces:
        nv, nf = len(s.points), len(s.faces)
        if nv > 32767 or nf > 32767:
            raise ValueError(f"{s.name}: {nv} vertices / {nf} faces exceeds the R12 polyface limit (32767)")
        lay = _layer(s.name)
        out += ["0", "POLYLINE", "8", lay, "66", "1", "10", "0.0", "20", "0.0", "30", "0.0",
                "70", "64", "71", str(nv), "72", str(nf)]
        for n, e, z in s.points:
            out += ["0", "VERTEX", "8", lay, "10", f"{e:.4f}", "20", f"{n:.4f}", "30", f"{z:.4f}", "70", "192"]
        for a, b, c in s.faces:
            out += ["0", "VERTEX", "8", lay, "10", "0.0", "20", "0.0", "30", "0.0", "70", "128",
                    "71", str(a + 1), "72", str(b + 1), "73", str(c + 1)]
        out += ["0", "SEQEND", "8", lay]
    out += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w", encoding="ascii", newline="\r\n") as fh:
        fh.write("\n".join(out) + "\n")


def write_dxf_polylines(path, loops, layer="BOUNDARY", closed=True):
    """R12 DXF of 2D polylines (POLYLINE/VERTEX/SEQEND - every reader takes these), e.g. patch outlines.
    loops: lists of (east, north)."""
    lay = _layer(layer)
    out = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "0", "ENDSEC",
           "0", "SECTION", "2", "ENTITIES"]
    for lp in loops:
        out += ["0", "POLYLINE", "8", lay, "66", "1", "10", "0.0", "20", "0.0", "30", "0.0",
                "70", "1" if closed else "0"]
        for x, y in lp:
            out += ["0", "VERTEX", "8", lay, "10", f"{x:.3f}", "20", f"{y:.3f}", "30", "0.0"]
        out += ["0", "SEQEND", "8", lay]
    out += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w", encoding="ascii", newline="\r\n") as fh:
        fh.write("\n".join(out) + "\n")


def write_pnezd(path, points, code="", start=1):
    """PNEZD CSV (no header) of points given as (north, east, z)."""
    with open(path, "w", encoding="ascii", newline="\r\n") as fh:
        for k, (n, e, z) in enumerate(points, start):
            fh.write(f"{k},{n:.4f},{e:.4f},{z:.4f},{code}\n")


# ---------------------------------------------------------------- TBC source-data-only exports

def read_source_only(path, bbox=None, max_edge=15.0, name=None):
    """Rebuild a TBC 'source data only' LandXML export (breaklines that are TIN edges, plus DataPoints, no faces)
    as a Surface: constrained Delaunay of the edges, or plain Delaunay if edges cross. Streams the file, so
    100+ MB exports are fine. bbox = (emin, nmin, emax, nmax) keeps only that window. Hull faces with an edge
    longer than max_edge are dropped."""
    key, V, E = {}, [], set()

    def vid(n, e, z):
        # snap to 0.1 mm, the precision LandXML is written at, so the TIN checked here is the one written
        n, e, z = round(n, 4), round(e, 4), round(z, 4)
        k = (e, n)
        if k not in key:
            key[k] = len(V)
            V.append((n, e, z))
        return key[k]

    def keep(a):
        if bbox is None:
            return np.ones(len(a), bool)
        return (a[:, 1] >= bbox[0]) & (a[:, 0] >= bbox[1]) & (a[:, 1] <= bbox[2]) & (a[:, 0] <= bbox[3])

    in_dp, sname = 0, None
    for ev, el in ET.iterparse(str(path), events=("start", "end")):
        tag = el.tag.rsplit("}", 1)[-1]
        if ev == "start":
            if tag == "DataPoints":
                in_dp += 1
            elif tag == "Surface" and sname is None:
                sname = el.get("name")
            continue
        if tag == "DataPoints":
            in_dp -= 1
        elif tag == "PntList3D" and el.text:
            a = np.array(el.text.split(), float).reshape(-1, 3)   # N E Z
            k = keep(a)
            if in_dp:
                for p in a[k]:
                    vid(*p)
            elif k.all():
                ids = [vid(*p) for p in a]
                E.update(tuple(sorted(x)) for x in zip(ids, ids[1:]) if x[0] != x[1])
            el.clear()
        elif tag in ("Breakline", "Surface"):
            el.clear()
    if len(V) < 3:
        raise ValueError(f"{path}: fewer than 3 points{' in bbox' if bbox else ''}")
    V = np.array(V, float)
    xy = V[:, [1, 0]] - V[:, [1, 0]].mean(0)
    args = {"vertices": xy}
    if E:
        args["segments"] = np.array(sorted(E))
    r = triangle.triangulate(args, "pcQ")
    if len(r["vertices"]) != len(V):     # crossing edges (overlapping layers): plain Delaunay
        r = triangle.triangulate({"vertices": xy}, "cQ")
    T = np.asarray(r["triangles"], int)
    Pt = xy[T]
    L = np.max(np.linalg.norm(Pt - np.roll(Pt, 1, axis=1), axis=2), axis=1)
    T = T[L < max_edge]
    return Surface(name or sname or os.path.splitext(os.path.basename(path))[0], V, T,
                   "rebuilt from source-data-only export")


# ---------------------------------------------------------------- point clouds

def clip_las(path, bbox, chunk=5_000_000, allow_download=False):
    """(x, y, z, classification) arrays of the points of a LAS/LAZ inside bbox (emin, nmin, emax, nmax),
    streamed in chunks. Returns None if the file is online-only (unless allow_download) or misses bbox.
    Needs laspy[lazrs]."""
    import laspy
    if not allow_download and is_online_only(path):
        return None
    X, Y, Z, C = [], [], [], []
    with laspy.open(path) as f:
        h = f.header
        if h.maxs[0] < bbox[0] or h.mins[0] > bbox[2] or h.maxs[1] < bbox[1] or h.mins[1] > bbox[3]:
            return None
        for ch in f.chunk_iterator(chunk):
            x, y = np.asarray(ch.x), np.asarray(ch.y)
            m = (x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3])
            if m.any():
                X.append(x[m]); Y.append(y[m]); Z.append(np.asarray(ch.z)[m])
                C.append(np.asarray(ch.classification)[m])
    if not X:
        return None
    return tuple(np.concatenate(a) for a in (X, Y, Z, C))


# ---------------------------------------------------------------- orthophotos

def _world_file(image_path):
    stem = os.path.splitext(str(image_path))[0]
    for ext in (".jgw", ".jgwx", ".wld", ".tfw", ".pgw"):
        p = stem + ext
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"no world file (.jgw/.wld/.tfw/.pgw) beside {image_path}")


def ortho_crop(image, bbox_en, cache=None, downsample=2):
    """Crop a georeferenced orthophoto (with a world file beside it) to `bbox_en = (emin, nmin, emax, nmax)`.

    Uses PIL draft mode so gigapixel JPEGs are not decoded at full resolution. When `cache` is given, saves the
    cropped image and `<cache>.json` extent and reuses them on later calls.
    Returns `(PIL.Image, (e0, e1, n0, n1))` where the tuple is matplotlib `imshow` extent order.
    """
    from PIL import Image
    cache_str = str(cache) if cache else None
    if cache_str and os.path.isfile(cache_str) and os.path.isfile(cache_str + ".json"):
        with open(cache_str + ".json", encoding="utf-8") as fh:
            ext = tuple(json.load(fh))
        return Image.open(cache_str), ext
    wf = _world_file(image)
    with open(wf, encoding="utf-8") as fh:
        A, D, B, E, X0, Y0 = [float(x) for x in fh.read().split()[:6]]
    if abs(D) > 1e-9 or abs(B) > 1e-9:
        raise ValueError(f"{wf}: rotated world files are not supported (D={D}, B={B})")
    e0, n0, e1, n1 = [float(v) for v in bbox_en]
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image) as im:
        W, H = im.size
        if downsample and downsample > 1:
            im.draft("RGB", (max(1, W // downsample), max(1, H // downsample)))
        k = W / im.size[0]
        px, py = A * k, E * k
        # World file (X0, Y0) is the centre of original pixel (0, 0); its outer corner is half an original pixel out.
        e_edge, n_edge = X0 - A / 2.0, Y0 - E / 2.0
        c0 = max(0, int(math.floor((e0 - e_edge) / px)))
        c1 = min(im.size[0], int(math.ceil((e1 - e_edge) / px)))
        r0 = max(0, int(math.floor((n1 - n_edge) / py)))
        r1 = min(im.size[1], int(math.ceil((n0 - n_edge) / py)))
        if c1 <= c0 or r1 <= r0:
            raise ValueError(f"bbox {bbox_en} does not overlap orthophoto {image}")
        crop = im.crop((c0, r0, c1, r1)).convert("RGB")
    ext = (e_edge + c0 * px, e_edge + c1 * px, n_edge + r1 * py, n_edge + r0 * py)
    if cache_str:
        parent = os.path.dirname(cache_str)
        if parent:
            os.makedirs(parent, exist_ok=True)
        crop.save(cache_str, quality=90)
        with open(cache_str + ".json", "w", encoding="utf-8") as fh:
            json.dump(list(ext), fh)
    return crop, ext
