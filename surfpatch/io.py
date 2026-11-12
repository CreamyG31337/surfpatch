"""File helpers: survey CSVs, DXF boundaries and meshes, TBC source-data-only LandXML, LAS/LAZ clipping.

LandXML TIN read/write itself is rxlkit.landxml (Surface: points north, east, z).
"""
import csv
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

def read_polygons(path):
    """2D/3D polylines with >= 3 vertices from a DXF, as (k, 2) east/north arrays."""
    from ezdxf import recover
    doc, _ = recover.readfile(str(path))
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


# ---------------------------------------------------------------- TBC source-data-only exports

def read_source_only(path, bbox=None, max_edge=15.0, name=None):
    """Rebuild a TBC 'source data only' LandXML export (breaklines that are TIN edges, plus DataPoints, no faces)
    as a Surface: constrained Delaunay of the edges, or plain Delaunay if edges cross. Streams the file, so
    100+ MB exports are fine. bbox = (emin, nmin, emax, nmax) keeps only that window. Hull faces with an edge
    longer than max_edge are dropped."""
    key, V, E = {}, [], set()

    def vid(n, e, z):
        k = (round(e, 3), round(n, 3))
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
