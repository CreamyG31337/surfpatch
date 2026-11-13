"""Cut/fill (C-F) map sheets of scanned channel walls against a design corridor (RoadXML .rxl).

Each scan point's signed distance to the design wall is measured square to the wall (CloudCompare cloud-to-mesh,
or the design cross-section in Python), so curved walls need no projection planes. Each wall is unrolled to
station / elevation, triangulated, coloured with TBC-style cut/fill bands inside the scanned outline, and placed
on a title-blocked sheet with sheetwrap.

Signs: distances.csv keeps + = rock inside the design line (still to cut), - = overbreak. The map uses TBC's
convention: C = cut, F = fill (overbreak); in a TBC .tcf colour file cuts are the negative values.

Config (JSON), all keys optional except "alignment":
  alignment      .rxl corridor; walls are built from it (rxlkit.wall_surface) unless "walls" gives LandXML files
  walls          {"L": xml, "R": xml}
  wall_names     {"L": "WEST", "R": "EAST"}
  tcf            TBC colour mapping file; default DEFAULT_BANDS
  grid           {"major_sta": 10, "minor_sta": 5, "major_el": [..] or step, "minor_el": [..] or step}
  max_edge       m; scan triangles with a longer edge are gaps (default 3)
  toe_margin     m; points below design toe + this are floor and dropped (default 0.3)
  max_dist       m; points farther than this from the wall are dropped (default 3)
  thin_cell      m; average points into cells this size in the unrolled view; "auto" = 0.1 m above 20k points
  label_cell     m; at most one value label per cell; "auto" = every point when they fit, else spaced so
                 labels don't overlap at the drawing scale
  sheet_length   m of wall per sheet (default 40); longer walls get several sheets, on round stations
  engine         "cloudcompare", "section" or "auto"
  sheet          sheetwrap job values (project, datum, drawn_by, logo, extends, sheet_size, ...); "title" may use
                 {wall}; "notes" are added after the standard notes, {scan} and {design} are filled in
"""
import csv
import datetime as dt
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# TBC-like stepped bands when no .tcf is given: (inner breaks, colours) in TBC sign (negative = cut)
DEFAULT_BANDS = ([-0.3, -0.2, -0.1, -0.03, 0.1, 0.2, 0.3],
                 [(1, 0, 0), (1, 165 / 255, 0), (253 / 255, 209 / 255, 2 / 255), (1, 1, 0), (1, 1, 1),
                  (0, 172 / 255, 0), (28 / 255, 141 / 255, 1), (0, 0, 1)])
ALPHA, MESH_GREY = 0.6, 0.62
GRID_MAJOR = dict(color="0.35", lw=0.7, label="#1f3b5a", size=10)
GRID_MINOR = dict(color="0.7", lw=0.4, label="#5f7891", size=8)
OUTLINE_LW, OUTLINE_C = 0.6, "#b0189a"
DEFAULTS = {"wall_names": {"L": "LEFT", "R": "RIGHT"}, "tcf": None,
            "grid": {"major_sta": 10, "minor_sta": 5, "major_el": 10, "minor_el": 5},
            "max_edge": 3.0, "toe_margin": 0.3, "max_dist": 3.0, "thin_cell": "auto", "label_cell": "auto",
            "sheet_length": 40.0,
            "engine": "auto", "sheet": {}}


# ---------------------------------------------------------------- colours, labels

def read_tcf(path):
    """TBC colour mapping (.tcf XML) -> (inner breaks, colours), len(colours) == len(breaks) + 1.
    Stepped: colour i fills [value i, value i+1). The first and last values are only the min/max of the map the
    file was saved from, so they are not breaks: the first colour fills everything below the second value, and
    the last entry's colour (which only marks the top) is not used. Duplicate values are dropped."""
    vals = [(float(v.findtext("elevation")),
             tuple(int(v.findtext(c)) / 255 for c in ("Red", "Green", "Blue")))
            for v in ET.parse(path).getroot().iter("value")]
    vals = sorted(vals, key=lambda v: v[0])
    vals = [v for i, v in enumerate(vals) if i == len(vals) - 1 or v[0] < vals[i + 1][0]]
    return [v[0] for v in vals[1:-1]], [v[1] for v in vals[:-1]]


def fmt_sta(s, dec=2):
    a = abs(s)
    return f"{'-' if s < 0 else ''}{int(a // 1000)}+{a % 1000:0{4 + dec}.{dec}f}"


def cf_label(v):
    """TBC-style label from a TBC-signed value (negative = cut)."""
    return f"{abs(v):.3f}{'C' if v < 0 else 'F'}"


def as_drawn(c):
    """Band colour as it looks on the map (translucent over the grey mesh), as hex."""
    return "#" + "".join(f"{round((ALPHA * x + (1 - ALPHA) * MESH_GREY) * 255):02x}" for x in c)


def _levels(spec, lo, hi):
    if isinstance(spec, (int, float)):
        return list(np.arange(np.floor(lo / spec) * spec, hi + 1e-9, spec)) if spec > 0 else []
    return list(spec or [])


# ---------------------------------------------------------------- design walls

class Design:
    """Corridor + wall meshes, with normals wound toward the centreline (into the channel)."""

    def __init__(self, cfg):
        from rxlkit.landxml import read_surface
        from rxlkit.roadxml import Corridor
        from rxlkit.surfaces import wall_surface
        self.rxl = Path(cfg["alignment"])
        self.cor = Corridor.load(self.rxl)
        self.al = self.cor.alignment
        self.walls = {}
        for side in ("L", "R"):
            src = (cfg.get("walls") or {}).get(side)
            try:
                s = read_surface(src) if src else wall_surface(self.cor, side)
            except ValueError:
                continue                          # no wall on this side
            self.walls[side] = s
        if not self.walls:
            raise ValueError(f"{self.rxl.name}: no walls")

    def mesh(self):
        """(V east/north/z, F) of all walls, every face wound counter-clockwise in plan (normal up).
        Design walls are 2.5D TINs that never overhang, so 'up' is the open (channel) side everywhere, including
        corner cross faces, which run square to the CL and so can't be oriented 'toward the CL'."""
        Vs, Fs, n = [], [], 0
        for s in self.walls.values():
            V = s.points[:, [1, 0, 2]]
            F = np.asarray(s.faces, int).copy()
            A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            up = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
            F[up < 0] = F[up < 0][:, [0, 2, 1]]
            Vs.append(V); Fs.append(F + n); n += len(V)
        return np.concatenate(Vs), np.concatenate(Fs)

    def near_step(self, s, dist):
        """Points within `dist` m of a design step/corner (where a single cross-section is not the nearest wall)."""
        st = [b for a, b in self.cor.steps()]
        if not st:
            return np.zeros(len(s), bool)
        return np.min(np.abs(np.asarray(s)[:, None] - np.array(st)[None, :]), axis=1) < dist

    def bbox(self, pad=5.0):
        P = np.concatenate([s.points for s in self.walls.values()])
        return (P[:, 1].min() - pad, P[:, 0].min() - pad, P[:, 1].max() + pad, P[:, 0].max() + pad)

    def wall_line(self, sta, side):
        """(toe offset, toe z, top offset, top z) of the design wall at sta."""
        pts = self.cor.side(sta, side)
        return pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]

    def section_distance(self, s, o, z):
        """Signed distance in the cross-section plane to the wall line (+ = toward the centreline)."""
        d = np.empty(len(s))
        for i in range(len(s)):
            o0, z0, o1, z1 = self.wall_line(s[i], "L" if o[i] < 0 else "R")
            t = np.array([o1 - o0, z1 - z0]); t /= np.linalg.norm(t)
            nrm = np.array([t[1], -t[0]])
            if np.sign(nrm[0]) == np.sign(o0):
                nrm = -nrm
            d[i] = np.dot([o[i] - o0, z[i] - z0], nrm)
        return d


# ---------------------------------------------------------------- scan input

def read_scan(paths, design, order="PNEZD", download=False):
    """(names, points east/north/z) from CSVs (any order io.read_shots understands; E/N columns swapped
    automatically when the 'northing' looks like an easting and vice versa) and LAS/LAZ (clipped to the walls)."""
    from . import io
    names, pts = [], []
    for p in paths:
        p = str(p)
        if p.lower().endswith((".las", ".laz")):
            c = io.clip_las(p, design.bbox(), allow_download=download)
            if c is None:
                print(f"  skipped {Path(p).name} (online-only, or not over the walls)")
                continue
            names += [""] * len(c[0])
            pts.append(np.stack(c[:3], 1))
        else:
            sh = io.read_shots(p, order)
            ne = sh.points
            if len(ne) and np.median(ne[:, 1]) > 1e6 > np.median(ne[:, 0]):
                ne = ne[:, [1, 0, 2]]
                print(f"  {Path(p).name}: columns look like E,N - swapped")
            names += sh.name
            pts.append(ne[:, [1, 0, 2]])
    return names, (np.concatenate(pts) if pts else np.zeros((0, 3)))


def thin(s, z, d, cell):
    """Average (station, elevation, distance) into square cells of the unrolled view."""
    k = np.stack([np.floor(s / cell), np.floor(z / cell)], 1).astype(np.int64)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    avg = lambda v: np.bincount(inv, v) / cnt
    return avg(s), avg(z), avg(d)


# ---------------------------------------------------------------- map drawing

def scan_mesh(s, Z, max_edge):
    from matplotlib.tri import Triangulation
    tri = Triangulation(s, Z)
    T = tri.triangles
    e = np.stack([np.hypot(s[T[:, a]] - s[T[:, b]], Z[T[:, a]] - Z[T[:, b]]) for a, b in ((0, 1), (1, 2), (2, 0))], 1)
    tri.set_mask(e.max(1) > max_edge)
    return tri


def outline_segments(tri):
    T = tri.triangles[~tri.mask] if tri.mask is not None else tri.triangles
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    return u[c == 1]


def band_areas(tri, cf, levels, g=0.05, window=None):
    """Unrolled area (m2) of the scanned mesh in each band, from a g-m raster (only stations inside window)."""
    from matplotlib.tri import LinearTriInterpolator
    x0, x1 = tri.x.min(), tri.x.max()
    if window:
        x0, x1 = max(x0, window[0]), min(x1, window[1])
    X, Y = np.meshgrid(np.arange(x0, x1 + g, g), np.arange(tri.y.min(), tri.y.max() + g, g))
    v = LinearTriInterpolator(tri, cf)(X, Y).compressed()
    return np.bincount(np.digitize(v, levels[1:-1]), minlength=len(levels) - 1) * g * g


def label_mask(s, z, cell):
    """One label per cell (the point nearest the cell centre); all if cell is falsy. cell = size or (ds, dz)."""
    if not cell:
        return np.ones(len(s), bool)
    cs, cz = cell if isinstance(cell, (tuple, list)) else (cell, cell)
    k = np.stack([np.floor(s / cs), np.floor(z / cz)], 1)
    dc = np.hypot((s - (k[:, 0] + 0.5) * cs) / cs, (z - (k[:, 1] + 0.5) * cz) / cz)
    order = np.lexsort((dc, k[:, 1], k[:, 0]))
    first = np.ones(len(order), bool)
    first[1:] = np.any(k[order][1:] != k[order][:-1], axis=1)
    m = np.zeros(len(s), bool)
    m[order[first]] = True
    return m


AX_W, AX_H = 14 * 72 * 0.86, 8.5 * 72 * 0.82          # map axes size (pt) on the view page


def map_page(pdf, design, s, Z, d, side, cfg, bands, window=None):
    """One unrolled wall view (VE 1.0) to `pdf`. `window` = (from, to) station: the view shows that range
    (points slightly beyond it keep the mesh continuous at the edges). Returns the legend bar and stats."""
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    cf = -d                                           # TBC sign: negative = cut
    inner, colours = bands
    levels = [min(cf.min(), inner[0]) - 1.0] + list(inner) + [max(cf.max(), inner[-1]) + 1.0]
    g = {**DEFAULTS["grid"], **cfg.get("grid", {})}
    pad = 2.0
    wa, wb = window if window else (s.min(), s.max())
    vis = (s >= wa) & (s <= wb)
    s0, s1 = max(s.min(), wa) - pad, min(s.max(), wb) + pad
    Zv = Z[vis]
    major_el = _levels(g["major_el"], Z.min() - 20, Z.max() + 20)
    minor_el = [z for z in _levels(g["minor_el"], Z.min() - 20, Z.max() + 20) if min(
        [abs(z - m) for m in major_el] or [1]) > 1e-6]
    zl = [z for z in major_el if z <= Zv.min()]
    zh = [z for z in major_el if z >= Zv.max()]
    z0 = min(Zv.min() - pad, (max(zl) - 0.8) if zl else Zv.min() - pad)
    z1 = max(Zv.max() + pad, (min(zh) + 0.8) if zh else Zv.max() + pad)
    pt_per_m = min(AX_W / (s1 - s0), AX_H / (z1 - z0))

    fig = plt.figure(figsize=(14, 8.5))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.03, 0.12, 0.86, 0.82])       # room on the right for the elevation labels
    ax.patch.set_alpha(0)
    tri = scan_mesh(s, Z, cfg["max_edge"])
    ax.tricontourf(tri, np.zeros_like(cf), levels=[-1, 1], colors=[str(MESH_GREY)], zorder=1)
    ax.tricontourf(tri, cf, levels=levels, colors=colours, alpha=ALPHA, zorder=1)
    for a, b in outline_segments(tri):
        ax.plot([s[a], s[b]], [Z[a], Z[b]], color=OUTLINE_C, lw=OUTLINE_LW, zorder=3, solid_capstyle="round")
    lab_cell = cfg["label_cell"]
    if lab_cell == "auto":                             # a label is ~24 x 9 pt at 6.5 pt type; keep them apart
        cell = (38 / pt_per_m, 13 / pt_per_m)
        nn = len(s) / max(1e-9, (s.max() - s.min() + 1) * (Z.max() - Z.min() + 1))
        lab_cell = 0 if nn * cell[0] * cell[1] < 0.6 and len(s) <= 400 else cell
    lm = label_mask(s, Z, lab_cell) & vis
    ax.plot(s[lm], Z[lm], "+", color="0.35", ms=5, mew=0.6, zorder=4)
    halo = [pe.withStroke(linewidth=1.6, foreground=(1, 1, 1, 0.75))]
    for x, z, v in zip(s[lm], Z[lm], cf[lm]):
        ax.annotate(cf_label(v), (x, z), xytext=(2, 2), textcoords="offset points", fontsize=6.5, color="0.1",
                    zorder=5, path_effects=halo)

    steps = [round(b, 2) for a, b in design.cor.steps() if s0 <= b <= s1]
    lines = {b: True for b in steps}
    for x in np.arange(np.ceil(s0 / g["minor_sta"]) * g["minor_sta"], s1 + 1e-6, g["minor_sta"]):
        lines.setdefault(round(x, 2), abs(x / g["major_sta"] - round(x / g["major_sta"])) < 1e-6)
    label_gap = 55 / pt_per_m                          # m: a station label is ~50 pt wide
    for x, major in sorted(lines.items()):
        st = GRID_MAJOR if major else GRID_MINOR
        ax.axvline(x, color=st["color"], lw=st["lw"], zorder=2)
        if x not in steps and any(abs(x - b) < label_gap for b in steps):
            continue                                   # the step's own label wins
        ax.text(x, z0 - 0.015 * (z1 - z0), fmt_sta(x), color=st["label"], fontsize=st["size"], ha="center",
                va="top", clip_on=False)
    for z in sorted(major_el + minor_el):
        if z0 <= z <= z1:
            st = GRID_MAJOR if z in major_el else GRID_MINOR
            ax.axhline(z, color=st["color"], lw=st["lw"], zorder=2)
            ax.text(1.01, z, f"EL. {z:.3f}", color=st["label"], fontsize=st["size"] + 1, ha="left", va="center",
                    transform=ax.get_yaxis_transform(), clip_on=False)
    ss = np.linspace(s0, s1, 300)
    lo, hi = design.cor.extent
    toe = np.array([design.wall_line(x, side)[1] if lo <= x <= hi else np.nan for x in ss])
    toe_shown = bool(((toe > z0) & (toe < z1)).any())
    if toe_shown:
        ax.plot(ss, toe, color="saddlebrown", lw=1.2, ls="--", zorder=2)
    ax.set_xlim(s0, s1); ax.set_ylim(z0, z1)
    if side == "R":           # right wall seen from the channel: station increases to the left
        ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    pdf.savefig(fig, transparent=True)
    plt.close(fig)

    areas = band_areas(tri, cf, levels, window=(wa, wb))
    cfv = cf[vis]
    used = [i for i in range(len(colours)) if areas[i] > 0.005 or ((cfv >= levels[i]) & (cfv < levels[i + 1])).any()]
    keep = range(min(used), max(used) + 1)
    bar = {"kind": "colorbar", "top": "FILLS", "bottom": "CUTS",
           "bands": [{"color": as_drawn(colours[i]), "text": f"{areas[i]:.1f} m2"} for i in reversed(keep)],
           "breaks": [f"{abs(levels[i]):.3f} {'C' if levels[i] < 0 else 'F'}" for i in reversed(keep) if i > keep[0]]}
    return {"bar": bar, "area": float(areas.sum()), "toe_shown": toe_shown, "labels_all": not lab_cell,
            "bands": {f"{levels[i]:+.3f}..{levels[i + 1]:+.3f}": float(areas[i]) for i in keep}}


# ---------------------------------------------------------------- sheet

def sheet_job(cfg, design, scan_names, pages, map_pdf, out_pdf, stem=""):
    """sheetwrap job dict. String values in cfg["sheet"] may use {stem} (output name) and {prefix} (its first
    word, e.g. the date code of '261112-W-Wall-Scan')."""
    import re
    fill = {"stem": stem, "prefix": re.split(r"[-_ ]", stem)[0] if stem else ""}
    sc = {"drawing_no": "{prefix}-CF"}
    sc.update(cfg.get("sheet", {}))
    sc = {k: v.replace("{stem}", fill["stem"]).replace("{prefix}", fill["prefix"]) if isinstance(v, str) else v
          for k, v in sc.items()}
    title = sc.pop("title", "{wall} WALL - C-F MAP VS DESIGN")
    extra_notes = sc.pop("notes", [])
    scan = ", ".join(scan_names)
    job = {"extends": ["company"], "scale": f"AS SHOWN - {cfg['grid'].get('minor_sta', 5)} m STATION GRID, VE 1.0",
           "sheet_size": [1224, 792], "holes_label": "SCAN", "page_order": None, "checked_by": "",
           "revision": "0", "date": dt.date.today().isoformat()}
    job.update(sc)
    job.update({
        "input": str(map_pdf), "output": str(out_pdf),
        "sign_convention": [
            "C (CUT) = SCANNED ROCK IS INSIDE THE DESIGN LINE (STILL TO EXCAVATE). F (FILL) = SCANNED ROCK IS "
            "BEYOND THE DESIGN LINE (OVERBREAK).",
            "VALUES ARE DISTANCES (m) MEASURED SQUARE TO THE DESIGN WALL SURFACE, NOT VERTICAL OR HORIZONTAL.",
            "VIEW IS UNROLLED ALONG THE CL, LOOKING AT THE WALL FROM INSIDE THE CHANNEL.",
        ],
        "notes": [f"DESIGN: {design.rxl.name}.", f"SCAN: {scan}.",
                  "DISTANCES COMPUTED WITH " + ("CLOUDCOMPARE (CLOUD TO MESH) AND CHECKED AGAINST THE DESIGN SECTIONS."
                                                if pages and pages[0]["engine"] == "cloudcompare"
                                                else "THE DESIGN CROSS-SECTIONS."),
                  f"COLOURS ARE SHOWN ONLY INSIDE THE SCANNED OUTLINE (TRIANGLES WITH EDGES OVER {cfg['max_edge']:g} m "
                  "ARE GAPS AND ARE LEFT BLANK). AREAS ARE UNROLLED AREAS ON THE WALL."]
                 + [n.format(scan=scan, design=design.rxl.name) for n in extra_notes],
        "pages": [],
    })
    for i, p in enumerate(pages, 1):
        info, lr = p["info"], ("LT" if p["side"] == "L" else "RT")
        rng = f"{fmt_sta(p['s0'])} TO {fmt_sta(p['s1'])}"
        legend = [info["bar"],
                  {"kind": "line", "color": OUTLINE_C, "width": 1.0, "label": "OUTLINE OF THE SCANNED AREA"},
                  {"kind": "line", "color": "#595959", "width": 0.7,
                   "label": "GRID: STATIONS AND ELEVATIONS (LIGHTER = MINOR)"},
                  {"kind": "text", "sample": "+ 0.123C", "label": "SCAN POINT AND ITS VALUE (m)"
                   + ("" if p["labels_all"] else ", ONE PER CELL")}]
        if info["toe_shown"]:
            legend.append({"kind": "line", "color": "#8b4513", "width": 1.2, "label": "DESIGN TOE (DASHED)"})
        job["pages"].append({
            "source_page": i,
            "title": title.format(wall=p["wall"]) + f"\nSTA {rng}",
            "wall_name": f"{p['wall']} ({'LEFT' if p['side'] == 'L' else 'RIGHT'}) WALL",
            "holes": f"{p['n']} POINTS, {info['area']:.1f} m2",
            "station": rng,
            "offset": f"{lr} {p['o0']:.1f} TO {p['o1']:.1f} m",
            "elevation": f"SCAN EL. {p['z0']:.1f} TO {p['z1']:.1f}",
            "views": [{"title": f"C-F MAP - {p['wall']} WALL",
                       "subtitle": "UNROLLED ALONG CL, LOOKING AT THE WALL FROM THE CHANNEL  -  VE 1.0"}],
            "legend": legend,
        })
    return job


# ---------------------------------------------------------------- driver

def windows(s, length, align):
    """Station windows of `length` m (starting on a multiple of `align`) covering s; a short last window is
    merged into the previous one."""
    if not length or s.max() - s.min() <= length:
        return [(float(s.min()), float(s.max()))]
    a0 = np.floor(s.min() / align) * align
    edges = list(np.arange(a0, s.max(), length)) + [None]
    out = []
    for a, b in zip(edges, edges[1:]):
        b = float(s.max()) if b is None else float(b)
        if ((s >= a) & (s <= b)).any():
            out.append([float(max(a, s.min())), b])
    if len(out) > 1 and out[-1][1] - out[-1][0] < length / 4:
        tail = out.pop()
        out[-1][1] = tail[1]
    return [tuple(w) for w in out]



def load_config(path_or_dict):
    if isinstance(path_or_dict, dict):
        cfg, base = dict(path_or_dict), Path.cwd()
    else:
        p = Path(path_or_dict)
        cfg, base = json.loads(p.read_text()), p.parent
    out = {**DEFAULTS, **cfg}
    out["grid"] = {**DEFAULTS["grid"], **cfg.get("grid", {})}
    rel = lambda v: str(v if Path(v).is_absolute() else base / v)
    out["alignment"] = rel(out["alignment"])
    if out.get("tcf"):
        out["tcf"] = rel(out["tcf"])
    if out.get("walls"):
        out["walls"] = {k: rel(v) for k, v in out["walls"].items()}
    return out


def run(cfg, scans, out_dir, name=None, order="PNEZD", download=False, build_sheet=True):
    """Make the C-F map for scan files `scans`. Returns a summary dict (paths, per-wall stats)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from . import cc
    cfg = load_config(cfg)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    design = Design(cfg)
    names, P = read_scan(scans, design, order, download)
    if not len(P):
        raise ValueError("no scan points")
    al = design.al
    s, o = al.station_offset(P[:, 1], P[:, 0])
    lo, hi = design.cor.extent
    side = np.where(o < 0, "L", "R")
    ok = (s > lo) & (s < hi) & np.isin(side, list(design.walls))
    toe = np.array([design.wall_line(s[i], side[i])[1] if ok[i] else np.inf for i in range(len(s))])
    ok &= P[:, 2] >= toe + cfg["toe_margin"]

    engine = cfg["engine"]
    if engine == "auto":
        engine = "cloudcompare" if cc.find_exe() else "section"
    d_sec = np.full(len(P), np.nan)
    d_sec[ok] = design.section_distance(s[ok], o[ok], P[ok, 2])
    if engine == "cloudcompare":
        V, F = design.mesh()
        d = np.full(len(P), np.nan)
        d[ok] = cc.c2m(P[ok], V, F, workdir=out / "cc")
    else:
        d = d_sec
    ok &= np.abs(d) <= cfg["max_dist"]
    # CloudCompare vs the design sections, away from steps/corners (there the nearest wall isn't in the section)
    ck = ok & ~design.near_step(s, cfg["max_dist"] + 2.0)
    check = float(np.nanmax(np.abs(d - d_sec)[ck])) if ck.any() else 0.0

    stem = name or Path(scans[0]).stem
    with open(out / f"{stem} distances.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["P", "E", "N", "Z", "station", "offset", "used", "dist", "dist_section"])
        for i in range(len(P)):
            w.writerow([names[i] if i < len(names) else "", *np.round(P[i], 3), round(s[i], 3), round(o[i], 3),
                        int(ok[i]), "" if np.isnan(d[i]) else round(d[i], 3),
                        "" if np.isnan(d_sec[i]) else round(d_sec[i], 3)])

    bands = read_tcf(cfg["tcf"]) if cfg.get("tcf") else DEFAULT_BANDS
    map_pdf = out / f"{stem} C-F MAP - views.pdf"
    pages = []
    with PdfPages(map_pdf) as pdf:
        for sd in ("L", "R"):
            k = ok & (side == sd)
            if k.sum() < 3:
                continue
            ss, zz, dd = s[k], P[k, 2], d[k]
            cell = cfg["thin_cell"]
            if cell == "auto":
                cell = 0.1 if k.sum() > 20000 else 0
            if cell:
                ss, zz, dd = thin(ss, zz, dd, cell)
            for wa, wb in windows(ss, cfg["sheet_length"], cfg["grid"]["major_sta"]):
                near = (ss >= wa - 2 * cfg["max_edge"]) & (ss <= wb + 2 * cfg["max_edge"])
                kin = k & (s >= wa) & (s <= wb)
                if near.sum() < 3 or kin.sum() < 1:
                    continue
                info = map_page(pdf, design, ss[near], zz[near], dd[near], sd, cfg, bands, window=(wa, wb))
                pages.append({"side": sd, "wall": cfg["wall_names"].get(sd, sd), "info": info,
                              "n": int(kin.sum()), "s0": float(s[kin].min()), "s1": float(s[kin].max()),
                              "o0": float(np.abs(o[kin]).min()), "o1": float(np.abs(o[kin]).max()),
                              "z0": float(P[kin, 2].min()), "z1": float(P[kin, 2].max()), "engine": engine,
                              "labels_all": info["labels_all"]})
    if not pages:
        raise ValueError("no wall has 3 or more usable scan points")
    out_pdf = out / f"{stem} C-F MAP - TB.pdf"
    job_path = out / f"{stem} C-F MAP - sheet.json"
    job_path.write_text(json.dumps(sheet_job(cfg, design, [Path(p).name for p in scans], pages, map_pdf, out_pdf,
                                             stem), indent=2))
    built = False
    if build_sheet:
        r = subprocess.run([sys.executable, "-m", "sheetwrap", "build", str(job_path)], capture_output=True, text=True)
        built = r.returncode == 0
        if not built:
            print(f"sheetwrap did not build the sheet: {(r.stderr or r.stdout).strip()[-400:]}")
    return {"engine": engine, "points": len(P), "used": int(ok.sum()), "check_max_diff": check,
            "views": map_pdf, "job": job_path, "sheet": out_pdf if built else None,
            "walls": [{k: v for k, v in p.items() if k != "info"} | {"area": p["info"]["area"],
                                                                     "bands": p["info"]["bands"]} for p in pages]}
