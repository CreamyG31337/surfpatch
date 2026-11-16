"""Point-label placement without overlaps (greedy candidate-position labelling, one pass).

Each label tries, in order:
  1. 8 positions touching its anchor (NE, NW, SE, SW, E, W, N, S) - no leader;
  2. rings of positions further out (16 directions per ring) - drawn with a leader line;
and takes the first that keeps its box clear of every placed label box, every anchor marker and every placed
leader, and inside the bounds. A leader also may not cross a placed box or leader. Labels that find no spot are
dropped (the anchor marker stays). Labels are placed in the given priority order, so put the important ones first.

All geometry is in points (1/72 in) in display orientation (x right, y up). Works for any anchor set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DIRECT = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]


@dataclass
class Placement:
    i: int                 # index of the anchor
    cx: float              # label box centre (pt)
    cy: float
    leader: tuple | None   # (x0, y0, x1, y1) from the box edge to the anchor, or None


def text_size(text: str, fontsize: float) -> tuple[float, float]:
    """(width, height) in pt of `text` set in matplotlib's default font, digits/caps height."""
    from matplotlib.textpath import TextPath
    ext = TextPath((0, 0), text, size=fontsize).get_extents()
    return float(ext.width), float(fontsize * 0.75)


class _Grid:
    """Uniform bucket grid of axis-aligned boxes and segments, for overlap queries."""

    def __init__(self, cell=24.0):
        self.c = cell
        self.b: dict[tuple, list] = {}

    def _keys(self, x0, y0, x1, y1):
        c = self.c
        for i in range(int(math.floor(x0 / c)), int(math.floor(x1 / c)) + 1):
            for j in range(int(math.floor(y0 / c)), int(math.floor(y1 / c)) + 1):
                yield i, j

    def add(self, item):
        kind, g = item
        x0, y0, x1, y1 = g if kind == "box" else (min(g[0], g[2]), min(g[1], g[3]), max(g[0], g[2]), max(g[1], g[3]))
        for k in self._keys(x0, y0, x1, y1):
            self.b.setdefault(k, []).append(item)

    def near(self, x0, y0, x1, y1):
        seen = set()
        for k in self._keys(x0, y0, x1, y1):
            for it in self.b.get(k, ()):
                if id(it) not in seen:
                    seen.add(id(it))
                    yield it


def _box_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _seg_cross(p, q):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    a, b, c, d = (p[0], p[1]), (p[2], p[3]), (q[0], q[1]), (q[2], q[3])
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def _seg_box(s, box):
    """Segment s intersects box (inclusive of either end inside)."""
    x0, y0, x1, y1 = box
    for px, py in ((s[0], s[1]), (s[2], s[3])):
        if x0 < px < x1 and y0 < py < y1:
            return True
    edges = [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
    return any(_seg_cross(s, e) for e in edges)


def _edge_point(box, x, y):
    """Point on the box boundary nearest to (x, y), for the leader end."""
    return min(max(x, box[0]), box[2]), min(max(y, box[1]), box[3])


def place_labels(xy, sizes, bounds, marker=2.5, gap=1.0, rings=(10.0, 16.0, 24.0, 34.0)):
    """Place one label per anchor.

    xy      (n, 2) anchors in pt; sizes: (n, 2) label (w, h) in pt; bounds: (x0, y0, x1, y1) labels must stay in.
    marker  half-size of the anchor marker (pt), kept clear of all labels; gap: clearance around boxes (pt).
    rings   leader distances (pt) from the anchor to the label box centre, tried after the direct positions.
    Returns a list of Placement for the labels that fit, in the order placed.
    """
    xy = np.asarray(xy, float)
    sizes = np.asarray(sizes, float)
    grid = _Grid(cell=max(24.0, float(sizes[:, 0].max()) if len(sizes) else 24.0))
    for x, y in xy:
        grid.add(("box", (x - marker, y - marker, x + marker, y + marker)))
    out = []
    angles = [k * math.pi / 8 for k in range(16)]
    for i, ((ax_, ay), (w, h)) in enumerate(zip(xy, sizes)):
        hw, hh = w / 2 + gap, h / 2 + gap
        cands = [(ax_ + dx * (marker + hw), ay + dy * (marker + hh), False) for dx, dy in DIRECT]
        for r in rings:
            cands += [(ax_ + (r + hw * 0.6) * math.cos(t), ay + (r + hh) * math.sin(t), True) for t in angles]
        own = (ax_ - marker, ay - marker, ax_ + marker, ay + marker)
        for cx, cy, lead in cands:
            box = (cx - hw, cy - hh, cx + hw, cy + hh)
            if box[0] < bounds[0] or box[1] < bounds[1] or box[2] > bounds[2] or box[3] > bounds[3]:
                continue
            seg = None
            if lead:
                ex, ey = _edge_point(box, ax_, ay)
                seg = (ax_, ay, ex, ey)
                if math.hypot(ex - ax_, ey - ay) < marker + 1:
                    continue
            q0 = (min(box[0], ax_), min(box[1], ay), max(box[2], ax_), max(box[3], ay))
            ok = True
            for kind, g in grid.near(*q0):
                if kind == "box":
                    if g == own:
                        continue
                    if _box_overlap(box, g) or (seg and _seg_box(seg, g)):
                        ok = False
                        break
                else:
                    if _seg_box(g, box) or (seg and _seg_cross(seg, g)):
                        ok = False
                        break
            if ok:
                grid.add(("box", box))
                if seg:
                    grid.add(("seg", seg))
                out.append(Placement(i, cx, cy, seg))
                break
    return out
