"""Plan-view survey map furniture for matplotlib figures (north arrow, coordinate grid, scale bar)."""
import math

import numpy as np


def north_arrow(ax, size_in=0.8, corner="upper right", pad_in=0.08, label="GRID", color="#0f172a"):
    """Draw a survey-style grid north arrow (split head, double ring, 'N') in a corner of a plan view.

    Uses its own equal-aspect inset axes so the arrow is never stretched and renders as crisp vector
    graphics in PDF output. Call after setting axis limits and aspect ratio.
    """
    from matplotlib import patheffects
    from matplotlib.patches import Circle, Polygon

    ax.apply_aspect()
    bb, (fw, fh) = ax.get_position(), ax.figure.get_size_inches()
    aw, ah = max(bb.width * fw, 1e-6), max(bb.height * fh, 1e-6)
    w, h = size_in / aw, size_in * 1.25 / ah
    px, py = pad_in / aw, pad_in / ah
    x = 1.0 - w - px if "right" in corner else px
    y = 1.0 - h - py if "upper" in corner else py

    na = ax.inset_axes([x, y, w, h], zorder=20)
    na.set_xlim(-1.0, 1.0)
    na.set_ylim(-1.15, 1.35)
    na.set_aspect("equal")
    na.axis("off")
    na.patch.set_alpha(0.0)
    na.set_anchor(("N" if "upper" in corner else "S") + ("E" if "right" in corner else "W"))

    white = "#ffffff"
    k = size_in / 0.95
    na.add_patch(Circle((0, 0), 0.92, facecolor=white, edgecolor="none", alpha=0.82))
    na.add_patch(Circle((0, 0), 0.62, facecolor="none", edgecolor=color, linewidth=1.1 * k))
    na.add_patch(Circle((0, 0), 0.54, facecolor="none", edgecolor=color, linewidth=0.5 * k))
    for a in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
        na.plot([0.62 * math.cos(a), 0.80 * math.cos(a)],
                [0.62 * math.sin(a), 0.80 * math.sin(a)],
                color=color, linewidth=1.0 * k, solid_capstyle="butt")
    tip, notch, lb, rb = (0, 0.98), (0, -0.40), (-0.30, -0.78), (0.30, -0.78)
    na.add_patch(Polygon([tip, lb, notch], closed=True, facecolor=color, edgecolor=color, linewidth=0.9 * k))
    na.add_patch(Polygon([tip, notch, rb], closed=True, facecolor=white, edgecolor=color, linewidth=0.9 * k))
    na.text(0, 1.18, "N", ha="center", va="center", fontsize=11 * k, fontweight="bold", color=color,
            path_effects=[patheffects.withStroke(linewidth=3.0 * k, foreground=white)])
    if label:
        na.text(0, -1.02, label, ha="center", va="center", fontsize=5.0 * k, color=color)
    return na


def _nice_step(span, target_ticks=5):
    if span <= 0:
        return 1.0
    raw = span / max(target_ticks, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if mult * mag >= raw:
            return mult * mag
    return 10.0 * mag


def coord_grid(ax, step=None, color="#64748b", alpha=0.35, lw=0.5, fontsize=7,
               xlabel="Easting (m)", ylabel="Northing (m)"):
    """Apply clean survey Easting/Northing grid lines and non-scientific tick labels to ``ax``."""
    from matplotlib.ticker import FixedFormatter, FixedLocator

    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    if step is None:
        step = _nice_step(max(x1 - x0, y1 - y0))

    xt = np.arange(math.ceil(x0 / step) * step, x1 + 1e-9, step)
    yt = np.arange(math.ceil(y0 / step) * step, y1 + 1e-9, step)
    dec = len(f"{step:.6f}".rstrip("0").split(".")[1])     # 10 -> 0, 2.5 -> 1, 0.25 -> 2
    fmt = f"{{:.{dec}f}}"

    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.xaxis.set_major_formatter(FixedFormatter([fmt.format(v) for v in xt]))
    ax.yaxis.set_major_formatter(FixedFormatter([fmt.format(v) for v in yt]))
    ax.tick_params(labelsize=fontsize)
    ax.grid(True, which="major", color=color, alpha=alpha, linewidth=lw, linestyle="--")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize + 1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize + 1)
    return step


def scale_bar(ax, length=None, corner="lower left", pad_frac=0.04, color="#0f172a", units="m"):
    """Draw a clean 2-segment alternating survey scale bar on an equal-aspect plan axes."""
    from matplotlib.patches import Rectangle

    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    span_x, span_y = x1 - x0, y1 - y0
    if length is None:
        length = _nice_step(span_x * 0.22, target_ticks=1)

    bar_h = span_y * 0.014
    pad_x = span_x * pad_frac
    pad_y = span_y * pad_frac
    bx = (x1 - pad_x - length) if "right" in corner else (x0 + pad_x)
    by = (y1 - pad_y - bar_h * 3.0) if "upper" in corner else (y0 + pad_y)

    # Translucent white backing
    ax.add_patch(Rectangle(
        (bx - span_x * 0.012, by - bar_h * 0.8),
        length + span_x * 0.024, bar_h * 3.4,
        facecolor="#ffffff", edgecolor="none", alpha=0.82, zorder=18,
    ))
    half = length / 2.0
    ax.add_patch(Rectangle((bx, by), half, bar_h, facecolor=color, edgecolor=color, lw=0.7, zorder=19))
    ax.add_patch(Rectangle((bx + half, by), half, bar_h, facecolor="#ffffff", edgecolor=color, lw=0.7, zorder=19))

    def _lbl(v):
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{v:g}"

    ty = by + bar_h * 1.4
    ax.text(bx, ty, "0", ha="center", va="bottom", fontsize=6.5, color=color, fontweight="bold", zorder=20)
    ax.text(bx + half, ty, _lbl(half), ha="center", va="bottom", fontsize=6.5, color=color, fontweight="bold", zorder=20)
    ax.text(bx + length, ty, f"{_lbl(length)} {units}", ha="center", va="bottom",
            fontsize=6.5, color=color, fontweight="bold", zorder=20)
    return length
