"""Synthetic tests for surfpatch.maputil (north_arrow, coord_grid, scale_bar)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surfpatch.maputil import coord_grid, north_arrow, scale_bar


def test_map_furniture_renders_cleanly(tmp_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([2000, 2050, 2050, 2000, 2000], [1000, 1000, 1030, 1030, 1000])
    ax.set_xlim(1990, 2060)
    ax.set_ylim(990, 1040)
    ax.set_aspect("equal")

    step = coord_grid(ax)
    assert step in (10.0, 20.0, 25.0)
    na = north_arrow(ax, size_in=0.6, corner="upper right")
    assert na is not None
    sb_len = scale_bar(ax, corner="lower left")
    assert sb_len > 0

    out = tmp_path / "plan.pdf"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 1000
