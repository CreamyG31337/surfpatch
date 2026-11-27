"""Overlap-free label placement (surfpatch.labels)."""
import numpy as np

from surfpatch.labels import _box_overlap, _seg_box, place_labels


def boxes(placed, sizes, gap=1.0):
    return [(p.cx - sizes[p.i][0] / 2 - gap, p.cy - sizes[p.i][1] / 2 - gap,
             p.cx + sizes[p.i][0] / 2 + gap, p.cy + sizes[p.i][1] / 2 + gap) for p in placed]


def test_sparse_points_all_direct():
    xy = np.array([[50, 50], [150, 50], [50, 150]], float)
    sizes = [(24, 5)] * 3
    placed = place_labels(xy, sizes, (0, 0, 300, 300))
    assert len(placed) == 3 and all(p.leader is None for p in placed)


def test_dense_cluster_no_overlaps_and_priority():
    rng = np.random.default_rng(1)
    xy = rng.uniform(100, 160, (80, 2))           # far more labels than fit next to their points
    sizes = [(24, 5)] * 80
    placed = place_labels(xy, sizes, (0, 0, 300, 300), marker=2.5)
    assert 10 < len(placed) < 80
    assert [p.i for p in placed] == sorted(p.i for p in placed)   # placed in priority order
    bx = boxes(placed, sizes)
    for a in range(len(bx)):
        for b in range(a + 1, len(bx)):
            assert not _box_overlap(bx[a], bx[b])
        for x, y in xy:                            # no label covers any point's marker
            assert not _box_overlap(bx[a], (x - 2.5, y - 2.5, x + 2.5, y + 2.5))
        for q in placed:                           # no leader runs through another label
            if q.leader and q is not placed[a]:
                assert not _seg_box(q.leader, bx[a])
    assert any(p.leader for p in placed)


def test_bounds_respected():
    placed = place_labels(np.array([[2.0, 2.0]]), [(24, 5)], (0, 0, 100, 100))
    p = placed[0]
    assert p.cx - 12 >= 0 and p.cy - 2.5 >= 0
