"""Command line: surfpatch <command> ...   (python -m surfpatch also works)

  info     XML...                          surfaces, counts, extents, TIN checks
  spikes   XML [--boundary DXF]            vertices far from their neighbours' median
  sparse   XML                             areas of long-edged triangles (missing data)
  trace    XML --csv F...                  which CSV files the vertices came from
  cut      XML OUT (--csv F... | --boundary DXF)   remove vertices, re-triangulate only the hole
  insert   XML OUT --csv F...              add shots, re-triangulate only around them
  diff     NEW OLD [--boundary DXF]        volume NEW - OLD
  todxf    XML...                          LandXML TIN -> DXF polyface mesh
  rebuild-source XML OUT                   TBC source-data-only export -> TIN

CSV arguments accept wildcards. Surfaces are read/written with rxlkit.landxml; every written TIN passes
rxlkit.tin.quality_ok and an area check.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from rxlkit.landxml import Surface, read_surface, read_surfaces, write_surfaces
from rxlkit.tin import quality, quality_ok

from . import compare, io, mesh


def _csvs(patterns):
    out = []
    for p in patterns or []:
        hits = sorted(glob.glob(p))
        if not hits and not os.path.exists(p):
            sys.exit(f"no file matches {p}")
        out += hits or [p]
    return out


def _surface(a):
    return read_surface(a.xml, a.surface)


def _write(a, s, P, F, desc):
    if os.path.abspath(a.out) == os.path.abspath(a.xml):
        sys.exit("refusing to overwrite the input; choose another OUT")
    name = a.name or s.name
    write_surfaces(a.out, [Surface(name, P, F, desc, s.color)], application="surfpatch")
    print(f"wrote {a.out}  ('{name}': {len(P)} points, {len(F)} faces)")


def _report(info):
    q = info["quality"]
    print(f"rebuilt {info['region_faces']} faces -> {info['patch_faces']}; removed {info['removed']} vertices, "
          f"added {info['added']}; area {info['region_area']:.2f} -> {info['patch_area']:.2f} m2"
          + (f" (hull area lost {info['hull_area_lost']:.2f} m2)" if info["hull_area_lost"] > 0.01 else ""))
    for k in ("skipped_near_vertex", "skipped_duplicate", "skipped_outside"):
        if info.get(k):
            print(f"  {k.replace('_', ' ')}: {info[k]}")
    if quality_ok(q):
        print(f"TIN checks: OK {q}")
    else:
        print(f"TIN checks: input already failed them and the patch made nothing worse {q}")


def cmd_info(a):
    for path in a.xml:
        for s in read_surfaces(path):
            q = quality(s.points, s.faces) if len(s.faces) else {}
            P = s.points
            print(f"{os.path.basename(path)} :: '{s.name}'  {len(P)} points, {len(s.faces)} faces")
            if len(P):
                print(f"  E {P[:, 1].min():.2f}..{P[:, 1].max():.2f}  N {P[:, 0].min():.2f}..{P[:, 0].max():.2f}"
                      f"  Z {P[:, 2].min():.2f}..{P[:, 2].max():.2f}")
            if q:
                print(f"  TIN checks: {'OK' if quality_ok(q) else 'FAILED'} {q}")
            elif len(P):
                print("  no faces: TBC 'source data only' export? try: surfpatch rebuild-source")


def cmd_spikes(a):
    s = _surface(a)
    verts = None
    if a.boundary:
        pg = io.read_boundary(a.boundary)
        verts = np.flatnonzero(compare.inside(pg, s.points[:, 1], s.points[:, 0]))
        verts = np.intersect1d(verts, np.unique(s.faces))
    v, dz = compare.spikes(s.points, s.faces, verts)
    big = np.abs(dz) > a.threshold
    print(f"'{s.name}': {len(v)} vertices checked, |dz| > {a.threshold} m: {big.sum()}, > {2 * a.threshold} m: "
          f"{(np.abs(dz) > 2 * a.threshold).sum()}")
    order = np.argsort(-np.nan_to_num(np.abs(dz)))[:a.top]
    for k in order:
        if not big[k]:
            break
        n, e, z = s.points[v[k]]
        print(f"  E {e:.3f} N {n:.3f} Z {z:.3f}  dz {dz[k]:+.2f}")


def cmd_sparse(a):
    s = _surface(a)
    regs = compare.sparse_regions(s.points, s.faces, a.max_edge, a.min_area)
    print(f"'{s.name}': {len(regs)} sparse area(s) (faces with an edge > {a.max_edge} m, clusters > {a.min_area} m2)")
    for r in regs:
        print(f"  {r['area']:8.0f} m2  {len(r['faces'])} faces  ~{r['size'][0]:.0f} x {r['size'][1]:.0f} m  "
              f"centre E {r['centre'][0]:.1f} N {r['centre'][1]:.1f}  z {r['z'][0]:.2f}..{r['z'][1]:.2f}")
    if a.json:
        en = s.points[:, [1, 0]]
        json.dump([{"area": r["area"], "centre": r["centre"],
                    "tris": en[s.faces[r["faces"]]].tolist()} for r in regs], open(a.json, "w"))
        print("wrote", a.json)


def cmd_trace(a):
    s = _surface(a)
    verts = np.unique(s.faces)
    if a.boundary:
        pg = io.read_boundary(a.boundary)
        verts = verts[compare.inside(pg, s.points[verts, 1], s.points[verts, 0])]
    _, dz = compare.spikes(s.points, s.faces, verts)
    tree = cKDTree(s.points[verts])
    hit = np.zeros(len(verts), bool)
    rows = []
    for path in _csvs(a.csv):
        sh = io.read_shots(path, a.order)
        if not len(sh):
            continue
        d, k = tree.query(sh.points, distance_upper_bound=a.tol)
        k = np.unique(k[np.isfinite(d)])
        if len(k):
            hit[k] = True
            rows.append((len(k), int((np.abs(dz[k]) > 0.75).sum()), os.path.basename(path),
                         sorted({c for c, dd in zip(sh.code, d) if np.isfinite(dd)})[:8]))
    print(f"'{s.name}': {len(verts)} vertices, matched to a CSV shot (within {a.tol} m): {hit.sum()}")
    for n, sp, name, codes in sorted(rows, reverse=True):
        print(f"  {n:6d} vertices  spikes>0.75 m {sp:4d}  {name}  codes={codes}")
    if (~hit).any():
        print(f"  {(~hit).sum():6d} vertices from none of these files")


def cmd_cut(a):
    s = _surface(a)
    used = np.unique(s.faces)
    if a.csv:
        sh = io.read_shots_many(_csvs(a.csv), a.order, a.code)
        d, k = cKDTree(s.points[used]).query(sh.points, distance_upper_bound=a.tol)
        drop = np.unique(used[k[np.isfinite(d)]])
        why = f"{len(sh)} shots from {len(a.csv)} CSV pattern(s)"
    elif a.boundary:
        pg = io.read_boundary(a.boundary)
        drop = used[compare.inside(pg, s.points[used, 1], s.points[used, 0])]
        why = f"inside {os.path.basename(a.boundary)}"
    else:
        sys.exit("give --csv or --boundary")
    print(f"vertices to remove ({why}): {len(drop)}")
    P, F, info = mesh.remove_vertices(s.points, s.faces, drop)
    _report(info)
    _write(a, s, P, F, a.desc or f"{s.name} with {len(drop)} vertices removed ({why})")


def cmd_insert(a):
    s = _surface(a)
    sh = io.read_shots_many(_csvs(a.csv), a.order, a.code)
    pts = sh.points
    if a.boundary:
        pg = io.read_boundary(a.boundary)
        pts = pts[compare.inside(pg, pts[:, 1], pts[:, 0])]
    print(f"shots to insert: {len(pts)}")
    P, F, info = mesh.insert_points(s.points, s.faces, pts, a.min_sep, a.rings)
    _report(info)
    _write(a, s, P, F, a.desc or f"{s.name} plus {info['added']} shots")


def cmd_diff(a):
    new, old = read_surface(a.new, a.surface), read_surface(a.old, a.surface_old or a.surface)
    pg = io.read_boundary(a.boundary) if a.boundary else None
    r = compare.diff((new.points, new.faces), (old.points, old.faces), a.step, pg)
    print(f"NEW '{new.name}' minus OLD '{old.name}'" + (f" inside {os.path.basename(a.boundary)}" if pg is not None else ""))
    print(f"  compared {r.compared_area:,.0f} m2, changed {r.changed_area:,.0f} m2 (|dz| > 1 mm), cell {a.step} m")
    print(f"  raised +{r.raised:,.1f} m3, lowered -{r.lowered:,.1f} m3, net {r.net:+,.1f} m3; "
          f"max raise {r.max_raise:+.3f} m, max drop {r.max_drop:+.3f} m")


def cmd_todxf(a):
    for path in a.xml:
        surfs = read_surfaces(path)
        dst = os.path.splitext(path)[0] + ".dxf" if not a.out else a.out
        io.write_dxf_mesh(dst, surfs)
        print(f"wrote {dst}: " + ", ".join(f"'{s.name}' ({len(s.points)} pts, {len(s.faces)} faces)" for s in surfs))


def cmd_rebuild_source(a):
    s = io.read_source_only(a.xml, a.bbox, a.max_edge, a.name)
    q = quality(s.points, s.faces)
    print(f"'{s.name}': {len(s.points)} points, {len(s.faces)} faces; TIN checks {'OK' if quality_ok(q) else 'FAILED'} {q}")
    write_surfaces(a.out, [s], application="surfpatch")
    print("wrote", a.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="surfpatch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn, xml=True, out=False):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        if xml:
            sp.add_argument("xml")
            sp.add_argument("--surface", help="surface name when the file holds several")
        if out:
            sp.add_argument("out")
            sp.add_argument("--name", help="name of the output surface (default: input name)")
            sp.add_argument("--desc", help="description written into the output surface")
        return sp

    def shots(sp, required):
        sp.add_argument("--csv", nargs="+", required=required, help="survey CSV files (wildcards ok)")
        sp.add_argument("--code", help="regex: only shots whose code matches")
        sp.add_argument("--order", default="PNEZD", help="column order for CSVs without a header")

    sp = sub.add_parser("info"); sp.set_defaults(fn=cmd_info); sp.add_argument("xml", nargs="+")
    sp = p("spikes", cmd_spikes); sp.add_argument("--boundary"); sp.add_argument("--threshold", type=float, default=0.75)
    sp.add_argument("--top", type=int, default=20)
    sp = p("sparse", cmd_sparse); sp.add_argument("--max-edge", type=float, default=12.0)
    sp.add_argument("--min-area", type=float, default=300.0); sp.add_argument("--json", help="write regions (for insert)")
    sp = p("trace", cmd_trace); shots(sp, True); sp.add_argument("--boundary"); sp.add_argument("--tol", type=float, default=0.006)
    sp = p("cut", cmd_cut, out=True); shots(sp, False); sp.add_argument("--boundary")
    sp.add_argument("--tol", type=float, default=0.006, help="3D match distance shot -> vertex (m)")
    sp = p("insert", cmd_insert, out=True); shots(sp, True); sp.add_argument("--boundary")
    sp.add_argument("--min-sep", type=float, default=0.10); sp.add_argument("--rings", type=int, default=1)
    sp = sub.add_parser("diff"); sp.set_defaults(fn=cmd_diff); sp.add_argument("new"); sp.add_argument("old")
    sp.add_argument("--surface"); sp.add_argument("--surface-old"); sp.add_argument("--boundary")
    sp.add_argument("--step", type=float, default=0.25)
    sp = sub.add_parser("todxf"); sp.set_defaults(fn=cmd_todxf); sp.add_argument("xml", nargs="+"); sp.add_argument("--out")
    sp = sub.add_parser("rebuild-source"); sp.set_defaults(fn=cmd_rebuild_source); sp.add_argument("xml"); sp.add_argument("out")
    sp.add_argument("--bbox", type=float, nargs=4, metavar=("EMIN", "NMIN", "EMAX", "NMAX"))
    sp.add_argument("--max-edge", type=float, default=15.0); sp.add_argument("--name")

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except mesh.PatchError as e:
        sys.exit(f"not written: {e}")


if __name__ == "__main__":
    main()
