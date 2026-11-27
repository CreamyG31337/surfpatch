# surfpatch

Check and repair LandXML TIN surfaces built from survey data. Every surface it writes passes
[rxlkit](../rxlkit)'s TIN checks (one winding, no zero-area faces, no duplicate XY, no edge on more than two
faces) plus an area check. Faces outside the repaired area are kept exactly as they were.

## Commands

```
python -m surfpatch info     SURF.xml [...]                       counts, extents, TIN checks
python -m surfpatch spikes   SURF.xml [--boundary B.dxf] [--threshold 0.75]
python -m surfpatch sparse   SURF.xml [--max-edge 12 --min-area 300] [--json regions.json]
python -m surfpatch trace    SURF.xml --csv "DIR\*.csv" [--boundary B.dxf]      which CSVs the vertices came from
python -m surfpatch cut      SURF.xml OUT.xml --csv BAD.csv ... [--code REGEX] [--name "X REV1"]
python -m surfpatch cut      SURF.xml OUT.xml --boundary AREA.dxf               remove every vertex inside
python -m surfpatch insert   SURF.xml OUT.xml --csv SHOTS.csv ... [--code REGEX] [--boundary AREA.dxf]
python -m surfpatch diff     NEW.xml OLD.xml [--boundary AREA.dxf] [--step 0.25]  volume NEW - OLD (grid, quick)
python -m surfpatch isopach  OG.xml ASBUILT.xml [--boundary AREA.dxf] [--interval 4 --origin 1] [--zero-dxf ZERO.dxf]
                             exact cut/fill, areas, max depths, volumes by elevation range (TBC Earthwork Volume Report)
python -m surfpatch drone    SURF.xml OUT_DIR --laz "flights\*.laz" [--boundary AREA.dxf] [--check TLS.laz]
python -m surfpatch todxf    SURF.xml [...]                                       R12 polyface mesh per surface
python -m surfpatch rebuild-source EXPORT.xml OUT.xml [--bbox EMIN NMIN EMAX NMAX]
python -m surfpatch mtx-to-xyz GRID.MTX [OUT.xyz]                               HYPACK .MTX bathymetry -> E N Z .xyz
python -m surfpatch cfmap    CONFIG.json SCAN.csv|.laz [...] --out DIR [--name STEM] [--engine section]
python -m surfpatch volreport JOB.json [NAME...] [--out DIR] [--overwrite]        3-page TIN-to-TIN volume report PDFs
                                                    (existing PDFs are kept unless --overwrite)
python -m surfpatch lazcheck FLIGHT.laz --control CTRL.csv [--radius 0.5] [--classes 2] [--stat plane|median|low]
                                                    vertical bias & planar tilt vs control (robust local plane per point)
python -m surfpatch lazcheck NEW.laz --against OLD.laz [--cell 1]                 flight vs flight: lowest-per-cell difference,
                                                    tilt plane (real change dropped) and 100 m block medians
```

- `cut --csv` removes the vertices that match a shot within 6 mm in 3D (`--tol`). The faces that used them are
  re-triangulated (constrained Delaunay, the hole's outline held fixed), so the patch meets the rest
  edge-for-edge. If a removed vertex was on the outer hull, the small hull area it made is lost and reported.
- `insert` skips shots within 0.10 m of an existing vertex (`--min-sep`), duplicates, and shots off the
  surface. It rebuilds the faces containing a shot plus one ring of neighbours (`--rings`).
- CSVs: a header row is used if present (Point/Northing/Easting/Elevation/Code or P/N/E/Z/D). Otherwise the
  columns are `--order PNEZD`.
- `rebuild-source` turns a TBC "source data only" export (2-point breaklines = TIN edges, no faces) back into a
  TIN. It streams the file, so 100+ MB exports are fine. Where breaklines cross, a point is added at the crossing
  (elevation interpolated along the breaklines). If the export has a `<Boundary>` (outer / void), the TIN is clipped
  to it, as TBC does; only without one are hull triangles with an edge over `--max-edge` (15 m) dropped. (Before
  2026-11-20 it always used the 15 m limit, which removed the plateau of a designed pad, and fell back to a plain
  Delaunay when breaklines crossed.)
- `mtx-to-xyz` converts a HYPACK sparse ASCII matrix (.MTX: origin, size, cell size, clockwise rotation, then
  `cell z` rows) to a space-delimited `E N Z` file, points at cell centres. Checked exact against paired
  .MTX/.xyz deliveries. In Python: `io.read_mtx`, `io.read_xyz`, `io.write_xyz`, and `tinbuild.tin_from_grid`
  (Delaunay, edges over 2.6 m dropped: matches TBC's TIN of a 0.5 m grid).
- `drone` patches a surface that is out of date because the ground was dug lower after it was built. The flights
  are binned into 0.5 m cells (`--step`), taking each flight's median Z per cell. A cell's ground is the lowest
  value that a second flight confirms within 0.10 m (`--confirm`), so one bad or low flight is never used on its
  own. The patch covers only cells where that ground is more than 0.10 m below the surface (`--min-drop`). Wall
  cells (high spread or steeper than 45 deg) are not trusted. Neither are cells where an independent `--check`
  cloud (e.g. a terrestrial scan) disagrees by more than 0.25 m. Small pieces under 4 m2 (`--min-area`) are
  dropped. The patch never raises the surface. It writes `NAME.xml` (grid TIN), `NAME.csv` (PNEZD) and
  `NAME BNDY.dxf` to OUT_DIR, for pasting into the composite in TBC. OneDrive online-only flights are skipped
  unless `--download` is given.
- Multi-surface files: `--surface NAME`.
- `cfmap` makes cut/fill map sheets of a wall scan against a design corridor (.rxl).
  - **Distance:** each point is measured square to the design wall, by CloudCompare cloud-to-mesh (`surfpatch.cc`)
    or, without CloudCompare, the design cross-section (`--engine section`). The wall mesh is built from the RXL
    (`rxlkit.wall_surface`) unless the config names LandXML walls. Faces are wound up, i.e. toward the open side.
  - **Points dropped:** floor points (below toe + `toe_margin`), points off the walls' station range, and points
    more than `max_dist` from a wall.
  - **Map:** each wall is unrolled to station / elevation and split into `sheet_length` (40 m) windows. Points are
    triangulated, with triangles over `max_edge` dropped as gaps. The view is coloured with stepped TBC cut/fill
    bands (a TBC `.tcf` file or the built-in scheme), outlined, and labelled C/F where the labels fit.
  - **Sheets:** sheetwrap places the views on title-blocked sheets.
  - **Outputs:** `STEM distances.csv`, `STEM C-F MAP - views.pdf`, `STEM C-F MAP - sheet.json` (editable;
    rebuild with `python -m sheetwrap build`) and `STEM C-F MAP - TB.pdf`.
  - **Config:** keys are listed in `surfpatch/cfmap.py`.
  - **Sign:** distances.csv `dist` is + for rock inside the design (cut). The sheet shows TBC's C (cut) / F (fill).
- `volreport` builds 3-page TIN-to-TIN volume report PDFs from a JSON job file (depression reports or
  surface-to-surface reports along a RoadXML `.rxl` corridor):
  - **Page 1:** summary (stations, placement survey dates, bottom/top descriptions with `YY-DDD` converted to readable
    dates, exact isopach fill/cut/area/max fill depth, optional slope-length deduction, `NET FILL`, field data table,
    and an alignment locator map over an optional orthophoto).
  - **Page 2:** plan view of fill depth over the orthophoto with TIN edges, red boundary, station ticks, section line
    A-A, scale bar, north arrow, and colorbar.
  - **Page 3:** true-scale cross-section A-A at the top-surface shot line nearest mid-report (preferring lines with
    cut <= 5% of fill), showing this report's fill (and any cut), earlier placements down to a configurable base line,
    and optional design template shapes (`box`, `circles`).
  - **Header/footer:** optional vector logo (SVG/PDF, `${ENV}` expanded and skipped quietly if missing) top right,
    plus output path, timestamp, user, project name, and `Page i of N` on every page. If the target PDF is open in a
    viewer, writes `<name> Volume Report (NEW).pdf` beside it.

### `volreport` job-file schema

Relative paths resolve against the JSON file's directory. Example (synthetic local coordinates):

```json
{
  "alignment": "trench.rxl",
  "year": 2026,
  "out_dir": "reports",
  "project": "0000 - UTILITY TRENCH",
  "title": "Volume Report",
  "subtitle": "Utility Trench - Section 1",
  "crs_note": "LOCAL GRID",
  "logo": "${COMPANY_LOGO}",
  "base_label": "trench bottom",
  "ortho": {
    "image": "ortho.jpg",
    "cache": "ortho_crop.jpg",
    "date": "Oct 30 2026",
    "downsample": 2
  },
  "footer_note": "Stations on alignment {alignment}.  Volumes TIN to TIN (isopach).  Orthophoto {ortho_date}.",
  "field": {
    "glob": "field/*/CREW/{yy}-{doy:03d}-*{filter}*.csv",
    "filename_date": "(?P<yy>\\d\\d)-(?P<doy>\\d{3})-",
    "exclude": ["ORIG", "UTIL", "ASBUILT"],
    "filter": "TRENCH",
    "max_offset": 5.0
  },
  "design": [
    {"from": 0, "to": 50, "shape": "circles", "d": 0.1143, "x": [-0.2, 0.0, 0.2], "z": [0.14285, -0.05715], "label": "ducts (design)"},
    {"from": 50, "shape": "box", "w": 0.77, "h": 0.57, "top": 0.0, "label": "concrete encasement (design)"},
    {"from": 50, "shape": "circles", "d": 0.1143, "x": [-0.2, 0.0, 0.2], "z": [-0.185, -0.385], "label": "ducts (design)"}
  ],
  "reports": [
    {
      "name": "26-299 TRENCH LIFT1",
      "kind": "depression",
      "bottom_csv": "26-299 TRENCH LIFT1.csv",
      "boundary_dxf": "26-299 TRENCH LIFT1 BOUNDARY.dxf",
      "top_csv": "26-299 TRENCH LIFT1 TOP.csv",
      "bottom_desc": "Trench bottom",
      "top_desc": "26-299 FILL LIFT1",
      "sources": {"bottom": [[290, "BOT"]], "top": [[299, "LIFT1"]]},
      "base": "self",
      "deduction": {
        "label": "DUCT VOL",
        "area_m2": 0.01026,
        "count": 6,
        "stations": [10.0, 30.0]
      }
    },
    {
      "name": "26-301 LIFT2",
      "kind": "surf2surf",
      "bottom_specs": [[299, "LIFT1"]],
      "top_specs": [[301, "LIFT2"]],
      "stations": [10.0, 30.0],
      "bottom_desc": "26-299 FILL LIFT1",
      "top_desc": "26-301 FILL LIFT2",
      "sources": {"bottom": [[299, "LIFT1"]], "top": [[301, "LIFT2"]]},
      "base": [{"report": "26-299 TRENCH LIFT1"}]
    }
  ]
}
```

- `kind = "depression"`: constrained TIN of `bottom_csv` with rings from `boundary_dxf` vs `top_csv` + rim ring (or lid of the rim ring alone when `top_csv` is omitted).
- `kind = "surf2surf"`: plain Delaunay TINs from `bottom_specs`/`top_specs` (or `bottom_csv`/`top_csv` or `bottom_xml`/`top_xml`), clipped to `boundary_dxf` or the overlap of both convex hulls inside `stations`.
- `base`: `"self"`, or a preference-ordered list of `{"report": name}`, `{"depression": stem}`, `{"field": [day, code]}`, or `{"xml": path}`.
- `section_station` (optional): a station, or a list of stations, e.g. `[13.0, 28.0]`. Each one gets its own
  section page (A-A, B-B, ...) and its own line on the plan. By default there is one section, at the top-surface
  shot line nearest mid-report that is nearly all fill. Set it when that lands on the edge of a structure, or to
  show both sides of one.
- `deduction`: `count * area_m2 * slope_length` over `stations` along the corridor profile; optional `"format"` string customises the stamp text (`{label}`, `{count}`, `{area_m2}`, `{slope}`, `{volume}`).

Checked on real data: `cut` on a 14,000-point original-ground surface, using three CSVs of wrongly included
post-excavation shots, reproduced a hand-built repair exactly. It had the same points and faces and zero volume
difference, and spikes in the bad area went from 98 to 0. `drone` was checked against a hand-built patch of a
1,600 m2 excavation composite. With 8 of the original 11 weekly flights it matched exactly on 75% of the shared
area. The rest differs where the 3 missing flights had supplied the lower ground.

`isopach` overlays the two triangulations in plan (each triangle of one surface cut by the triangles of the
other and by the boundary), so on every piece both surfaces are flat planes and the volume is exact. Pieces are
also split at the cut/fill zero line and wherever either surface crosses an elevation-range limit. Checked
against two TBC Earthwork Volume Reports with the exact input surfaces (49,000 and 65,000 m3 cut): totals within
0.01 %, fill, max depths and z range identical, all but two elevation ranges within 0.1 m3. It
handles about 1 M triangles in 20-30 s. `diff` (grid sampling) is faster and, at 0.1 m cells, was within 0.005 %
of `isopach` on that job; its error is at the surface and boundary edges.

## Library

```python
from rxlkit.landxml import read_surface, write_surfaces, Surface
from surfpatch import mesh, compare, io

s = read_surface("og.xml")                                    # points (north, east, z), faces 0-based
P, F, info = mesh.remove_vertices(s.points, s.faces, ids)     # or insert_points(P, F, pts) / retriangulate(...)
r = compare.diff((P, F), (s.points, s.faces), step=0.25, polygon=io.read_boundary("area.dxf"))
r = isopach.isopach((og.points, og.faces), (ab.points, ab.faces), polygon, interval=4, origin=1)
#   r.cut, r.fill, r.area_cut, r.area_fill, r.area_compared, r.max_cut, r.max_fill, r.bins = [(z1, z2, cut, fill)]
z = compare.Sampler(P, F)(east_array, north_array)            # NaN off the surface
v, dz = compare.spikes(P, F)
x, y, z, cls = io.clip_las("flight.laz", (emin, nmin, emax, nmax))   # None if online-only / outside
from surfpatch import drone
r = drone.drone_patch(P, F, [(x1, y1, z1), (x2, y2, z2), ...], mask_polygon=None, check=None)
r.points, r.faces, r.stats                                      # patch TIN + area / volume / flights used
```

`mesh.PatchError` is raised instead of returning a bad TIN.

```python
from surfpatch import cc, cfmap, tinbuild, fielddata, volreport, maputil, pipes, pdfmarkup, lazcheck
d = cc.c2m(points_enz, V_enz, F)                        # signed cloud-to-mesh via CloudCompare -SILENT (+ = normal side)
r = cfmap.run("site_cfmap.json", ["scan.csv"], "out")   # same as the cfmap command; returns paths + stats
bot, lid = tinbuild.depression(P_bot, [rim, bring], rim)
res = volreport.run("job.json", out_dir="out")          # 3-page volume report PDFs; returns per-report metrics
maputil.north_arrow(ax); maputil.coord_grid(ax); maputil.scale_bar(ax)
pipes.pipes_to_dxf("pipes.dxf", [pipes.Pipe("PF1", cl_nez, 0.103)])      # miter-correct rings at bends
p = pipes.Pipe.from_segments("PF1", line_curve_segs, 0.103)             # true arcs: rails written as <Curve>
pipes.pipes_to_landxml("pipes.xml", [p]); pipes.min_pipe_clearance([p, q])   # exact segment-segment gap
pdfmarkup.overlay_boundaries("plan.pdf", "marked.pdf", pages_georef, layers)  # + "legend_rect" per page
chk = lazcheck.check_laz_bias("flight.laz", "control.csv", classes=(2,))
pair = lazcheck.check_laz_pair("nov.laz", "oct.laz", cell=1.0)
```

`cc` finds CloudCompare from `$CLOUDCOMPARE_EXE` or `C:\Projects\cloudcompare\CloudCompare_*_bin_x64`. It shifts
UTM coordinates to a local origin, runs CloudCompare inside the work folder (its `-SAVE_CLOUDS FILE` splits on
spaces), and checks that the results come back in input order. Tests that need CloudCompare skip without it.

## Install / test

Inside `C:\Projects` it's already installed (editable) in the shared venv. Run
`C:\Projects\.venv\Scripts\python.exe -m pytest` in this folder; the tests use synthetic surfaces and need no site
data. Standalone: `uv sync` here (needs `../rxlkit`).

## Not yet

Tracing via file-finder's footprints index (finding the candidate CSVs automatically); `trace` takes the CSVs
to check as arguments.

## License

MIT.
