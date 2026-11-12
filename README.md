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
python -m surfpatch diff     NEW.xml OLD.xml [--boundary AREA.dxf] [--step 0.25]  volume NEW - OLD
python -m surfpatch todxf    SURF.xml [...]                                       R12 polyface mesh per surface
python -m surfpatch rebuild-source EXPORT.xml OUT.xml [--bbox EMIN NMIN EMAX NMAX]
```

- `cut --csv` removes the vertices that match a shot within 6 mm in 3D (`--tol`). The faces that used them are
  re-triangulated (constrained Delaunay, the hole's outline held fixed), so the patch meets the rest
  edge-for-edge. If a removed vertex was on the outer hull, the small hull area it made is lost and reported.
- `insert` skips shots within 0.10 m of an existing vertex (`--min-sep`), duplicates, and shots off the
  surface. It rebuilds the faces containing a shot plus one ring of neighbours (`--rings`).
- CSVs: a header row is used if present (Point/Northing/Easting/Elevation/Code or P/N/E/Z/D). Otherwise the
  columns are `--order PNEZD`.
- `rebuild-source` turns a TBC "source data only" export (2-point breaklines = TIN edges, no faces) back into a
  TIN. It streams the file, so 100+ MB exports are fine.
- Multi-surface files: `--surface NAME`.

Checked on real data: `cut` on a 14,000-point original-ground surface, using three CSVs of wrongly included
post-excavation shots, reproduced a hand-built repair exactly. It had the same points and faces and zero volume
difference, and spikes in the bad area went from 98 to 0.

## Library

```python
from rxlkit.landxml import read_surface, write_surfaces, Surface
from surfpatch import mesh, compare, io

s = read_surface("og.xml")                                    # points (north, east, z), faces 0-based
P, F, info = mesh.remove_vertices(s.points, s.faces, ids)     # or insert_points(P, F, pts) / retriangulate(...)
r = compare.diff((P, F), (s.points, s.faces), step=0.25, polygon=io.read_boundary("area.dxf"))
z = compare.Sampler(P, F)(east_array, north_array)            # NaN off the surface
v, dz = compare.spikes(P, F)
x, y, z, cls = io.clip_las("flight.laz", (emin, nmin, emax, nmax))   # None if online-only / outside
```

`mesh.PatchError` is raised instead of returning a bad TIN.

## Install / test

Inside `C:\Projects` it's already installed (editable) in the shared venv. Run
`C:\Projects\.venv\Scripts\python.exe -m pytest` in this folder; the tests use synthetic surfaces and need no site
data. Standalone: `uv sync` here (needs `../rxlkit`).

## Not yet

Drone-cloud patching (lowest-confirmed-flight grid, as in the OG rock job's `mcb_patch.py`), and tracing via
file-finder's footprints index (the job's `trace.py` does both).

## License

MIT.
