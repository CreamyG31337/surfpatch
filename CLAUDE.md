# surfpatch

Tools for checking and repairing LandXML TIN surfaces built from survey data: trace vertices back to source
shots, find spikes, cut out and re-triangulate bad areas, insert missed shots, patch from drone/TLS point clouds,
and compare surfaces (volumes, before/after).

Status: the working code is in `legacy/` (one-off scripts, gitignored, see CLAUDE.local.md). The plan is to turn
them into a `surfpatch` package with a CLI (`trace`, `spikes`, `cut`, `insert`, `drone`, `diff`) and tests on
synthetic surfaces, the way rxlkit is tested.

## Sibling repos (open via surfpatch.code-workspace; access granted in .claude/settings.json)

- `../rxlkit` - RoadXML (.rxl) corridors to checked LandXML TINs. Reuse, don't copy:
  - `rxlkit.landxml.write_surfaces(path, [Surface])`, `Surface(name, points (N,E,Z), faces 0-based, desc)`
  - `rxlkit.tin.quality(P, F)` / `quality_ok(q)` - winding, zero area, duplicate XY, non-manifold edges
  - `rxlkit.tin.elevations_at`, `rxlkit.tin.distance`
  - Its `read_surfaces` uses ElementTree and needs `<Pnts>/<Faces>`: fine for normal files, not for 100+ MB
    files or TBC "source data only" exports (breaklines + DataPoints, no faces) - surfpatch needs a streaming reader.
  - Read `../rxlkit/README.md` before changing anything there; it has its own tests (`python -m unittest discover -s tests`).
- `../file-finder` - `survey_index.py`: indexes survey files (LandXML, LAZ, JXL scans, CSV, GeoTIFF, TBC) into an
  HTML report plus `<report>.footprints.json` (5 m cells `[ix, iy, count]` per CSV/surface; bbox per LAZ/ortho;
  station + frame polygon per JXL scan). Its CSV helpers (`_guess_enz`, `_numcount`, header-name sets) are what
  `legacy/trace.py` uses to read arbitrary survey CSVs. Read `../file-finder/README.md` for the details.
  file-finder has a graphify knowledge graph (`../file-finder/graphify-out/`); use `graphify query` there.

When a change belongs in a sibling (e.g. a generic TIN check -> rxlkit, a footprint field -> file-finder), make it
there, run that repo's tests, and say which repo was changed.

## Conventions

- Keep company/project/site specifics (paths, feature codes, surface names, thresholds tuned to one site) out of
  code and committed docs; they go in config or CLAUDE.local.md. Code defaults stay generic.
- Scripts are uv PEP 723 single files (`uv run --script x.py`). Write them with the Write tool: a UTF-8 BOM from
  PowerShell `Set-Content -Encoding utf8` breaks the `# /// script` header.
- Run Python that reads downloaded/untrusted files with `-I`.
- Never open OneDrive online-only files by accident (reading even a LAZ header downloads the whole file):
  check `os.stat(p).st_file_attributes & (0x400000 | 0x40000 | 0x1000)` first.
- Every TIN written must pass `rxlkit.tin.quality_ok`, plus an area check when a hole is re-triangulated.
- LandXML `<P>` is `N E Z`; internal arrays in surfpatch are (E, N, Z) unless named otherwise - convert at the edges.
