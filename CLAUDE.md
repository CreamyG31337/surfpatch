# surfpatch

Tool repo (being built): check and repair LandXML TIN surfaces built from survey data - trace vertices back to
source shots, find spikes, cut out and re-triangulate bad areas, insert missed shots, patch from drone/TLS point
clouds, compare surfaces (volumes, before/after). Planned CLI: `trace`, `spikes`, `cut`, `insert`, `drone`, `diff`,
with tests on synthetic surfaces like rxlkit's.

Source material: the one-off scripts in `C:\Projects\jobs\2026-11-05 OG rock + MCB surface repair\scripts\`
(see its NOTES.md). Generalise them; don't copy site specifics (paths, codes, surface names) into this repo.

## Reuse from sibling repos (the hub C:\Projects\CLAUDE.md lists all tools)

- `../rxlkit`: `rxlkit.landxml.write_surfaces`, `Surface`, `rxlkit.tin.quality` / `quality_ok`, `elevations_at`.
  Its `read_surfaces` needs `<Pnts>/<Faces>` and loads the whole file - surfpatch needs a streaming reader for
  100+ MB files and TBC source-data-only exports (2-point breaklines, no faces).
- `../file-finder`: footprints JSON format and CSV column guessing (`survey_index._guess_enz`, header-name sets).

## Conventions

- Code defaults generic; project specifics in config.
- uv PEP 723 for one-off scripts; write files with the Write tool (a BOM breaks the `# /// script` header).
- Check OneDrive online-only (`st_file_attributes & (0x400000 | 0x40000 | 0x1000)`) before opening big files.
- Every TIN written must pass `rxlkit.tin.quality_ok`, plus an area check when a hole is re-triangulated.
- LandXML `<P>` is `N E Z`; internal arrays are (E, N, Z) unless named otherwise.
