# surfpatch

Tool repo: check and repair LandXML TIN surfaces built from survey data. v0.1 (2026-11-11) has these CLI
commands: `info`, `spikes`, `sparse`, `trace`, `cut`, `insert`, `diff`, `todxf`, `rebuild-source`. See README.md.

- `surfpatch/mesh.py`: local re-triangulation core (`retriangulate`, `remove_vertices`, `insert_points`,
  `locate`, `grow`). Every edit to a TIN goes through `retriangulate`, which constrains the region outline,
  filters by centroid-in-region, checks area, orients faces, and refuses (PatchError) if TIN quality gets worse.
- `surfpatch/compare.py`: `Sampler`, `diff`, `spikes`, `sparse_regions`, `inside`.
- `surfpatch/io.py`: `read_shots` (CSV), `read_boundary`/`read_polygons` (ezdxf), `write_dxf_mesh`,
  `read_source_only`, `clip_las`, `is_online_only`.
- `tests/test_surfpatch.py`: synthetic grid TINs at site-scale coordinates.

Source material (kept as the record): `C:\Projects\jobs\2026-11-05 OG rock + MCB surface repair\scripts\`, plus
`og_patch.py`/`tin_diff.py` in `jobs\2026-11-10 channel wall asbuilt` and `jobs\2026-11-06 LLO wall meshes dxf`.
Still to port from those jobs: the drone patch (`mcb_patch.py`, `mcb_bias.py`) and footprints-based tracing
(`trace.py` + file-finder's `index.footprints.json`). Generalise; keep site specifics (paths, codes, surface
names) out of this repo.

## Conventions

- Run with the shared venv: `C:\Projects\.venv\Scripts\python.exe -m surfpatch ...` and `-m pytest`. Not the
  `surfpatch.exe` shim, which Defender ASR blocks.
- Surfaces are `rxlkit.landxml.Surface`: points (north, east, z), matching LandXML `<P>` order. Plan-only
  arrays are (east, north) and named `en`/`xy`. Polygons from DXF are (east, north).
- Every TIN written must pass `rxlkit.tin.quality_ok` (or be no worse than the input), plus the area check.
- Check OneDrive online-only (`io.is_online_only`) before opening big files on S:.
- Code defaults generic; project specifics in config or on the command line.
