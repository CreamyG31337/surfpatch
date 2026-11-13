# surfpatch

Tool repo: check and repair LandXML TIN surfaces built from survey data. v0.1 (2026-11-11) has these CLI
commands: `info`, `spikes`, `sparse`, `trace`, `cut`, `insert`, `diff`, `drone`, `todxf`, `rebuild-source`,
`cfmap` (added 2026-11-13).
See README.md.

- `surfpatch/mesh.py`: local re-triangulation core (`retriangulate`, `remove_vertices`, `insert_points`,
  `locate`, `grow`). Every edit to a TIN goes through `retriangulate`, which constrains the region outline,
  filters by centroid-in-region, checks area, orients faces, and refuses (PatchError) if TIN quality gets worse.
- `surfpatch/compare.py`: `Sampler`, `diff`, `spikes`, `sparse_regions`, `inside`.
- `surfpatch/io.py`: `read_shots` (CSV), `read_boundary`/`read_polygons` (ezdxf), `write_dxf_mesh`,
  `read_source_only`, `clip_las`, `is_online_only`.
- `surfpatch/drone.py`: `drone_patch` (confirmed-lowest ground from repeat flights), `cell_stats`,
  `confirmed_lowest`, `grid_tin`, `region_outline`. Ported from the job's `mcb_patch.py`; the site-specific
  "only where the composite came from the 26-07 surface" mask became the generic `mask_polygon`/`--boundary`,
  and the Sep TLS check became `check`/`--check`.
- `surfpatch/cc.py`: CloudCompare command-line wrapper (`find_exe`, `run`, `c2m`). Arrays are (east, north, z).
- `surfpatch/cfmap.py`: cut/fill map sheets of wall scans vs an .rxl corridor. Ported 2026-11-13 from
  `jobs\2026-11-13 LLO channel folder cleanup\cc_wall_compare.py`; the site config is `llo_cfmap.json` there.
  Design wall faces are wound up (CCW in plan), not "toward the CL", which got corner cross faces wrong.
- `tests/test_surfpatch.py`, `tests/test_drone.py`, `tests/test_cfmap.py`: synthetic data at site-scale
  coordinates. The CloudCompare tests skip when it isn't installed.

Source material (kept as the record): `C:\Projects\jobs\2026-11-05 OG rock + MCB surface repair\scripts\`, plus
`og_patch.py`/`tin_diff.py` in `jobs\2026-11-10 channel wall asbuilt` and `jobs\2026-11-06 LLO wall meshes dxf`.
Still to port from those jobs: footprints-based tracing (`trace.py` + file-finder's `index.footprints.json`),
and the drone vs TLS bias check (`mcb_bias.py`, a one-off analysis). Generalise; keep site specifics (paths, codes, surface
names) out of this repo.

## Conventions

- Run with the shared venv: `C:\Projects\.venv\Scripts\python.exe -m surfpatch ...` and `-m pytest`. Not the
  `surfpatch.exe` shim, which Defender ASR blocks.
- Surfaces are `rxlkit.landxml.Surface`: points (north, east, z), matching LandXML `<P>` order. Plan-only
  arrays are (east, north) and named `en`/`xy`. Polygons from DXF are (east, north).
- Every TIN written must pass `rxlkit.tin.quality_ok` (or be no worse than the input), plus the area check.
- Check OneDrive online-only (`io.is_online_only`) before opening big files on S:.
- Code defaults generic; project specifics in config or on the command line.
