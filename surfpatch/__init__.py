"""surfpatch: check and repair LandXML TIN surfaces built from survey data.

    from surfpatch import mesh, compare, io, tinbuild, fielddata, volreport, maputil, pipes, pdfmarkup, lazcheck
    mesh.remove_vertices / insert_points / retriangulate   local re-triangulation, kept faces untouched
    compare.Sampler / diff / spikes / sparse_regions        checks and volumes
    io.read_shots / read_boundary / write_dxf_mesh / read_source_only / clip_las / ortho_crop
    tinbuild.tin_from_points / depression / overlap_boundary
    volreport.run                                           3-page TIN-to-TIN volume report PDFs
    maputil.north_arrow / coord_grid / scale_bar            plan-view survey map furniture
    pipes.sweep_pipe / pipes_to_dxf / pipes_to_landxml      3D pipe/conduit meshes, rails, clearance checks
    pdfmarkup.fit_affine / overlay_boundaries               georeferenced Bluebeam/Acrobat layered PDF markups
    lazcheck.check_laz_bias                                 streaming LAZ vs control vertical bias & tilt check

Surfaces are rxlkit.landxml.Surface (points north, east, z).
"""
__version__ = "0.1.0"
