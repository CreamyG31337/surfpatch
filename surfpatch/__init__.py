"""surfpatch: check and repair LandXML TIN surfaces built from survey data.

    from surfpatch import mesh, compare, io
    mesh.remove_vertices / insert_points / retriangulate   local re-triangulation, kept faces untouched
    compare.Sampler / diff / spikes / sparse_regions        checks and volumes
    io.read_shots / read_boundary / write_dxf_mesh / read_source_only / clip_las

Surfaces are rxlkit.landxml.Surface (points north, east, z).
"""
__version__ = "0.1.0"
