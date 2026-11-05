# surfpatch

Check and repair LandXML TIN surfaces built from survey data:

- trace surface vertices back to the survey shots they came from
- find spikes (vertices far from their neighbours)
- cut out a bad area and re-triangulate it, or insert missed shots
- patch a surface from drone or scanner point clouds
- compare two surfaces (volume, difference maps)

Work in progress: the tools are being turned into a package with a command-line interface. It builds on
[rxlkit](../rxlkit) for LandXML writing and TIN checks.

## License

MIT.
