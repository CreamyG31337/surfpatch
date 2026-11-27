"""CloudCompare command line, headless (-SILENT): find the exe, run commands, cloud-to-mesh distances.

CloudCompare is a GUI-subsystem program, so it is always run with subprocess and waited for. Things this wrapper
handles so callers don't have to:
- UTM coordinates lose precision in CloudCompare's float32 storage; points and meshes are shifted by a common
  local origin before they are written and shifted back after.
- `-SAVE_CLOUDS FILE <name>` splits its argument on spaces, so CloudCompare runs inside the work folder with
  bare file names.
- Results come back in input order (checked).

Arrays here are (east, north, z) = CloudCompare's (x, y, z), not rxlkit's (north, east, z).
"""
import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_ROOTS = [r"C:\Projects\cloudcompare"]


class CCError(RuntimeError):
    pass


def find_exe():
    """CloudCompare.exe: $CLOUDCOMPARE_EXE, else the newest CloudCompare_*_bin_x64 under DEFAULT_ROOTS,
    else CloudCompare on PATH. None if not found."""
    env = os.environ.get("CLOUDCOMPARE_EXE")
    if env and Path(env).is_file():
        return Path(env)
    for root in DEFAULT_ROOTS:
        hits = sorted(glob.glob(os.path.join(root, "CloudCompare_*_bin_x64", "CloudCompare.exe")))
        if hits:
            return Path(hits[-1])
    w = shutil.which("CloudCompare")
    return Path(w) if w else None


def run(args, cwd, exe=None, log="cc_log.txt", timeout=1800):
    """Run CloudCompare -SILENT with `args` (a list, file names relative to cwd) and wait. Returns the log text.
    Raises CCError with the log tail if it fails."""
    exe = exe or find_exe()
    if exe is None:
        raise CCError("CloudCompare.exe not found (set CLOUDCOMPARE_EXE)")
    cmd = [str(exe), "-SILENT", "-LOG_FILE", log, "-AUTO_SAVE", "OFF"] + [str(a) for a in args]
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    lp = Path(cwd) / log
    text = lp.read_text(errors="replace") if lp.exists() else ""
    bad = [ln for ln in text.splitlines() if "Invalid parameter" in ln or "[ERROR]" in ln.upper()]
    if r.returncode != 0 or bad:
        tail = "\n".join((bad or text.splitlines())[-8:])
        raise CCError(f"CloudCompare failed (exit {r.returncode}):\n{tail}")
    return text


def write_obj(path, V, F):
    """Mesh as OBJ: V (n, 3) x y z, F (m, 3) 0-based. Face winding is kept (it sets the C2M sign)."""
    with open(path, "w") as fh:
        fh.writelines(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in V)
        fh.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in np.asarray(F, int))


def c2m(points, V, F, workdir=None, shift=None, max_dist=None, exe=None):
    """Signed cloud-to-mesh distance of each point (east, north, z) to the mesh (V east/north/z, F faces).
    + = on the side the faces' normals point to (right-hand rule on the winding). Returns an array in input
    order. workdir: keep the files there (default: a temp folder, removed afterwards)."""
    P = np.asarray(points, float)
    V = np.asarray(V, float)
    if shift is None:
        shift = np.floor(np.concatenate([P, V]).min(0) / 100) * 100
        shift[2] = 0.0
    tmp = None
    if workdir is None:
        tmp = tempfile.mkdtemp(prefix="cc_c2m_")
        workdir = tmp
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    try:
        np.savetxt(wd / "c2m_points.xyz", P - shift, fmt="%.4f")
        write_obj(wd / "c2m_mesh.obj", V - shift, F)
        args = ["-C_EXPORT_FMT", "ASC", "-PREC", "4", "-SEP", "SPACE",
                "-O", "c2m_points.xyz", "-O", "c2m_mesh.obj", "-C2M_DIST"]
        if max_dist:
            args += ["-MAX_DIST", f"{max_dist:g}"]
        args += ["-SAVE_CLOUDS", "FILE", "c2m_result.asc"]
        run(args, wd, exe=exe)
        a = np.loadtxt(wd / "c2m_result.asc", ndmin=2)
        if len(a) != len(P) or not np.allclose(a[:, :3] + shift, P, atol=2e-3):
            raise CCError("CloudCompare returned points in a different order or position")
        return a[:, -1]
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
