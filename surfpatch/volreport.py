"""Generic TIN-to-TIN volume report PDFs (summary + locator, fill-depth plan over orthophoto, cross-section).

Driven by a JSON job file; see README.md for the schema and `python -m surfpatch volreport` for the CLI.
Relative paths in the job file resolve against the job file's directory; environment variables (`${VAR}`) are
expanded in file paths (missing optional paths such as `logo` are skipped quietly).
"""
import datetime
import getpass
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from rxlkit.landxml import read_surface
from rxlkit.roadxml import Corridor

from . import compare, fielddata, io, isopach, tinbuild


def fmt_sta(s, dec=1):
    """Format station as '0+191.0' (dec=1) or '0+050' (dec=0)."""
    a = abs(float(s))
    sign = "-" if s < 0 else ""
    km = int(a // 1000)
    rem = a % 1000
    if dec <= 0:
        return f"{sign}{km}+{rem:03.0f}"
    return f"{sign}{km}+{rem:0{4 + dec}.{dec}f}"


def _resolve_path(val, base_dir):
    """Expand env vars and resolve relative to `base_dir`. Returns None if empty or env var undefined."""
    if not val:
        return None
    s = os.path.expandvars(str(val))
    if "$" in s or "%" in s:
        return None
    p = Path(s)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    return str(p)


def load_job(path_or_dict):
    """Load a volume report job JSON (or copy a dict), attaching `_base` for relative path resolution."""
    if isinstance(path_or_dict, dict):
        job = dict(path_or_dict)
        job.setdefault("_base", os.getcwd())
        return job
    p = Path(path_or_dict).resolve()
    job = json.loads(p.read_text(encoding="utf-8"))
    job["_base"] = str(p.parent)
    return job


def deduction_volume(ded, corridor):
    """Compute `(volume_m3, slope_length_m, stamp_text)` for a deduction config dict."""
    s0, s1 = [float(x) for x in ded["stations"]]
    L = s1 - s0
    dz = float(corridor.profile.elevation(s1) - corridor.profile.elevation(s0)) if corridor.profile else 0.0
    slope = float(np.hypot(L, dz))
    count = ded.get("count", 1)
    area = float(ded["area_m2"])
    vol = float(count) * area * slope
    label = ded.get("label", "DEDUCTION")
    fmt = ded.get("format")
    if fmt:
        stamp = fmt.format(label=label, count=count, area_m2=area, slope=slope,
                           length=L, dz=dz, volume=vol, dv=vol)
    elif count == 1:
        stamp = f"LESS {label} ({area:g} m\u00b2 \u00d7 {slope:.2f} m SLOPE LENGTH) = {vol:.2f} m\u00b3"
    else:
        stamp = f"LESS {label} ({count} \u00d7 {area:g} m\u00b2 \u00d7 {slope:.2f} m SLOPE LENGTH) = {vol:.2f} m\u00b3"
    return vol, slope, stamp


def _load_depression_files(csv_path, dxf_path, top_csv=None, rim_suffix="DEPR TOP", snap=0.002, order="PNEZD"):
    Pd = io.read_shots(csv_path, order=order).points
    polys = io.read_polylines_3d(dxf_path)
    if not polys:
        raise ValueError(f"{dxf_path}: no 3D polylines found")
    rim = None
    suf = rim_suffix.upper()
    for layer, pts in polys:
        if layer.upper().endswith(suf):
            rim = pts
            break
    if rim is None:
        rim = max((pts for _, pts in polys), key=lambda a: io.polygon_area(a[:, [1, 0]]))
    rings = [rim] + [pts for _, pts in polys if pts is not rim]
    Pt_raw = io.read_shots(top_csv, order=order).points if top_csv else None
    (Pb, Fb), (Pt, Ft) = tinbuild.depression(Pd, rings, rim, P_top=Pt_raw, snap=snap)
    return Pb, Fb, Pt, Ft, rim[:, [1, 0]]


def _build_report_surfaces(rep, job, corridor, field):
    """Build `(Pb, Fb, Pt, Ft, boundary_en)` for one report entry."""
    base_dir = job["_base"]
    kind = rep.get("kind", "depression")
    order = (job.get("field") or {}).get("order", "PNEZD")
    snap = float(rep.get("snap", 0.002))

    if kind == "depression":
        stem = rep.get("stem", rep["name"])
        bcsv = _resolve_path(rep.get("bottom_csv", f"{stem}.csv"), base_dir)
        bdxf = _resolve_path(rep.get("boundary_dxf", f"{stem} BOUNDARY.dxf"), base_dir)
        tcsv_cfg = rep.get("top_csv")
        tcsv = _resolve_path(tcsv_cfg, base_dir) if tcsv_cfg else None
        return _load_depression_files(bcsv, bdxf, top_csv=tcsv,
                                      rim_suffix=rep.get("rim_suffix", "DEPR TOP"), snap=snap, order=order)

    if kind == "surf2surf":
        max_off = float(rep.get("max_offset", (job.get("field") or {}).get("max_offset", 5.0)))
        pad = float(rep.get("station_pad", 3.0))
        sta = rep.get("stations")

        def _one(prefix):
            xml_key = f"{prefix}_xml"
            csv_key = f"{prefix}_csv"
            spec_key = f"{prefix}_specs"
            if rep.get(xml_key):
                s = read_surface(_resolve_path(rep[xml_key], base_dir), rep.get(f"{prefix}_surface"))
                return s.points, s.faces
            if rep.get(csv_key):
                P = io.read_shots(_resolve_path(rep[csv_key], base_dir), order=order).points
            elif rep.get(spec_key):
                if field is None:
                    raise ValueError(f"{rep['name']}: '{spec_key}' requires a 'field' section in the job file")
                P = field.shots(rep[spec_key])
            else:
                raise ValueError(f"{rep['name']}: surf2surf report needs {spec_key}, {csv_key}, or {xml_key}")
            if sta is not None:
                s0, s1 = float(sta[0]), float(sta[1])
                s, o = corridor.alignment.station_offset(P[:, 0], P[:, 1])
                P = P[(np.abs(o) < max_off) & (s > s0 - pad) & (s < s1 + pad)]
            if len(P) < 3:
                raise ValueError(f"{rep['name']}: fewer than 3 {prefix} points")
            return P, Delaunay(P[:, [1, 0]]).simplices

        Pb, Fb = _one("bottom")
        Pt, Ft = _one("top")
        if rep.get("boundary_dxf"):
            bnd = io.read_boundary(_resolve_path(rep["boundary_dxf"], base_dir))
        else:
            band = tinbuild.station_band(corridor.alignment, float(sta[0]), float(sta[1]), max_off) if sta else None
            bnd = tinbuild.overlap_boundary([(Pb, Fb), (Pt, Ft)], band=band)
        return Pb, Fb, Pt, Ft, bnd

    raise ValueError(f"{rep['name']}: unknown report kind {kind!r} (expected 'depression' or 'surf2surf')")


def _sample_section(sampler, s, offs, corridor):
    n, e = corridor.alignment.point(s, offs)
    return sampler(e, n)


def section_station(Pb, Fb, Pt, Ft, boundary, s0, s1, corridor, half_width=4.5):
    """Pick a top-surface shot line nearest mid-report whose cut is <= 5% of fill."""
    s, o = corridor.alignment.station_offset(Pt[:, 0], Pt[:, 1])
    cand = sorted({round(float(x), 1) for x in s[(np.abs(o) < 1.0) & (s > s0 + 1) & (s < s1 - 1)]})
    mid = (s0 + s1) / 2.0
    if not cand:
        return round(mid, 1)
    offs = np.linspace(-half_width, half_width, 901)
    sb = compare.Sampler(Pb, Fb)
    st_s = compare.Sampler(Pt, Ft)
    good = []
    for st in cand:
        n, e = corridor.alignment.point(st, offs)
        d = st_s(e, n) - sb(e, n)
        d = d[compare.inside(boundary, e, n) & np.isfinite(d)]
        if len(d) and -d[d < 0].sum() <= 0.05 * d[d > 0].sum():
            good.append(st)
    pick = good or cand
    return min(pick, key=lambda st: abs(st - mid))


def base_profile(base_spec, s, offs, zb, corridor, resolve_report, field=None, base_dir=None, search_stations=8.0):
    """Base-line elevations along the section at `s`, joined to the ends of `zb` (NaN outside `zb`)."""
    if not base_spec or base_spec == "self":
        return zb.copy()
    z = np.full_like(zb, np.nan)
    for src in base_spec:
        if "report" in src:
            P, F = resolve_report(src["report"])[:2]
        elif "depression" in src:
            dep = src["depression"]
            if isinstance(dep, str):
                bcsv = _resolve_path(f"{dep}.csv", base_dir)
                bdxf = _resolve_path(f"{dep} BOUNDARY.dxf", base_dir)
                rim_suf = "DEPR TOP"
            else:
                bcsv = _resolve_path(dep["bottom_csv"], base_dir)
                bdxf = _resolve_path(dep["boundary_dxf"], base_dir)
                rim_suf = dep.get("rim_suffix", "DEPR TOP")
            P, F = _load_depression_files(bcsv, bdxf, top_csv=None, rim_suffix=rim_suf)[:2]
        elif "field" in src:
            if field is None:
                continue
            P = field.shots([tuple(src["field"])])
            if len(P):
                P = P[np.abs(corridor.alignment.station_offset(P[:, 0], P[:, 1])[0] - s) < search_stations]
            if len(P) < 3:
                continue
            F = Delaunay(P[:, [1, 0]]).simplices
        elif "xml" in src:
            surf = read_surface(_resolve_path(src["xml"], base_dir), src.get("surface"))
            P, F = surf.points, surf.faces
        else:
            raise ValueError(f"unsupported base source: {src!r}")
        zz = _sample_section(compare.Sampler(P, F), s, offs, corridor)
        z = np.where(np.isfinite(z), z, zz)

    have = np.isfinite(zb)
    if not have.any():
        return z
    z[have & (z > zb)] = np.nan
    i0, i1 = np.flatnonzero(have)[[0, -1]]
    z[:i0] = np.nan
    z[i1 + 1:] = np.nan
    ok = np.flatnonzero(np.isfinite(z))
    if not len(ok):
        return z
    xs = np.r_[offs[i0], offs[ok], offs[i1]]
    zs = np.r_[zb[i0], z[ok], zb[i1]]
    span = slice(i0, i1 + 1)
    z[span] = np.interp(offs[span], xs, zs)
    return z


def _make_ortho_loader(job, corridor):
    ortho_cfg = job.get("ortho")
    if not ortho_cfg or not (ortho_cfg.get("image") or ortho_cfg.get("cache")):
        return lambda: (None, None)
    base_dir = job["_base"]
    img_path = _resolve_path(ortho_cfg.get("image"), base_dir)
    cache_path = _resolve_path(ortho_cfg.get("cache"), base_dir)
    downsample = int(ortho_cfg.get("downsample", 2))
    margin = float(ortho_cfg.get("margin", 40.0))
    loaded = [None]

    def _get():
        if loaded[0] is not None:
            return loaded[0]
        pts = np.array([corridor.alignment.point(s, o)
                        for s in np.linspace(corridor.alignment.start, corridor.alignment.end, 60)
                        for o in (-margin, margin)])
        bbox_en = (float(pts[:, 1].min()), float(pts[:, 0].min()),
                   float(pts[:, 1].max()), float(pts[:, 0].max()))
        loaded[0] = io.ortho_crop(img_path, bbox_en, cache=cache_path, downsample=downsample)
        return loaded[0]

    return _get


def _draw_locator(fig, rect, boundary, corridor, ortho_loader, step=50.0):
    img, ext = ortho_loader()
    ax = fig.add_axes(rect)
    fg = "white" if img is not None else "0.15"
    if img is not None:
        ax.imshow(img, extent=ext, interpolation="bilinear")
    else:
        ax.set_facecolor("#f2f2f2")
    ss = np.linspace(corridor.alignment.start, corridor.alignment.end, 300)
    cl = np.array([corridor.alignment.point(s, 0.0) for s in ss])
    ax.plot(cl[:, 1], cl[:, 0], "-", color=fg, lw=1.5)
    ax.fill(boundary[:, 0], boundary[:, 1], fc="red", ec="red", alpha=0.6, lw=1.5)
    s_first = np.ceil(corridor.alignment.start / step) * step
    for s in np.arange(s_first, corridor.alignment.end, step):
        n, e = corridor.alignment.point(s, 0.0)
        ax.plot(e, n, "o", color=fg, ms=3)
        ax.annotate(fmt_sta(s, 0), (e, n), xytext=(6, -10), textcoords="offset points",
                    fontsize=7, color=fg, weight="bold")
    ax.set_xlim(cl[:, 1].min() - 25, cl[:, 1].max() + 35)
    ax.set_ylim(cl[:, 0].min() - 15, cl[:, 0].max() + 15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    x0, y0 = ax.get_xlim()[1] - 6, ax.get_ylim()[1] - 4
    ax.annotate("N", (x0, y0), xytext=(x0, y0 - 12), ha="center", color=fg, weight="bold",
                arrowprops=dict(arrowstyle="->", color=fg))


def _plan_page(pdf, Pb, Fb, Pt, Ft, boundary, corridor, ortho_loader, ortho_date=None,
               s_sec=None, plan_step=5.0, half_width=4.5):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    ztop = compare.Sampler(Pt, Ft)(Pb[:, 1], Pb[:, 0])
    thick = np.nan_to_num(ztop - Pb[:, 2], nan=0.0)
    img, ext = ortho_loader()
    ce, cn = float(boundary[:, 0].mean()), float(boundary[:, 1].mean())
    half = max(float(np.ptp(boundary[:, 0])), float(np.ptp(boundary[:, 1]))) / 2.0 + 8.0
    fg = "white" if img is not None else "0.15"

    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.2, 0.84, 0.65])
    if img is not None:
        oe0, oe1, on0, on1 = ext
        ax.imshow(img, extent=(oe0 - ce, oe1 - ce, on0 - cn, on1 - cn), interpolation="bilinear")
    else:
        ax.set_facecolor("#f2f2f2")
    tb = mtri.Triangulation(Pb[:, 1] - ce, Pb[:, 0] - cn, Fb)
    cen = np.c_[Pb[Fb, 1].mean(axis=1) - ce, Pb[Fb, 0].mean(axis=1) - cn]
    tb.set_mask(~compare.inside(np.c_[boundary[:, 0] - ce, boundary[:, 1] - cn], cen[:, 0], cen[:, 1]))
    tc = ax.tripcolor(tb, thick, shading="gouraud", cmap="jet", alpha=0.75, vmin=0)
    ax.triplot(tb, lw=0.3, color="k", alpha=0.4)
    ax.plot(np.r_[boundary[:, 0], boundary[0, 0]] - ce, np.r_[boundary[:, 1], boundary[0, 1]] - cn, "r-", lw=1)

    sa = corridor.alignment.station_offset(boundary[:, 1], boundary[:, 0])[0]
    for s in np.arange(np.ceil(sa.min() / plan_step) * plan_step, sa.max() + 1e-6, plan_step):
        n, e = corridor.alignment.point(s, 0.0)
        ax.plot(e - ce, n - cn, "+", color=fg, ms=7)
        ax.annotate(fmt_sta(s, 1), (e - ce, n - cn), xytext=(8, 4), textcoords="offset points",
                    fontsize=7, color=fg, weight="bold")
    if s_sec is not None:
        n, e = corridor.alignment.point(s_sec, np.array([-half_width, half_width]))
        ax.plot(e - ce, n - cn, "-", color=fg, lw=1.5)
        for k in (0, 1):
            ax.text(e[k] - ce, n[k] - cn, "A", color="white", weight="bold", fontsize=10,
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.2", fc="black", ec="white", lw=0.8))

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([-half + 1, -half + 6], [-half + 1, -half + 1], "-", color=fg, lw=2)
    ax.text(-half + 3.5, -half + 1.4, "5 m", color=fg, ha="center", fontsize=7)
    ax.annotate("N", (half - 1.5, half - 1), xytext=(half - 1.5, half - 4), ha="center",
                color=fg, weight="bold", arrowprops=dict(arrowstyle="->", color=fg))
    cax = fig.add_axes([0.25, 0.14, 0.5, 0.015])
    fig.colorbar(tc, cax=cax, orientation="horizontal", label="Fill depth (m)")
    if ortho_date:
        fig.text(0.08, 0.1, f"Orthophoto {ortho_date}", fontsize=7, color="0.4")
    pdf.savefig(fig)
    plt.close(fig)


def _draw_design(ax, design, s, corridor, zs):
    import matplotlib.pyplot as plt

    if not design or corridor.profile is None:
        return
    zp = float(corridor.profile.elevation(s))
    for item in design:
        if "from" in item and s < float(item["from"]):
            continue
        if "to" in item and s >= float(item["to"]):
            continue
        shape = item.get("shape")
        if shape == "box":
            w = float(item["w"])
            h = float(item["h"])
            top = float(item.get("top", 0.0))
            cx = float(item.get("cx", 0.0))
            ax.add_patch(plt.Rectangle((cx - w / 2.0, zp + top - h), w, h,
                                       fc=item.get("fc", "0.75"), ec=item.get("ec", "0.3"),
                                       lw=float(item.get("lw", 0.8)), label=item.get("label")))
            zs.append(np.array([zp + top - h]))
        elif shape == "circles":
            r = float(item["d"]) / 2.0
            xs_c = [float(v) for v in item["x"]]
            zs_c = [float(v) for v in item["z"]]
            first = True
            for dz in zs_c:
                for x in xs_c:
                    ax.add_patch(plt.Circle((x, zp + dz), r,
                                            fc=item.get("fc", "white"), ec=item.get("ec", "purple"),
                                            lw=float(item.get("lw", 0.8)),
                                            label=item.get("label") if first else None))
                    first = False
            if zs_c:
                zs.append(np.array([zp + min(zs_c) - r]))
        else:
            raise ValueError(f"unknown design shape {shape!r} (expected 'box' or 'circles')")


def _section_page(pdf, s, Pb, Fb, Pt, Ft, boundary, corridor, base_spec, resolve_report,
                  field=None, base_dir=None, design=None, base_label="trench bottom", half_width=4.5):
    import matplotlib.pyplot as plt

    offs = np.linspace(-half_width, half_width, 901)
    n, e = corridor.alignment.point(s, offs)
    in_bnd = compare.inside(boundary, e, n)
    zb = compare.Sampler(Pb, Fb)(e, n)
    zt = compare.Sampler(Pt, Ft)(e, n)
    zb[~in_bnd] = np.nan
    zt[~in_bnd] = np.nan

    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.1, 0.3, 0.82, 0.55])
    zs = [zb, zt]
    ax.fill_between(offs, zb, zt, where=np.isfinite(zb) & np.isfinite(zt) & (zt > zb),
                    color="orange", alpha=0.6, label="this report (fill)", lw=0)
    cut = np.isfinite(zb) & np.isfinite(zt) & (zt < zb - 0.005)
    if cut.any():
        ax.fill_between(offs, zt, zb, where=cut, color="0.6", alpha=0.6, lw=0,
                        label="cut (top below earlier surface)")

    zbase = base_profile(base_spec, s, offs, zb, corridor, resolve_report, field=field, base_dir=base_dir)
    under = np.isfinite(zbase) & np.isfinite(zb) & (zb > zbase + 0.005)
    if under.any():
        ax.fill_between(offs, zbase, zb, where=under, color="0.85", lw=0, label="earlier placements")
    ax.plot(offs, zbase, "-", color="tab:blue", lw=1.6, label=base_label)
    zs.append(zbase)
    ax.plot(offs, zt, "k-", lw=1.2, label="top")
    ax.plot(offs, zb, "k--", lw=1, label="bottom")

    _draw_design(ax, design, s, corridor, zs)

    allz = np.concatenate(zs)
    lo, hi = float(np.nanmin(allz)) - 0.25, float(np.nanmax(allz)) + 0.25
    xo = offs[np.isfinite(zb) | np.isfinite(zt)]
    x0, x1 = float(xo.min()) - 0.5, float(xo.max()) + 0.5
    ax.set_ylim(lo, hi)
    ax.set_xlim(x0, x1)
    ax.set_aspect(1, adjustable="box")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Offset from CL (m), looking up-station")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Section A-A at {fmt_sta(s, 1)}   (true scale)", fontsize=10)
    fig.canvas.draw()
    pos = ax.get_position()
    ax.set_position([pos.x0, 0.62 - pos.height / 2, pos.width, pos.height])
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(pos.x0, 0.62 - pos.height / 2 - 0.05))
    pdf.savefig(fig)
    plt.close(fig)


def add_header_footer(path, logo_path=None, project="", now=None):
    """Stamp vector logo top right and footer (path, timestamp, user, project, page i of N) on every page."""
    import pymupdf

    now = now or datetime.datetime.now()
    doc = pymupdf.open(str(path))
    logo = None
    try:
        if logo_path and os.path.isfile(logo_path):
            raw = pymupdf.open(str(logo_path))
            logo = raw if raw.is_pdf else pymupdf.open("pdf", raw.convert_to_pdf())
            if logo is not raw:
                raw.close()
        stamp = f"Generated {fielddata.nice_date(now.date())} {now:%H:%M} by {getpass.getuser()}"
        if project:
            stamp += f"   |   {project}"
        grey = (0.35, 0.35, 0.35)
        for i, page in enumerate(doc):
            W, H = page.rect.width, page.rect.height
            if logo is not None:
                page.show_pdf_page(pymupdf.Rect(W - 36 - 150, 24, W - 36, 54), logo, 0)
            page.draw_line((36, H - 34), (W - 36, H - 34), color=(0.7, 0.7, 0.7), width=0.5)
            page.insert_text((36, H - 24), str(path), fontsize=6.5, fontname="helv", color=grey)
            page.insert_text((36, H - 15), stamp, fontsize=6.5, fontname="helv", color=grey)
            txt = f"Page {i + 1} of {len(doc)}"
            page.insert_text((W - 36 - pymupdf.get_text_length(txt, "helv", 6.5), H - 15),
                             txt, fontsize=6.5, fontname="helv", color=grey)
        doc.saveIncr()
    finally:
        if logo is not None:
            logo.close()
        doc.close()


def _open_output_path(out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(str(out_dir), f"{name} Volume Report.pdf")
    try:
        open(out, "ab").close()
    except PermissionError:
        out = out[:-4] + " (NEW).pdf"
        print(f"  ! old PDF is open, wrote {os.path.basename(out)} instead")
    return out


def build_report(job, rep, corridor, out_dir, field=None, ortho_loader=None, surface_cache=None, report_map=None):
    """Build one 3-page volume report PDF and return its summary dict."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    name = rep["name"]
    base_dir = job["_base"]
    year = int(job.get("year", datetime.date.today().year))
    surface_cache = {} if surface_cache is None else surface_cache
    report_map = report_map or {r["name"]: r for r in job.get("reports", [])}
    ortho_loader = ortho_loader or _make_ortho_loader(job, corridor)

    def _resolve(rname):
        if rname not in surface_cache:
            if rname not in report_map:
                raise KeyError(f"referenced report {rname!r} not found in job")
            surface_cache[rname] = _build_report_surfaces(report_map[rname], job, corridor, field)
        return surface_cache[rname]

    Pb, Fb, Pt, Ft, boundary = _resolve(name)
    r = isopach.isopach((Pb, Fb), (Pt, Ft), polygon=boundary)

    sa = corridor.alignment.station_offset(boundary[:, 1], boundary[:, 0])[0]
    s_min, s_max = float(sa.min()), float(sa.max())

    lines = [
        ("Stations", f"{fmt_sta(s_min, 1)} to {fmt_sta(s_max, 1)}"),
        ("Bottom surface", rep.get("bottom_desc", "")),
        ("Top surface", rep.get("top_desc", "")),
        ("", ""),
        ("Fill", f"{r.fill:.1f} m\u00b3"),
        ("Cut", f"{r.cut:.1f} m\u00b3"),
        ("Area", f"{r.area_compared:.1f} m\u00b2"),
        ("Max fill depth", f"{r.max_fill:.2f} m"),
    ]

    top_files, bot_files = [], []
    sources = rep.get("sources") or {}
    if field is not None and sources:
        if sources.get("top"):
            top_files = field.files_used(sources["top"], s_min, s_max, corridor)
        if sources.get("bottom"):
            bot_files = field.files_used(sources["bottom"], s_min, s_max, corridor)
    dates = sorted({d for _, d in top_files})
    if dates:
        lines.insert(1, ("Placement surveyed", " and ".join(fielddata.nice_date(d) for d in dates)))
    lines = [(k, fielddata.readable(v, year)) for k, v in lines]

    net, dv, stamp = float(r.fill), 0.0, None
    ded = rep.get("deduction")
    if ded:
        dv, _, stamp = deduction_volume(ded, corridor)
        net -= dv

    out = _open_output_path(out_dir, name)
    rxl_stem = os.path.splitext(os.path.basename(_resolve_path(job["alignment"], base_dir)))[0]
    ortho_date = (job.get("ortho") or {}).get("date", "")
    title = job.get("title", "Volume Report")
    subtitle = job.get("subtitle", "")
    crs_note = job.get("crs_note", "")
    today = datetime.date.today()
    date_line = f"Report date {fielddata.nice_date(today)}" + (f"    {crs_note}" if crs_note else "")
    sub_line = f"{subtitle}    {name}" if subtitle else name

    footer_tpl = job.get("footer_note")
    if footer_tpl:
        footer_note = footer_tpl.format(alignment=rxl_stem, ortho_date=ortho_date)
    else:
        footer_note = f"Stations on alignment {rxl_stem}.  Volumes TIN to TIN (isopach)."
        if ortho_date:
            footer_note += f"  Orthophoto {ortho_date}."

    half_w = float(job.get("section_half_width", 4.5))
    plan_step = float(job.get("plan_step", 5.0))
    loc_step = float(job.get("locator_step", 50.0))
    base_label = rep.get("base_label", job.get("base_label", "trench bottom"))

    with PdfPages(out) as pdf:
        # Page 1: summary + locator
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.93, title, fontsize=15, weight="bold")
        fig.text(0.08, 0.905, sub_line, fontsize=11)
        fig.text(0.08, 0.885, date_line, fontsize=9, color="0.3")
        y = 0.83
        for k, v in lines:
            fig.text(0.08, y, k, fontsize=10, weight="bold")
            fig.text(0.3, y, v, fontsize=10)
            y -= 0.027
        y -= 0.02
        if stamp:
            fig.text(0.08, y, stamp, fontsize=9.5, color="firebrick")
            y -= 0.035
        fig.text(0.08, y, f"NET FILL = {net:.1f} m\u00b3", fontsize=12, weight="bold")
        y -= 0.045
        if top_files or bot_files:
            fig.text(0.08, y, "Field data", fontsize=10, weight="bold")
            y -= 0.022
            for role, files in (("Top (placement)", top_files), ("Bottom", bot_files)):
                for i, (f, d) in enumerate(files):
                    fig.text(0.08, y, role if i == 0 else "", fontsize=8.5, color="0.25")
                    fig.text(0.3, y, fielddata.nice_date(d), fontsize=8.5)
                    fig.text(0.47, y, f, fontsize=8.5, color="0.3")
                    y -= 0.019
        loc_h = max(0.12, y - 0.03 - 0.08)
        _draw_locator(fig, [0.08, 0.08, 0.84, loc_h], boundary, corridor, ortho_loader, step=loc_step)
        fig.text(0.08, 0.06, footer_note, fontsize=7, color="0.4")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: plan of fill depth
        s_sec = section_station(Pb, Fb, Pt, Ft, boundary, s_min, s_max, corridor, half_width=half_w)
        _plan_page(pdf, Pb, Fb, Pt, Ft, boundary, corridor, ortho_loader,
                   ortho_date=ortho_date, s_sec=s_sec, plan_step=plan_step, half_width=half_w)

        # Page 3: cross-section A-A
        _section_page(pdf, s_sec, Pb, Fb, Pt, Ft, boundary, corridor, rep.get("base", "self"),
                      _resolve, field=field, base_dir=base_dir, design=job.get("design"),
                      base_label=base_label, half_width=half_w)

    logo_path = _resolve_path(job.get("logo"), base_dir)
    add_header_footer(out, logo_path=logo_path, project=job.get("project", ""))

    return {
        "name": name,
        "pdf": out,
        "stations": (s_min, s_max),
        "section_station": float(s_sec),
        "fill": float(r.fill),
        "cut": float(r.cut),
        "area_compared": float(r.area_compared),
        "area_boundary": float(r.area_boundary),
        "max_fill": float(r.max_fill),
        "max_cut": float(r.max_cut),
        "deduction": float(dv),
        "deduction_text": stamp,
        "net": float(net),
    }


def run(job_path_or_dict, names=None, out_dir=None):
    """Run all reports (or those named in `names`) from a volume-report job JSON/dict.

    Returns a list of per-report result dicts.
    """
    job = load_job(job_path_or_dict)
    base_dir = job["_base"]
    rxl_path = _resolve_path(job["alignment"], base_dir)
    corridor = Corridor.load(rxl_path)
    year = int(job.get("year", datetime.date.today().year))
    field = fielddata.FieldData(job["field"], year, base_dir=base_dir) if job.get("field") else None

    target_dir = str(out_dir) if out_dir is not None else _resolve_path(job.get("out_dir", "."), base_dir)
    reports = job.get("reports", [])
    report_map = {r["name"]: r for r in reports}
    if names:
        missing = [n for n in names if n not in report_map]
        if missing:
            raise KeyError(f"unknown report(s): {missing}. Available: {list(report_map)}")
        todo = [report_map[n] for n in names]
    else:
        todo = reports

    ortho_loader = _make_ortho_loader(job, corridor)
    surface_cache = {}
    results = []
    for rep in todo:
        res = build_report(job, rep, corridor, target_dir, field=field,
                           ortho_loader=ortho_loader, surface_cache=surface_cache, report_map=report_map)
        results.append(res)
    return results
