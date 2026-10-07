"""Add a joint TESS + sparse-catalogue model (combined_tess_atlas.py: ATLAS, Gaia and the MPC survey photometry
of Pan-STARRS, ZTF, Mt. Lemmon and Catalina) to an object's page, shown by default with a toggle back to the
TESS-only catalogue model (assets/viewer.js; ?model=tess selects TESS only). The toggle reads "TESS + catalog";
the solution block lists the sources and point counts that went into the fit (--fit, the fit's JSON).

Input per object: a combined_tess_atlas.py fit -- its convexinv shape, parameters, model lightcurve
and -e uncertainties (prefix PREFIX in the work directory) -- and the object's TESS lightcurve.
Writes
  data/shapes/<key>_tessatlas.json        the mesh (minkowski_py, DAMIT minkowski as fallback), same format as the catalogue
  data/lightcurves/<key>_tessatlas.json   TESS folded on the fit's own period and rotation
                                          zero-point, binned like batch_shapes.py, with the fit's
                                          model rescaled per session to its data
and patches asteroid/<key>.html: window.AST.alt (shape, lightcurve, camera for the fitted pole, facet
arrays) and a "TESS + catalog solution" stats block beside the catalogue one.

The camera reproduces make_cameras.py's derivation (same frame and signed aspect) for the fitted
pole; the fold uses phi(t) = phi0 + 2 pi (t_lt - t0) / P with light-time-corrected epochs, the
convention the viewer's spin sync was validated against (validate_viewer_sync.py).

Usage: python shape/build_tess_atlas_variant.py <key> <work_dir> <clean_lc_csv> [--prefix opt_plain]
       [--label "..."] [--fit <combined fit JSON>]
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import batch_shapes as bs  # noqa: E402
from build_pages import stat, POLE_LABEL  # noqa: E402

NB = bs.NB
COORD_DP = 4
# MPC station codes of the survey photometry blocks ('<station>_<band>' sources in the fit)
STATION_NAMES = {"F51": "Pan-STARRS", "F52": "Pan-STARRS", "I41": "ZTF", "G96": "Mt. Lemmon", "703": "Catalina"}


def data_line(n_tess, fit_json):
    """"TESS n + ATLAS n + Gaia n + ZTF n + ..." from the fit's ATLAS info (points kept after clipping);
    surveys summed over their bands and stations."""
    if not fit_json:
        return f"TESS {n_tess:,} pts + sparse ATLAS / Gaia"
    info = json.load(open(fit_json))["atlas"]
    parts = [f"TESS {n_tess:,}", f"ATLAS {info['n_kept']:,}"]
    if info.get("n_kept_gaia"):
        parts.append(f"Gaia {info['n_kept_gaia']:,}")
    surveys = {}
    for src, v in (info.get("survey") or {}).items():
        name = STATION_NAMES.get(src.split("_")[0], src.split("_")[0])
        surveys[name] = surveys.get(name, 0) + int(v["n_kept"])
    parts += [f"{k} {n:,}" for k, n in surveys.items()]
    return " + ".join(parts) + " pts"


def mat2quat(R):
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    return (q / np.linalg.norm(q)).tolist()


def camera(sv, ev, lam_deg, bet_deg):
    """make_cameras.camera (that module runs its batch at import, so it is reproduced here)."""
    en = ev / np.linalg.norm(ev, axis=1, keepdims=True)
    snv = sv / np.linalg.norm(sv, axis=1, keepdims=True)
    l, b = np.radians(lam_deg), np.radians(bet_deg)
    Rz = np.array([[np.cos(l), np.sin(l), 0], [-np.sin(l), np.cos(l), 0], [0, 0, 1]])
    a = np.pi / 2 - b
    Ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    Rpf = Ry @ Rz
    epf = (Rpf @ en.T).T.mean(axis=0); epf /= np.linalg.norm(epf)
    spf = (Rpf @ snv.T).T.mean(axis=0); spf /= np.linalg.norm(spf)
    aspect = float(np.degrees(np.arccos(np.clip(epf[2], -1, 1))))
    zc = epf
    up = np.array([0.0, 0.0, 1.0]) - np.dot(zc, [0, 0, 1.0]) * zc
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 1.0, 0.0]) - np.dot(zc, [0, 1.0, 0]) * zc
    yc = up / np.linalg.norm(up)
    xc = np.cross(yc, zc)
    Rcam = np.vstack([xc, yc, zc])
    return mat2quat(Rcam), (Rcam @ spf).tolist(), aspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key"); ap.add_argument("work"); ap.add_argument("lc_csv")
    ap.add_argument("--prefix", default="opt_plain")
    ap.add_argument("--label", default="TESS + catalog")
    ap.add_argument("--fit", default=None, help="the combined fit's JSON, for the sources and point counts")
    a = ap.parse_args()
    w = f"{a.work}/{a.prefix}"
    tok = open(f"{w}_p.txt").read().split()
    lam, bet, per_hr, t0, phi0 = float(tok[0]), float(tok[1]), float(tok[2]), float(tok[3]), np.radians(float(tok[4]))
    err = {}
    for line in open(f"{w}_err.txt"):
        t = line.split()
        err[t[0]] = [float(x) for x in t[1:]]
    lam, bet, per_hr = err["lambda_deg"][0], err["beta_deg"][0], err["period_hr"][0]   # full precision

    # mesh
    V, F = bs.minkowski_py_mesh(f"{w}_s.txt") or bs.minkowski(f"{w}_s.txt")
    # Calibrated ATLAS data fix convexinv's absolute size scale, so these meshes come out ~0.03
    # units across against ~1 for the relative-only catalogue fits; rounded to COORD_DP decimals that
    # merged vertices (508 of Elektra's 540 left distinct). Normalise to unit maximum radius like the
    # catalogue meshes -- the viewer and the km scale bar depend only on relative size.
    V = V / np.linalg.norm(V, axis=1).max()
    wmax, wmin = bs.caliper_widths(V)
    mesh = {"v": [round(float(c), COORD_DP) for p in V for c in p], "f": [int(i) for t in F for i in t],
            "fn": [len(t) for t in F]}

    # TESS fold on the fit's period and zero-point; the fit's TESS model = the first rows of its model
    # output, one per point of each session with >= 5 points (combined_tess_atlas.write_lcs order)
    df = bs.prepare(pd.read_csv(a.lc_csv))
    sv, ev = bs.build_geometry(df)
    blocks = [idx for idx in bs.session_blocks(df) if len(idx) >= 5]
    order = np.concatenate(blocks)
    jd = bs.lt_jd(df, ev)[order]
    fl = df["rel_flux"].values[order]
    mod = np.loadtxt(f"{w}_f.txt").astype(float)[:len(order)]
    i0 = 0
    for idx in blocks:                                   # rescale EACH session to its own data
        sl = slice(i0, i0 + len(idx)); i0 += len(idx)
        if np.isfinite(mod[sl].mean()) and mod[sl].mean() != 0:
            mod[sl] *= fl[sl].mean() / mod[sl].mean()
    ph = ((phi0 + 2 * np.pi * (jd - t0) / (per_hr / 24.0)) / (2 * np.pi)) % 1.0
    b = np.clip((ph * NB).astype(int), 0, NB - 1)
    cnt = np.bincount(b, minlength=NB)
    mean = np.bincount(b, weights=fl, minlength=NB) / np.maximum(cnt, 1)
    var = np.bincount(b, weights=(fl - mean[b]) ** 2, minlength=NB)
    with np.errstate(divide="ignore", invalid="ignore"):
        sem = np.sqrt(var / np.maximum(cnt - 1, 1)) / np.sqrt(np.maximum(cnt, 1))
    ok = cnt >= 3
    ms = np.bincount(b, weights=mod, minlength=NB) / np.maximum(cnt, 1)
    dat = mean[ok]
    lc = dict(phase=[round(float(x), 5) for x in ((np.arange(NB) + .5) / NB)[ok]],
              flux=[round(float(x), 5) for x in dat], err=[round(float(x), 6) for x in np.nan_to_num(sem[ok])],
              model_flux=[round(float(x), 6) for x in ms[ok]])
    amp = float(-2.5 * np.log10(dat.min() / dat.max()))

    q, sun_cam, aspect = camera(sv[order], ev[order], lam, bet)
    json.dump(mesh, open(f"{ROOT}/data/shapes/{a.key}_tessatlas.json", "w"), separators=(",", ":"))
    json.dump(lc, open(f"{ROOT}/data/lightcurves/{a.key}_tessatlas.json", "w"), separators=(",", ":"))
    # the fitted parameters, enough to place the model at any epoch:
    # phi(t) = phi0 + 2 pi (t_lt - t0) / P, t_lt the light-time-corrected JD, about the body z axis
    params = dict(model=a.label, pole_lambda_deg=lam, pole_lambda_err_deg=err["lambda_deg"][1],
                  pole_beta_deg=bet, pole_beta_err_deg=err["beta_deg"][1], frame="ecliptic J2000",
                  period_hr=per_hr, period_err_hr=err["period_hr"][1], rotation_t0_jd=t0,
                  rotation_phi0_rad=float(phi0),
                  phase_function={"a": err["phase_par1"][0], "d": err["phase_par2"][0], "k": err["phase_par3"][0]},
                  equatorial_ratio=float(wmax / wmin), polar_ratio=float(np.ptp(V[:, 2]) / wmin),
                  shape=f"{a.key}_tessatlas.json", n_tess_points=int(len(order)))
    json.dump(params, open(f"{ROOT}/data/shapes/{a.key}_tessatlas_params.json", "w"), indent=1)

    # page
    path = f"{ROOT}/asteroid/{a.key}.html"
    s = open(path).read()
    m = re.search(r"<script>window\.AST=(\{.*?\});</script>", s, re.S)
    ast = json.loads(m.group(1))
    nf = len(mesh["fn"])
    # content hash in the URL: a rebuilt model must not be served from a browser's cache
    hv = lambda p: hashlib.sha1(open(p, "rb").read()).hexdigest()[:8]
    ast["alt"] = {"label": a.label,
                  "shape": f"../data/shapes/{a.key}_tessatlas.json?v={hv(f'{ROOT}/data/shapes/{a.key}_tessatlas.json')}",
                  "lc": f"../data/lightcurves/{a.key}_tessatlas.json?v={hv(f'{ROOT}/data/lightcurves/{a.key}_tessatlas.json')}",
                  "D": {"camQ": [round(x, 6) for x in q], "sunCam": [round(x, 6) for x in sun_cam],
                        "aspect": round(aspect, 2), "alb": [1.0] * nf, "weak": [False] * nf}}
    s = s[:m.start(1)] + json.dumps(ast, separators=(",", ":")) + s[m.end(1):]
    pm = lambda v, e, f: f"{v:{f}} &plusmn; {e:.2g}"
    rows = "".join([
        stat("Rotation period", f"{pm(per_hr, err['period_hr'][1], '.6f')} hours"),
        stat(POLE_LABEL, f'<span style="white-space:nowrap">{lam:.1f} &plusmn; {err["lambda_deg"][1]:.1f}&deg;, '
                         f'{bet:+.1f} &plusmn; {err["beta_deg"][1]:.1f}&deg;</span>'),
        stat("Amplitude", f"{amp:.3f} mag"),
        stat("Equatorial ratio", f"{wmax / wmin:.2f}"),
        stat("Polar / equatorial", f"{np.ptp(V[:, 2]) / wmin:.2f}"),
        stat("Data", data_line(len(order), a.fit)),
        stat("Facets", nf),
        stat("Viewing aspect", f"{aspect:.0f}&deg; from the pole")])
    downloads = (f'<p class="sec">{a.label} downloads</p><div class="dl">'
                 f'<a href="../data/shapes/{a.key}_tessatlas.json" download>Shape model (JSON mesh)</a>'
                 f'<a href="../data/shapes/{a.key}_tessatlas_params.json" download>Pole, period and phase (JSON)</a>'
                 f'<a href="../data/lightcurves/{a.key}_tessatlas.json" download>Folded lightcurve (JSON)</a></div>')
    block = (f'<div data-model="atlas">\n  <p class="sec">{a.label} solution</p>\n  {rows}\n  {downloads}\n</div>\n  ')
    sol_start = s.index('<p class="sec">TESSELLATE solution</p>')
    sol_end = s.index('\n  <p class="sec">Downloads</p>')
    if 'data-model="atlas"' in s:                         # rebuilt: replace the previous blocks
        a0 = s.index('<div data-model="atlas">'); a1 = s.index('<div data-model="tess"', a0)
        s = s[:a0] + block + s[a1:]
    else:
        s = (s[:sol_start] + block + '<div data-model="tess" hidden>\n  ' + s[sol_start:sol_end] + "\n</div>" + s[sol_end:])
    open(path, "w").write(s)
    print(f"{a.key}: pole ({lam:.1f} +- {err['lambda_deg'][1]:.1f}, {bet:+.1f} +- {err['beta_deg'][1]:.1f}), "
          f"P {per_hr:.6f} +- {err['period_hr'][1]:.1e} h, {nf} facets, aspect {aspect:.0f} deg, amplitude {amp:.3f} mag")


if __name__ == "__main__":
    main()
