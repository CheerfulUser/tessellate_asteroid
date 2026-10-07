"""Prototype: convex inversion of dense TESS lightcurves + sparse ATLAS photometry with a FREE pole.

The catalogue pipeline (batch_shapes.py) fits single-apparition TESS data with the pole fixed
perpendicular to the line of sight; that leaves the pole's azimuth essentially undetermined (folded
rms differs by a median factor of only 1.27 across the 12 candidate orientations) and, against
occultation chords, an arbitrary sky orientation. ATLAS photometry (SSCAT V3, 2016 - mid-2025, a
median ~1,900 points per catalogue object over several apparitions) adds the viewing-geometry
diversity that constrains the pole.

ATLAS preparation follows Durech et al. 2020 (A&A 643, A59; arXiv:2010.01820):
  - brightness reduced to 1 au from Sun and observer, treated as calibrated data;
  - one linear-exponential phase curve shared by both filters, the cyan-orange colour fitted as
    a magnitude offset, cyan shifted onto orange;
  - points more than 2.5 rms from that fit rejected.
TESS points are used unbinned: averaging into time bins smears the rotation (a 15-min bin loses ~8%
of a 2.2-h rotator's amplitude, ~36% at 1 h), and these sectors' ~10-min full-frame cadence means
binning hardly reduces the point count anyway. If TESS is found to swamp ATLAS (convexinv's input has
no per-point weights), thin each session by keeping every k-th point, which removes points without
averaging away rotational signal. Phase-function parameters a, d, k are free
(calibrated data present); Lambert c fixed at 0.1 as in the catalogue. The pole and period are fitted
from 10 starting poles spread over the sphere; the best chi2 wins.

Two modes:
  joint      one convexinv fit of TESS + ATLAS together, pole and period free, from 10 starting poles.
             convexinv cannot weight points, so thousands of TESS points can outvote ATLAS on the pole.
  iterative  TESS fixes the shape, ATLAS picks the pole. Shape and pole are coupled (a shape fitted at
             a wrong pole absorbs the wrong geometry), so the pole is never moved under a fixed shape:
             at every trial pole convexinv fits the shape and period to TESS alone with the pole held
             fixed, then that shape is forward-modelled against ATLAS (bright.c's scattering law) over
             a dense sidereal-period scan, with the rotation phase anchored at the TESS epoch. ATLAS rms
             ranks the poles. A 60-pole Fibonacci grid is refined twice (10 deg, then 4 deg) around
             the best pole of each of the 3 best separate sky regions (seeds > 45 deg apart), and a
             final joint fit is started from the ATLAS-picked pole and period; both poles are reported. Every forward model is checked against convexinv's own
             TESS model before its ATLAS score is used.

Run on ozstar (lightcurves + SSCAT parquet there):
    python combined_tess_atlas.py "(130) Elektra" <period_hr> [--mode joint|iterative] [--out DIR]
"""
import argparse
import json
import multiprocessing as mp
import os

# Forked pool workers on ozstar came up bound to ONE core (conda numpy's MKL / Intel OpenMP runtime
# pins each forked process): 8 workers ran at 1.0x, all on core 24. Measured on a compute node
# (kmp_test, 8 workers): MKL_THREADING_LAYER=SEQUENTIAL 8.1x, KMP_AFFINITY=none 7.6x, while
# KMP_AFFINITY=disabled makes the runtime assert (kmp_affinity.cpp 4314) and the workers hang.
# Workers are single-threaded, so take MKL off OpenMP altogether, with 'none' as a backstop.
# pole_step / joint_step still reset each worker's affinity to the parent's set.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_AFFINITY", "none")
os.environ.setdefault("OMP_PROC_BIND", "false")
import re
import subprocess
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import batch_shapes as bs  # noqa: E402

SSCAT = os.environ.get("SSCAT_DIR", "/fred/oz335/TESSdata/mpc/atlas")
SURVEY_DIR = os.environ.get("SURVEY_DIR", "/fred/oz335/TESSdata/mpc/mpc_obs")   # MPC survey photometry + geometry
SURVEY_MIN_PTS = 10                               # a (station, band) block needs this many points to get its own offset
GAIA_DIR = os.environ.get("GAIA_DIR", "/fred/oz335/TESSdata/mpc/gaia")   # Gaia DR3 transits + geometry (gaia_rows);
                                                                          # objects without them are fitted without Gaia
LCDIR = os.environ.get("LCDIR", "/fred/oz335/rridden/asteroids/clean_lightcurves")
B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
START_POLES = [(0, 60), (90, 60), (180, 60), (270, 60), (45, 0), (135, 0), (225, 0), (315, 0),
               (60, -60), (240, -60)]
CLIP_RMS = 2.5
FIB_N = int(os.environ.get("ITER_FIB_N", 60))   # coarse pole grid, ~26 deg spacing
REFINE = [(3, 10.0), (3, 4.0)]                    # (regions refined, neighbour offset deg) per level
REGION_SEP_DEG = 45.0                             # seeds closer than this are the same solution region
N_PHI = 360                                       # rotation-phase samples of the ATLAS brightness table
C_LAMBERT = 0.1                                   # Lambert weight, as write_control fixes it (L-S part 1)
TINY = 1e-8                                       # convexinv's visibility threshold on mu, mu0
N_PERIOD_STARTS = int(os.environ.get("ITER_PERIOD_STARTS", 5))   # weighted fits per pole, from scan minima
FINAL_GRID_ALIASES = 3                            # final dense period grid: +/- this many alias steps
# convexinv stop condition: < 1 iterates until the rms improves by less than this (CheerfulUser/DAMIT-convex fork,
# with the stalled-damping exit). 50 fixed iterations left Loreley's fit 4% short of its minimum,
# as large as the chi2 differences that decide between mirror poles; 1e-7 converges (287 iterations)
STOP_COND = os.environ.get("CI_STOP_COND", "1e-7")
# Pole ranking (TESS-only shapes and weighted fixed-pole fits over the Fibonacci grid and its refinements)
# can stop earlier: only the best poles' fits are carried on, and the period grid, final and mirror fits
# converge at STOP_COND. Profile (Elektra, 16 workers): warm-started ranking fits were 56% of 2.1 CPU-h.
RANK_STOP = os.environ.get("ITER_RANK_STOP", STOP_COND)
# weighted fits per pole on the coarse grid (the refinement levels use N_PERIOD_STARTS)
COARSE_STARTS = int(os.environ.get("ITER_COARSE_STARTS", N_PERIOD_STARTS))
# after ranking, the best CONVERGE_TOP poles (by ranking score, across all regions) are refitted from every
# period start at STOP_COND before the period grid, so a loose or single-start ranking only has to keep the
# right pole in the top few, not rank it first (0 = off). Loreley under RANK_STOP 1e-6 + 1 coarse start fell
# into the 7.22474 h minimum (chi2 0.00070 vs 0.00058).
CONVERGE_TOP = int(os.environ.get("ITER_CONVERGE_TOP", 0))
MODEL_CHECK_TOL = 1e-4                            # max |ours - convexinv| on block-normalised TESS model


def pack(desig):
    m = re.match(r"^\((\d+)\)", desig)
    if m:
        n = int(m.group(1))
        if n < 100000:
            return f"{n:05d}"
        if n < 620000:
            return B62[n // 10000] + f"{n % 10000:04d}"
        n -= 620000
        s = ""
        for _ in range(4):
            s = B62[n % 62] + s
            n //= 62
        return "~" + s
    m = re.match(r"^(\d{2})(\d{2}) ([A-Z])([A-Z])(\d*)$", desig)
    cent = {"18": "I", "19": "J", "20": "K"}[m.group(1)]
    num = int(m.group(5) or 0)
    return f"{cent}{m.group(2)}{m.group(3)}{B62[num // 10]}{num % 10}{m.group(4)}"


def atlas_rows(packed):
    b = zlib.crc32(packed.encode()) % 256
    return pd.read_parquet(f"{SSCAT}/sscat_v3/bucket_{b:03d}.parquet", filters=[("kast", "==", packed)])


def phase_model(alpha_deg, H, d, k, A0):
    # linear-exponential phase curve in magnitudes (Durech et al. 2020, eq. 2, as magnitudes); the
    # floor keeps trial parameters that drive the flux factor to <= 0 finite instead of NaN
    return H - 2.5 * np.log10(np.maximum(A0 * np.exp(-alpha_deg / d) + k * alpha_deg + 1, 1e-6))


PHASE_STARTS = [(d, k) for d in (1.0, 3.0, 6.0, 12.0, 25.0) for k in (-0.02, 0.0)]


def prepare_atlas(df):
    """Reduced magnitudes, shared phase curve with magnitude offsets for ATLAS cyan and Gaia G
    (both shifted onto ATLAS orange), 2.5-rms clipping within each source.

    The fit is nonlinear and multi-modal in d: from one start, the same 2994 Loreley rows gave
    rms 0.128 mag (d 15.8 deg) on ozstar and 0.089 mag (d 3.8 deg) locally, with different scipy.
    Every clipping pass therefore fits from 10 (d, k) starts and keeps the lowest cost.
    df carries src ('atlas' / 'gaia' / a survey '<station>_<band>'), filt ('c', 'o', 'G', or the survey key),
    m, R, delta, SOE, dm (per-point error). Every filter other than ATLAS orange gets a free magnitude offset."""
    from scipy.optimize import least_squares
    df = df[np.isfinite(df.m) & np.isfinite(df.delta) & np.isfinite(df.R)].copy()
    df["m_red"] = df.m - 5 * np.log10(df.R * df.delta)
    keep = np.ones(len(df), bool)
    alpha, mr = df.SOE.values, df.m_red.values
    groups = ["c", "G"] + sorted(set(df.filt) - {"o", "c", "G"})     # offsets onto ATLAS orange
    ind = np.column_stack([df.filt.values == g_ for g_ in groups]).astype(float)
    srcs = [s_ for s_ in ("atlas", "gaia") if (df.src == s_).any()] + sorted(set(df.src) - {"atlas", "gaia"})

    def resid(p, sel):
        H, d, k, A0 = p[:4]
        return mr[sel] - (phase_model(alpha[sel], H, d, k, A0) + ind[sel] @ p[4:])

    lo = [-10, 0.1, -0.2, 0, -2] + [-3] * (len(groups) - 1)
    hi = [30, 50, 0.2, 5, 2] + [3] * (len(groups) - 1)
    rms_src = {}
    for _ in range(3):
        fits = [least_squares(resid, [np.median(mr[keep]), d0, k0, 0.5, 0.3] + [0.0] * (len(groups) - 1), args=(keep,),
                              bounds=(lo, hi)) for d0, k0 in PHASE_STARTS]
        p = min(fits, key=lambda f: f.cost).x
        r = resid(p, np.ones(len(df), bool))
        new = np.zeros(len(df), bool)
        for s_ in srcs:                                   # each source clipped at its own scatter
            sel = (df.src.values == s_)
            rms_src[s_] = float(np.sqrt(np.mean(r[keep & sel] ** 2)))
            new |= sel & (np.abs(r) < CLIP_RMS * rms_src[s_])
        keep = new
    n_raw = {s_: int((df.src == s_).sum()) for s_ in srcs}
    df = df[keep].copy()
    df["m_o"] = df.m_red - ind[keep] @ p[4:]                                    # all onto ATLAS orange
    info = dict(colour_c_minus_o=float(p[4]), n_raw=n_raw["atlas"], n_kept=int((df.src == "atlas").sum()),
                phase_fit_rms=rms_src["atlas"], phase_d_deg=float(p[1]), phase_k=float(p[2]), phase_A0=float(p[3]))
    if "gaia" in srcs:
        info.update(colour_G_minus_o=float(p[5]), n_raw_gaia=n_raw["gaia"],
                    n_kept_gaia=int((df.src == "gaia").sum()), phase_fit_rms_gaia=rms_src["gaia"])
    off = dict(zip(groups, p[4:]))
    info["phase_fit_rms_by_src"] = rms_src
    info["survey"] = {s_: dict(colour_minus_o=float(off[s_]), n_raw=n_raw[s_], n_kept=int((df.src == s_).sum()),
                               phase_fit_rms=rms_src[s_]) for s_ in srcs if s_ not in ("atlas", "gaia")}
    return df, info


def atlas_geometry(df):
    """Asteroid-centric ecliptic Sun and observer vectors from the measured RA/Dec, delta, R.
    Observer = Earth's centre (ATLAS sites are on Earth; < 1e-4 au offset)."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    t = Time(df.MJD_obs.values, format="mjd", scale="utc")
    ra, dec = np.radians(df.ra.values), np.radians(df.dec.values)
    u = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    ep = get_body_barycentric("earth", t).xyz.to("au").value.T
    sp = get_body_barycentric("sun", t).xyz.to("au").value.T
    ap = ep + df.delta.values[:, None] * u
    return (sp - ap) @ bs._ICRS_TO_ECL.T, (ep - ap) @ bs._ICRS_TO_ECL.T


def with_geometry(df, sv, ev):
    df = df.copy()
    df[["sx", "sy", "sz"]], df[["ex", "ey", "ez"]] = sv, ev
    return df


def gaia_rows(desig):
    """Gaia DR3 transits of a numbered asteroid with their viewing geometry, from GAIA_DIR:
      gaia_dr3_sso/bucket_XXX.parquet       one row per transit (convert_gaia_sso_to_parquet.py: the
                                            8-9 CCD rows of a transit share one G magnitude)
      gaia_dr3_geometry/bucket_XXX.parquet  asteroid-centric ecliptic Sun and Gaia vectors, R, delta,
                                            phase angle, light-time-corrected epoch (gaia_transit_geometry.py:
                                            ASSIST orbits, within 300 km / 0.2 arcsec of JPL Horizons)
    bucket = number % 256. Returns None when the object has no Gaia data."""
    m = re.match(r"^\((\d+)\)", desig)
    if GAIA_DIR is None or m is None:
        return None
    n = int(m.group(1))
    pt = f"{GAIA_DIR}/gaia_dr3_sso/bucket_{n % 256:03d}.parquet"
    pg = f"{GAIA_DIR}/gaia_dr3_geometry/bucket_{n % 256:03d}.parquet"
    if not (os.path.exists(pt) and os.path.exists(pg)):
        return None
    tr = pd.read_parquet(pt, filters=[("number_mp", "==", n)])
    geo = pd.read_parquet(pg, filters=[("number_mp", "==", n)])
    d = tr.merge(geo, on=["number_mp", "transit_id"])
    d = d[np.isfinite(d.g_mag) & (d.g_flux > 0) & (d.g_flux_error > 0)].reset_index(drop=True)
    if not len(d):
        return None
    df = pd.DataFrame(dict(src="gaia", filt="G", m=d.g_mag.values,
                           dm=2.5 / np.log(10) * d.g_flux_error.values / d.g_flux.values,
                           R=d.R.values, delta=d.delta.values, SOE=d.SOE.values, MJD_lc=d.MJD_lc.values))
    return with_geometry(df, d[["sx", "sy", "sz"]].values, d[["ex", "ey", "ez"]].values)


def survey_rows(desig):
    """MPC survey photometry of a numbered asteroid with its viewing geometry, from
    SURVEY_DIR/survey_phot_geometry/bucket_XXX.parquet (survey_phot_geometry.py; bucket = number % 256).
    Each (station, band) with at least SURVEY_MIN_PTS points is its own source '<station>_<band>', with its own
    offset, clipping and error floor. dm is the reported per-point error when the file has one, else 0 (the
    block's error floor alone weights it). Returns None when the object has none."""
    m = re.match(r"^\((\d+)\)", desig)
    if SURVEY_DIR is None or m is None:
        return None
    n = int(m.group(1))
    path = f"{SURVEY_DIR}/survey_phot_geometry/bucket_{n % 256:03d}.parquet"
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path, filters=[("number", "==", n)])
    d = d[np.isfinite(d.mag)].reset_index(drop=True)
    key = (d.stn + "_" + d.band).values
    counts = pd.Series(key).value_counts()
    d = d[pd.Series(key).isin(counts[counts >= SURVEY_MIN_PTS].index).values].reset_index(drop=True)
    if not len(d):
        return None
    key = (d.stn + "_" + d.band).values
    dm = d.dm.fillna(0.0).values if "dm" in d else np.zeros(len(d))
    df = pd.DataFrame(dict(src=key, filt=key, m=d.mag.values, dm=dm, R=d.R.values, delta=d.delta.values,
                           SOE=d.SOE.values, MJD_lc=d.MJD_lc.values))
    return with_geometry(df, d[["sx", "sy", "sz"]].values, d[["ex", "ey", "ez"]].values)


def tess_points(df, sv, ev):
    """Every TESS point, light-time corrected (convexinv requires it), tagged with its session,
    with its relative photometric error e_flux / flux (NaN when the lightcurve has none)."""
    t_lt = bs.lt_jd(df, ev) - 2400000.5
    rel_err = (df.e_flux / df.flux).abs().values if "e_flux" in df else np.full(len(df), np.nan)
    return [(int(k), t_lt[i], df.rel_flux.values[i], sv[i], ev[i], rel_err[i])
            for k, idx in enumerate(bs.session_blocks(df)) for i in idx]


def write_lcs(path, tess_rows, n_sessions, atlas, sv_a, ev_a, w_tess=None, w_sparse=None):
    """Weighted format of the fork's convexinv (CheerfulUser/DAMIT-convex): w_tess, one weight per
    TESS row, and w_sparse, one per ATLAS/Gaia row, are written as a ninth column flagged by a
    fourth header number. None writes the plain format. Sparse rows are split into blocks by source
    (ATLAS, Gaia), each calibrated and reduced to 1 au."""
    blocks = []
    wt = None if w_tess is None else np.asarray(w_tess, float)
    for k in range(n_sessions):
        pts = [(r[1], r[2], r[3], r[4], None if wt is None else wt[j]) for j, r in enumerate(tess_rows) if r[0] == k]
        if len(pts) >= 5:
            blocks.append((0, pts))                          # relative
    n_tess_blocks, n_tess_written = len(blocks), sum(len(p) for _, p in blocks)
    is_tess = [True] * n_tess_blocks       # a fifth header number tags the source (0 TESS, 1 sparse);
                                           # convexinv reads only the first four
    flux = 10 ** (-0.4 * atlas.m_o.values)
    wsp = np.ones(len(atlas)) if w_sparse is None else np.asarray(w_sparse)
    src = atlas.src.values if "src" in atlas else np.array(["atlas"] * len(atlas))
    filt = atlas.filt.values if "filt" in atlas else np.array(["o"] * len(atlas))
    # ATLAS orange is the calibrated reference; ATLAS cyan and Gaia G are each one RELATIVE block,
    # i.e. a free overall scale = a fitted colour offset, while their phase-angle and rotational
    # variation still constrain the fit. One block per source, never chunked (the fork's
    # convexinv has no per-lightcurve point limit, and a chunk would add a spurious free offset).
    surveys = sorted(set(src) - {"atlas", "gaia"})
    for s_, f_, flag in [("atlas", "o", 1), ("atlas", "c", 0), ("gaia", "G", 0)] + [(k_, k_, 0) for k_ in surveys]:
        idx = np.where((src == s_) & (filt == f_))[0]
        if len(idx) < 5:
            continue
        pts = [(atlas.MJD_lc.values[i], flux[i], sv_a[i], ev_a[i], wsp[i] if w_sparse is not None else None)
               for i in idx]                                 # MJD_lc: light-time corrected
        blocks.append((flag, pts))
        is_tess.append(False)
    with open(path, "w") as f:
        f.write(f"{len(blocks)}\n")
        for (flag, pts), tess_blk in zip(blocks, is_tess):
            per_point = pts[0][4] is not None
            f.write(f"{len(pts)} {flag} 1 1 {0 if tess_blk else 1}\n" if per_point else f"{len(pts)} {flag}\n")
            for jd_mjd, fl, s, e, w in pts:
                f.write(f"{jd_mjd + 2400000.5:.6f} {fl:.6e} {s[0]:.6f} {s[1]:.6f} {s[2]:.6f} "
                        f"{e[0]:.6f} {e[1]:.6f} {e[2]:.6f}" + (f" {w:.8g}" if per_point else "") + "\n")
    return n_tess_written, sum(len(p) for p in [b[1] for b in blocks][n_tess_blocks:])


def run_ci(cp, lcs, sp, pp, fp, extra=()):
    """bs.run_pole with extra convexinv options (CheerfulUser/DAMIT-convex fork): -c/-i coefficient output/warm
    start, -e uncertainties, -y YORP, -r robust reweighting. Returns ((lambda, beta, period), None)
    or (None, reason)."""
    try:
        with open(lcs) as fin:
            r = subprocess.run([bs.CONVEXINV, *map(str, extra), "-o", sp, "-p", pp, cp, fp], stdin=fin,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=bs.T_INV)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {bs.T_INV}s"
    if r.returncode != 0:
        return None, (r.stderr.decode(errors="replace").strip().splitlines() or ["nonzero exit"])[-1]
    if not os.path.exists(pp) or not open(pp).read().strip():
        return None, "no params file written"
    return tuple(map(float, open(pp).read().split()[:3])), None


def read_errors(path):
    """convexinv -e output: {name: [value, sigma]} (counts as [value])."""
    out = {}
    for line in open(path):
        t = line.split()
        if len(t) >= 2:
            out[t[0]] = [float(x) for x in t[1:]]
    return out


def write_control(path, lam, bet, per, pole_free=True, phase_free=True, stop=None):
    """phase_free only with calibrated data present; relative-only fits keep a, d, k fixed."""
    fp, fa = int(pole_free), int(phase_free)
    with open(path, "w") as f:
        f.write(f"{lam}\t\t{fp}\tinital lambda\n{bet}\t\t{fp}\tinitial beta\n{per}\t\t1\tinital period\n")
        f.write(f"0\t\t\tzero time\n0\t\t\tinitial rotation angle\n{bs.CONVEXITY_W}\t\t\tconvexity regularization\n")
        f.write(f"{bs.HARM} {bs.HARM}\t\t\tdegree and order\n{bs.NROWS}\t\t\tnumber of rows\n")
        for line in [f"0.5\t\t{fa}\ta", f"0.1\t\t{fa}\td", f"-0.5\t\t{fa}\tk", "0.1\t\t0\tc"]:
            f.write(line + "\n")
        f.write(f"{stop or STOP_COND}\t\t\titeration stop condition\n")


def chi2_of(fit_path, lcs_path):
    """Residual sum of squares of convexinv's model against the input brightnesses, per block
    renormalised for relative blocks (as convexinv does)."""
    model = np.loadtxt(fit_path).astype(float)
    obs, flags, sizes = [], [], []
    with open(lcs_path) as f:
        nb = int(f.readline())
        for _ in range(nb):
            n, flag = map(int, f.readline().split()[:2])
            vals = [float(f.readline().split()[1]) for _ in range(n)]
            obs += vals; flags.append(flag); sizes.append(n)
    obs = np.array(obs)
    chi2, i = 0.0, 0
    for n, flag in zip(sizes, flags):
        o, m = obs[i:i + n], model[i:i + n]
        if flag == 0:
            o, m = o / o.mean(), m / m.mean()
        chi2 += float(np.sum((o / m - 1) ** 2))
        i += n
    return chi2 / len(obs)


def chi2_weighted(fit_path, lcs_path, par_path=None):
    """convexinv's own objective (mrqcof.c): residual over the block mean brightness, times the point
    weight; relative blocks compared after renormalising both to unit mean. Returns (weighted mean
    chi2, TESS mean squared residual, sparse mean squared residual).

    For relative blocks convexinv writes its model with the phase function divided out. Within a
    TESS session the phase angle barely moves, but ATLAS-cyan and Gaia relative blocks span years
    and tens of degrees, so with par_path the phase function 1 + a exp(-alpha/d) + k alpha (fitted
    a, d, k; alpha from each point's Sun and observer vectors) is restored before comparing."""
    model = np.loadtxt(fit_path).astype(float)
    phase = None
    if par_path is not None:
        a_, d_, k_ = map(float, open(par_path).read().split()[5:8])
        phase = lambda al: 1 + a_ * np.exp(-al / d_) + k_ * al
    tot, wsum, parts, i = 0.0, 0.0, {0: [], 1: []}, 0
    with open(lcs_path) as f:
        for _ in range(int(f.readline())):
            h = f.readline().split()
            n, flag, w = int(h[0]), int(h[1]), float(h[2]) if len(h) > 2 else 1.0
            pp = len(h) > 3 and int(h[3]) == 1
            rows = np.array([[float(x) for x in f.readline().split()] for _ in range(n)])
            o = rows[:, 1]
            wp = w * (rows[:, 8] if pp else np.ones(n))
            m = model[i:i + n]
            if flag == 0 and phase is not None:
                sv_, ev_ = rows[:, 2:5], rows[:, 5:8]
                al = np.arccos(np.clip(np.sum(sv_ * ev_, 1) / (np.linalg.norm(sv_, axis=1) * np.linalg.norm(ev_, axis=1)), -1, 1))
                m = m * phase(al)
            r = o / o.mean() - m / m.mean() if flag == 0 else (o - m) / o.mean()
            tot += float(np.sum(wp * r ** 2)); wsum += float(np.sum(wp))
            is_tess = (int(h[4]) == 0) if len(h) > 4 else (flag == 0 and not pp)   # source tag, else old layout
            parts[0 if is_tess else 1].append(r)
            i += n
    ms = {k: float(np.mean(np.concatenate(v) ** 2)) if v else np.nan for k, v in parts.items()}
    return tot / wsum, ms[0], ms[1]


def dataset_weights(scheme, w_tess_abs, w_sparse_abs):
    """Per-point weights for TESS and the sparse data (ATLAS, Gaia), each given as
    1 / (sigma_i^2 + sigma_model^2): the point's photometric error plus that dataset's model-error
    floor, in relative flux. Normalised to a mean of 1 per point so the balance against convexinv's
    convexity regularisation is unchanged.
      noise: as given (one noise model throughout, a proper chi^2)
      equal: TESS and the sparse data carry the same total weight; within each, the points keep
             their relative weights
      scaled: halfway (geometrically) between the two: total sparse / total TESS = sigma_TESS /
             sigma_sparse, each sigma the effective per-point scatter 1 / sqrt(mean weight).
             Equal is a ratio of 1 and noise (sigma_TESS / sigma_sparse)^2; equal let Kalliope's
             poor ATLAS (floor 0.12 mag, 2-3.5x the others) carry half the fit (tested: worse)"""
    w_t, w_s = np.asarray(w_tess_abs, float), np.asarray(w_sparse_abs, float)
    if scheme == "equal":
        w_t = w_t * (w_s.sum() / w_t.sum())
    elif scheme == "scaled":
        ratio = np.sqrt(np.mean(w_s) / np.mean(w_t))   # sigma_TESS / sigma_sparse
        w_t = w_t * (w_s.sum() / (w_t.sum() * ratio))
    elif scheme != "noise":
        raise ValueError(scheme)
    c = (len(w_t) + len(w_s)) / (w_t.sum() + w_s.sum())
    return w_t * c, w_s * c


def fib_poles(n):
    i = np.arange(n) + 0.5
    lon = np.degrees(np.pi * (1 + 5 ** 0.5) * i) % 360
    return [(float(l), float(np.degrees(np.arcsin(z)))) for l, z in zip(lon, 1 - 2 * i / n)]


def unit(lam, bet):
    l, b = np.radians(lam), np.radians(bet)
    return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def lonlat(v):
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360), float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))


def sep_deg(p, q):
    return float(np.degrees(np.arccos(np.clip(unit(*p) @ unit(*q), -1, 1))))


def neighbours(lam, bet, r_deg, n=8):
    p = unit(lam, bet)
    a = np.cross(p, [0.0, 0.0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(p, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(p, a)
    r = np.radians(r_deg)
    return [lonlat(np.cos(r) * p + np.sin(r) * (np.cos(t) * a + np.sin(t) * b))
            for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]


def read_areas(path):
    """Facet areas and unit normals from convexinv's output areas file, dropping the last entry:
    the 'dark facet' that only closes the shape (bright.c sums facets 1..Numfac)."""
    tok = open(path).read().split()
    n = int(tok[0])
    v = np.array(tok[1:1 + 4 * n], float).reshape(n, 4)
    return v[:-1, 0], v[:-1, 1:]


def body_frame(vec, lam, bet):
    """Unit vectors rotated by convexinv's blmatrix(90 - beta, lambda): ecliptic -> body frame
    before the spin rotation."""
    th, l = np.radians(90 - bet), np.radians(lam)
    cb, sb, cl, sl = np.cos(th), np.sin(th), np.cos(l), np.sin(l)
    B = np.array([[cb * cl, cb * sl, -sb], [-sl, cl, 0.0], [sb * cl, sb * sl, cb]])
    return (vec / np.linalg.norm(vec, axis=1, keepdims=True)) @ B.T


def brightness(A, N, e_obs, e_sun, phi):
    """Disc-integrated brightness as bright.c computes it, without the phase function: spin
    rotation by phi about body z (matrix.c), then the sum over facets both lit and visible of
    area * mu mu0 (c + 1 / (mu + mu0)). e_obs, e_sun: (n, 3) body-frame unit vectors;
    phi: (n,) or (n, m) radians. Returns phi's shape."""
    one_d = phi.ndim == 1
    phi = phi[:, None] if one_d else phi
    out = np.empty(phi.shape)
    step = max(1, int(4e6 // (phi.shape[1] * len(A))))
    for s0 in range(0, len(phi), step):
        sl = slice(s0, s0 + step)
        c, sn = np.cos(phi[sl])[..., None], np.sin(phi[sl])[..., None]

        def mu_of(e):
            p1 = e[sl, 0:1] * N[:, 0] + e[sl, 1:2] * N[:, 1]
            p2 = e[sl, 1:2] * N[:, 0] - e[sl, 0:1] * N[:, 1]
            return c * p1[:, None] + sn * p2[:, None] + (e[sl, 2:3] * N[:, 2])[:, None]
        mu, mu0 = mu_of(e_obs), mu_of(e_sun)
        lit = (mu > TINY) & (mu0 > TINY)
        out[sl] = np.where(lit, mu * mu0 * (C_LAMBERT + 1 / np.where(lit, mu + mu0, 1.0)), 0.0) @ A
    return out[:, 0] if one_d else out


_CTX = {}


def model_check(A, N, lam, bet, tok, fp):
    """Largest difference between our forward model and convexinv's own TESS model, both
    normalised per block, so a convention slip can never silently feed the ATLAS ranking."""
    c = _CTX
    per, jd0, phi0 = float(tok[2]), float(tok[3]), np.radians(float(tok[4]))
    eo, es = body_frame(c["tess_ev"], lam, bet), body_frame(c["tess_sv"], lam, bet)
    phi = phi0 + 2 * np.pi * (c["tess_jd"] - jd0) / (per / 24.0)
    # no phase function: for relative lightcurves convexinv writes the model with it divided out
    # (convexinv.c, output lc); including it leaves a 1.5e-3 phase-angle trend within each block
    ours = brightness(A, N, eo, es, phi)
    theirs = np.loadtxt(fp).astype(float)
    dev = 0.0
    for sl in c["tess_slices"]:
        dev = max(dev, float(np.max(np.abs(ours[sl] / ours[sl].mean() - theirs[sl] / theirs[sl].mean()))))
    return dev


def atlas_scan(A, N, lam, bet, per_hr, jd0, phi0):
    """ATLAS rms of the fixed shape over the sidereal-period scan. Rotation phase is anchored at
    the TESS mid-epoch, where the TESS fit pins it; the phase curve (offset, linear and exponential
    terms in phase angle) is refitted linearly at every period. Returns (best rms, best period h,
    median rms over the scan)."""
    c = _CTX
    eo, es = body_frame(c["atlas_ev"], lam, bet), body_frame(c["atlas_sv"], lam, bet)
    grid = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    mag = -2.5 * np.log10(np.maximum(brightness(A, N, eo, es, np.broadcast_to(grid, (len(eo), N_PHI))), 1e-30))
    P0 = per_hr / 24.0
    phi_mid = phi0 + 2 * np.pi * (c["tess_mid"] - jd0) / P0
    dt = c["atlas_jd"] - c["tess_mid"]
    rows = np.arange(len(dt))
    Q, y_obs, sw = c["atlas_Q"], c["atlas_m"], c["sparse_sw"]
    rms, best = [], (np.inf, None)
    for u in np.array_split(c["u_grid"], max(1, len(c["u_grid"]) // 500)):
        x = ((phi_mid + 2 * np.pi * dt[None, :] / (P0 * (1 + u[:, None]))) % (2 * np.pi)) / (2 * np.pi) * N_PHI
        i0 = np.floor(x).astype(int) % N_PHI
        f = x - np.floor(x)
        y = sw * (y_obs[None, :] - (mag[rows, i0] * (1 - f) + mag[rows, (i0 + 1) % N_PHI] * f))
        # weighted least squares, sqrt(w) scaled: |r|^2 = |y|^2 - |y Q|^2 (Q orthonormal), so the
        # residual matrix itself is formed only at the chunk's best period
        rr = np.sqrt(np.maximum(np.einsum("ij,ij->i", y, y) - np.sum((y @ Q) ** 2, axis=1), 0) / np.sum(sw ** 2))
        j = int(np.argmin(rr))
        if rr[j] < best[0]:
            best = (rr[j], (y[j] - (y[j] @ Q) @ Q.T) / sw)    # unscaled residuals at the best period
        rms.append(rr)
    rms = np.concatenate(rms)
    i = int(np.argmin(rms))
    per_src = {s_: float(np.sqrt(np.mean(best[1][m_] ** 2))) for s_, m_ in c["src_masks"].items()}
    # starting periods for the weighted fits: the best local minima of the scan, at least half an
    # alias step (one extra cycle over the data span, du = P / T) apart. convexinv's optimiser stays
    # in the minimum nearest its start (Loreley: within 3e-5 h), and the TESS-only shape can rank
    # the true minimum below an alias, so one start is not enough
    loc = np.where((rms[1:-1] <= rms[:-2]) & (rms[1:-1] <= rms[2:]))[0] + 1
    cands = []
    for j in loc[np.argsort(rms[loc])]:
        if all(abs(c["u_grid"][j] - c["u_grid"][q]) > 0.5 * c["du_alias"] for q in cands):
            cands.append(j)
        if len(cands) == N_PERIOD_STARTS:
            break
    starts = [float(per_hr * (1 + c["u_grid"][j])) for j in cands]
    return float(rms[i]), float(per_hr * (1 + c["u_grid"][i])), float(np.median(rms)), per_src, starts


def pole_step(job):
    """TESS-only shape at a fixed pole, then its ATLAS score."""
    k, lam, bet = job
    c = _CTX
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, c["cpus"])
    cp, sp, pp, fp = (f"{c['work']}/it{k}_{x}.txt" for x in "cspf")
    write_control(cp, lam, bet, c["period_hr"], pole_free=False, phase_free=False, stop=RANK_STOP)
    res, why = bs.run_pole(cp, c["lcs_tess"], sp, pp, fp)
    out = dict(k=k, lam=lam, bet=bet, pid=os.getpid(),
               n_cpus_allowed=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    if res is None:
        return dict(out, error=why)
    tok = open(pp).read().split()
    A, N = read_areas(sp)
    dev = model_check(A, N, lam, bet, tok, fp)
    out.update(period_tess_hr=float(tok[2]), rotation_t0_jd=float(tok[3]),
               rotation_phi0_rad=float(np.radians(float(tok[4]))), model_check=dev,
               chi2_tess=chi2_of(fp, c["lcs_tess"]), shape=sp)
    if dev > MODEL_CHECK_TOL:
        return dict(out, error=f"forward model differs from convexinv by {dev:.2e}")
    rms, per, med, per_src, starts = atlas_scan(A, N, lam, bet, float(tok[2]), float(tok[3]), np.radians(float(tok[4])))
    return dict(out, atlas_rms=rms, period_atlas_hr=per, atlas_rms_median=med, period_starts_hr=starts,
                **{f"rms_{s_}": v for s_, v in per_src.items()})


def joint_step(job):
    """Weighted TESS + sparse fit with the pole fixed, from each starting period; keeps the best.
    The first start runs cold (or from coef_in) and writes its shape coefficients; the others start
    from those, so they converge quickly and differ from it only by period."""
    k, lam, bet, starts, lcs_w = job[:5]
    coef_in = job[5] if len(job) > 5 else None
    c = _CTX
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, c["cpus"])
    best, tried, warm = None, [], coef_in
    for j, per0 in enumerate(starts):
        cp, sp, pp, fp, cf = (f"{c['work']}/jw{k}_{j}_{x}.txt" for x in ("c", "s", "p", "f", "coef"))
        write_control(cp, lam, bet, per0, pole_free=False, phase_free=True, stop=RANK_STOP if k < 100000 else None)
        res, why = run_ci(cp, lcs_w, sp, pp, fp, (["-i", warm] if warm else []) + ["-c", cf])
        if res is None:
            continue
        if warm is None:
            warm = cf
        tok = open(pp).read().split()
        c2, c2_t, c2_a = chi2_weighted(fp, lcs_w, pp)
        tried.append((round(per0, 7), round(float(tok[2]), 7), c2))
        if best is None or c2 < best["chi2_w"]:
            best = dict(k=k, chi2_w=c2, chi2_w_tess=c2_t, chi2_w_atlas=c2_a, period_joint_hr=float(tok[2]),
                        period_start_hr=per0, rotation_t0_jd_w=float(tok[3]),
                        rotation_phi0_rad_w=float(np.radians(float(tok[4]))), shape_w=sp, coef_w=cf)
    if best is None:
        return dict(k=k, joint_error="every period start failed")
    return dict(best, period_tries=tried)


def run_iterative(a, df, sv, ev, blocks, tess_rows, atlas, sv_a, ev_a, info, work, lcs_joint, weights=None,
                  tess_atlas_rows=None):
    # options for the final pole-free fits only (best and mirror): a free YORP term, and Huber
    # reweighting (k robust sigmas, 2 passes). Not used while ranking poles, where every pole must be
    # scored with the same weights and parameters.
    final_extra = (["-y", 0, 1] if getattr(a, "yorp", False) else []) + \
        (["-r", a.robust, 2] if getattr(a, "robust", None) else [])
    lcs_tess = f"{work}/lcs_tess.txt"
    bs.write_lcs(df, sv, ev, blocks, lcs_tess)
    order = np.concatenate(blocks)
    jd = bs.lt_jd(df, ev)[order]
    edges = np.cumsum([0] + [len(b) for b in blocks])
    # widest period range the TESS fit could be off by: one cycle over the longest visit
    t_vis = max(np.ptp(df.mjd.values[b]) for b in blocks)
    atlas_jd = atlas.MJD_lc.values + 2400000.5
    tess_mid = float(np.mean(jd))
    P0 = a.period_hr / 24.0
    half = P0 / max(t_vis, P0)
    du = 0.05 * P0 / np.max(np.abs(atlas_jd - tess_mid))         # <= 0.05 cycle drift per step
    n_u = int(2 * half / du) + 1
    if n_u > 40000:
        print(f"  period scan capped at 40000 steps (needed {n_u})", flush=True)
        n_u = 40000
    # alias step: one extra rotation over the whole data span, as a fraction of the period
    t_all = np.concatenate([atlas_jd, jd])
    du_alias = P0 / np.ptp(t_all)
    alpha = atlas.SOE.values
    src = atlas.src.values
    src_masks = {s_: (src == s_) for s_ in ["atlas", "gaia"] + sorted(set(src) - {"atlas", "gaia"}) if (src == s_).any()}
    cols = [np.ones_like(alpha), alpha, np.exp(-alpha / info["phase_d_deg"])]
    for s_ in src_masks:
        if s_ != "atlas":
            cols.append(src_masks[s_].astype(float))   # free Gaia / survey zero-point in every period's fit
    is_cyan = (atlas.filt.values == "c") if "filt" in atlas else np.zeros(len(atlas), bool)
    if is_cyan.sum() >= 5:
        cols.append(is_cyan.astype(float))            # free ATLAS cyan offset, as in the weighted fits
    # scan weights 1 / (sigma_i^2 + floor^2); before any shape exists the floor is each source's
    # phase-curve residual (which still contains the rotational lightcurve), so they are gentle
    sig2 = atlas.dm.values ** 2
    w0 = np.empty(len(atlas))
    for s_, m_ in src_masks.items():
        rms_ph = info["phase_fit_rms_by_src"][s_]
        w0[m_] = 1 / (sig2[m_] + max(rms_ph ** 2 - np.median(sig2[m_]), 1e-6))
    w0 /= w0.mean()
    sw = np.sqrt(w0)
    X = np.column_stack(cols) * sw[:, None]
    _CTX.update(work=work, period_hr=a.period_hr, lcs_tess=lcs_tess, tess_jd=jd,
                tess_sv=sv[order], tess_ev=ev[order],
                tess_slices=[slice(edges[i], edges[i + 1]) for i in range(len(blocks))],
                tess_mid=tess_mid, atlas_jd=atlas_jd, atlas_sv=sv_a, atlas_ev=ev_a,
                atlas_m=atlas.m_o.values, atlas_Q=np.linalg.qr(X)[0], u_grid=np.linspace(-half, half, n_u),
                sparse_sw=sw, src_masks=src_masks, du_alias=du_alias)
    print(f"  period scan: +/-{100 * half:.3f}% in {n_u} steps", flush=True)

    cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count()))
    ncpu = int(os.environ.get("ITER_NPROC", len(cpus)))   # local runs: leave the machine headroom
    _CTX["cpus"] = cpus
    print(f"  {ncpu} workers, cpus allowed {sorted(cpus)}", flush=True)
    done, k, lcs_w, wmeta = [], 0, None, None
    n_t, n_a = len(order), len(atlas)
    src_masks = _CTX["src_masks"]
    score_name = f"weighted chi2 ({weights})" if weights else "ATLAS rms"
    score = (lambda r: r["chi2_w"]) if weights else (lambda r: r["atlas_rms"])
    with mp.get_context("fork").Pool(ncpu) as pool:
        poles = fib_poles(FIB_N)
        for level in range(len(REFINE) + 1):
            jobs = [(k + i, l, b) for i, (l, b) in enumerate(poles)]
            k += len(jobs)
            new = []
            for r in pool.imap_unordered(pole_step, jobs):
                done.append(r); new.append(r)
                if "error" in r:
                    print(f"  pole ({r['lam']:6.1f},{r['bet']:+5.1f}) failed: {r['error']}", flush=True)
            if weights:
                if lcs_w is None:                         # weights fixed once, from the coarse grid
                    good = [r for r in done if "error" not in r]
                    var_t = min(r["chi2_tess"] for r in good)
                    # per-point sparse weights 1 / (sigma_i^2 + floor^2), the floor (model error) from
                    # each source's best coarse-grid residual less its median photometric variance;
                    # magnitudes -> relative flux by 0.4 ln 10
                    k2 = (0.4 * np.log(10)) ** 2
                    sig2 = atlas.dm.values ** 2
                    w_abs, floors = np.empty(n_a), {}
                    for s_, m_ in src_masks.items():
                        rms_min = min(r[f"rms_{s_}"] for r in good)
                        floors[s_] = float(np.sqrt(max(rms_min ** 2 - np.median(sig2[m_]), 1e-6)))
                        w_abs[m_] = 1 / (k2 * (sig2[m_] + floors[s_] ** 2))
                    # TESS likewise per point, 1 / (sigma_i^2 + floor^2) with sigma_i = e_flux / flux and
                    # the floor from the best TESS-only residual (mean squared relative residual)
                    t_sig2 = np.array([r[5] for r in tess_atlas_rows[0]], float) ** 2
                    if not np.isfinite(t_sig2).any():
                        t_sig2 = np.zeros(len(t_sig2))            # no errors: uniform TESS weights
                    t_sig2 = np.where(np.isfinite(t_sig2), t_sig2, np.nanmedian(t_sig2))
                    floor_t = float(np.sqrt(max(var_t - np.median(t_sig2), 1e-12)))
                    w_t, w_s = dataset_weights(weights, 1 / (t_sig2 + floor_t ** 2), w_abs)
                    lcs_w = f"{work}/lcs_weighted.txt"
                    write_lcs(lcs_w, *tess_atlas_rows, w_tess=w_t, w_sparse=w_s)
                    tot = {s_: float(w_s[m_].sum()) for s_, m_ in src_masks.items()}
                    wmeta = dict(scheme=weights, sigma_tess=float(np.sqrt(var_t)), model_floor_tess=floor_t,
                                 tess_median_rel_err=float(np.sqrt(np.median(t_sig2))),
                                 model_floor_mag=floors, total_weight={"tess": float(w_t.sum()), **tot},
                                 median_point_weight={"tess": float(np.median(w_t)),
                                                      **{s_: float(np.median(w_s[m_])) for s_, m_ in src_masks.items()}})
                    print(f"  weights ({weights}): TESS median {np.median(w_t):.3g}/point (range {w_t.min():.3g}-{w_t.max():.3g}), "
                          f"total {w_t.sum():.0f}, floor {floor_t:.4f}, median error {np.sqrt(np.median(t_sig2)):.4f}; " +
                          "; ".join(f"{s_} median {wmeta['median_point_weight'][s_]:.3g}/point, total {tot[s_]:.0f}, "
                                    f"floor {floors[s_]:.4f} mag" for s_ in src_masks), flush=True)
                by_k = {r["k"]: r for r in done}
                n_starts = COARSE_STARTS if level == 0 else N_PERIOD_STARTS
                jobs_w = [(r["k"], r["lam"], r["bet"], (r["period_starts_hr"] or [r["period_atlas_hr"]])[:n_starts], lcs_w)
                          for r in new if "error" not in r]
                for jr in pool.imap_unordered(joint_step, jobs_w):
                    by_k[jr["k"]].update(jr)
                    if "joint_error" in jr:
                        print(f"  weighted fit at pole k={jr['k']} failed: {jr['joint_error']}", flush=True)
            ok = sorted((r for r in done if "error" not in r and "joint_error" not in r), key=score)
            print(f"  level {level}: {len(jobs)} poles; best {score_name} " +
                  ", ".join(f"({r['lam']:.0f},{r['bet']:+.0f}) {score(r):.5f}" for r in ok[:3]), flush=True)
            if level == len(REFINE):
                break
            n_keep, r_deg = REFINE[level]
            # the best pole of each separate sky region, so a runner-up region is refined as deeply as
            # the leader (Elektra: the region 12 deg from DAMIT stopped at 10 deg resolution while two
            # seeds 11 deg apart in the leading region took the 4-deg level)
            seeds = []
            for r in ok:
                if all(sep_deg((r["lam"], r["bet"]), (s["lam"], s["bet"])) > REGION_SEP_DEG for s in seeds):
                    seeds.append(r)
                if len(seeds) == n_keep:
                    break
            poles = [p for s in seeds for p in neighbours(s["lam"], s["bet"], r_deg)
                     if min(sep_deg(p, (d["lam"], d["bet"])) for d in done) > r_deg / 2]

    ok = sorted((r for r in done if "error" not in r and "joint_error" not in r), key=score)
    if weights and CONVERGE_TOP:
        top = ok[:CONVERGE_TOP]
        with mp.get_context("fork").Pool(ncpu) as pool:
            res_c = pool.map(joint_step, [(r["k"] + 500000, r["lam"], r["bet"],
                                           r["period_starts_hr"] or [r["period_atlas_hr"]], lcs_w) for r in top])
        for r, jr in zip(top, res_c):
            if "joint_error" in jr:
                continue
            r["chi2_w_ranking"] = r["chi2_w"]
            r.update({kk: v for kk, v in jr.items() if kk != "k"})
        ok = sorted(ok, key=score)
        print(f"  converged top {len(top)}: best " + ", ".join(f"({r['lam']:.0f},{r['bet']:+.0f}) {score(r):.5f}"
                                                            for r in ok[:3]), flush=True)
    if weights:
        # dense period grid (+/- FINAL_GRID_ALIASES alias steps, quarter-step spacing) around the
        # best period of the best pole in each of the top regions
        seeds = []
        for r in ok:
            if all(sep_deg((r["lam"], r["bet"]), (s_["lam"], s_["bet"])) > REGION_SEP_DEG for s_ in seeds):
                seeds.append(r)
            if len(seeds) == REFINE[-1][0]:
                break
        n_g = 4 * FINAL_GRID_ALIASES
        with mp.get_context("fork").Pool(ncpu) as pool:
            jobs_g = [(r["k"] + 100000 * (g + 1), r["lam"], r["bet"],
                       [r["period_joint_hr"] * (1 + du_alias * (g - n_g) / 4)], lcs_w, r["coef_w"])
                      for r in seeds for g in range(2 * n_g + 1)]
            res_g = pool.map(joint_step, jobs_g)
        for r, chunk in zip(seeds, [res_g[i:i + 2 * n_g + 1] for i in range(0, len(res_g), 2 * n_g + 1)]):
            good = [x for x in chunk if "joint_error" not in x]
            if not good:
                continue
            g_best = min(good, key=lambda x: x["chi2_w"])
            r["final_grid"] = [(x["period_start_hr"], x["period_joint_hr"], x["chi2_w"]) for x in good]
            if g_best["chi2_w"] < r["chi2_w"]:
                print(f"  period grid at ({r['lam']:.0f},{r['bet']:+.0f}): {r['period_joint_hr']:.6f} -> "
                      f"{g_best['period_joint_hr']:.6f} h, chi2 {r['chi2_w']:.5f} -> {g_best['chi2_w']:.5f}", flush=True)
                r.update({kk: v for kk, v in g_best.items() if kk not in ("k", "period_tries")})
        ok = sorted(ok, key=score)
    best = ok[0]
    far = [r for r in ok if sep_deg((r["lam"], r["bet"]), (best["lam"], best["bet"])) > 45]
    alt = far[0] if far else None
    print(f"ITERATIVE {a.designation}: pole ({best['lam']:.1f},{best['bet']:+.1f}) "
          f"P {best.get('period_joint_hr', best['period_atlas_hr']):.7f} h {score_name} {score(best):.5f}, "
          f"ATLAS rms {best['atlas_rms']:.4f} mag"
          + (f"; best pole >45 deg away ({alt['lam']:.0f},{alt['bet']:+.0f}) {score(alt):.5f}" if alt else ""),
          flush=True)

    # final fit with the pole free, from the best pole: unweighted joint polish in the ATLAS-only
    # ranking, the same weighted objective as the ranking otherwise
    cp, sp, pp, fp, ef = (f"{work}/final_{x}.txt" for x in ("c", "s", "p", "f", "err"))
    lcs_final = lcs_w if weights else lcs_joint
    write_control(cp, best["lam"], best["bet"], best.get("period_joint_hr", best["period_atlas_hr"]))
    final_opts = ["-e", ef] + (["-i", best["coef_w"]] if best.get("coef_w") else []) + final_extra
    res, why = run_ci(cp, lcs_final, sp, pp, fp, final_opts)
    joint = None
    if res is None:
        print(f"  final joint fit failed: {why}", flush=True)
    else:
        tok = open(pp).read().split()
        joint = dict(lam=res[0], bet=res[1], period_hr=res[2],
                     chi2=chi2_weighted(fp, lcs_w, pp)[0] if weights else chi2_of(fp, lcs_joint), shape=sp,
                     rotation_t0_jd=float(tok[3]), rotation_phi0_rad=float(np.radians(float(tok[4]))),
                     uncertainties=read_errors(ef))
        print(f"JOINT-POLISH {a.designation}: pole ({res[0]:.1f},{res[1]:+.1f}) P {res[2]:.7f} h "
              f"chi2 {joint['chi2']:.5f}", flush=True)

    mirror = None
    if weights and joint:
        # the (lambda + 180, beta) mirror: photometry of a near-ecliptic orbit constrains the pole only
        # up to it (Loreley: 0.00067 vs 0.00068), so it is fitted the same way and reported alongside
        ml, mb = (joint["lam"] + 180.0) % 360.0, joint["bet"]
        n_g = 4 * FINAL_GRID_ALIASES
        with mp.get_context("fork").Pool(ncpu) as pool:
            res_m = pool.map(joint_step, [(900000 + g, ml, mb, [joint["period_hr"] * (1 + du_alias * (g - n_g) / 4)], lcs_w)
                                          for g in range(2 * n_g + 1)])
        good = [x for x in res_m if "joint_error" not in x]
        if good:
            m_best = min(good, key=lambda x: x["chi2_w"])
            cp, sp, pp, fp, ef = (f"{work}/mirror_{x}.txt" for x in ("c", "s", "p", "f", "err"))
            write_control(cp, ml, mb, m_best["period_joint_hr"])
            res, why = run_ci(cp, lcs_w, sp, pp, fp, ["-e", ef, "-i", m_best["coef_w"]] + final_extra)
            if res is not None:
                tok = open(pp).read().split()
                c2 = chi2_weighted(fp, lcs_w, pp)[0]
                mirror = dict(lam=res[0], bet=res[1], period_hr=res[2], chi2=c2, chi2_ratio_to_best=c2 / joint["chi2"],
                              rotation_t0_jd=float(tok[3]), rotation_phi0_rad=float(np.radians(float(tok[4]))), shape=sp,
                              uncertainties=read_errors(ef))
                print(f"MIRROR {a.designation}: pole ({res[0]:.1f},{res[1]:+.1f}) P {res[2]:.7f} h chi2 {c2:.5f} "
                      f"({c2 / joint['chi2']:.3f} x the best)", flush=True)

    def mesh_of(shape):
        mk = bs.minkowski_py_mesh(shape) or bs.minkowski(shape)
        if mk is None:
            return None, None
        V, F = mk
        wmax, wmin = bs.caliper_widths(V)
        return float(wmax / wmin), {"v": V.ravel().tolist(), "f": [i for f in F for i in f], "fn": [len(f) for f in F]}
    # weighted: the pole's model is its weighted fixed-pole fit (shape_w, period_joint_hr, *_w phase);
    # otherwise the TESS shape with the ATLAS-scan rotation phase:
    # phi(t) = phi0 + 2 pi (tess_mid - t0) / P_tess + 2 pi (t - tess_mid) / P_atlas
    ratio_i, mesh_i = mesh_of(best["shape_w"] if weights else best["shape"])
    out = dict(mode="weighted" if weights else "iterative", tess_mid_jd=tess_mid, weights=wmeta,
               iterative=dict(best, equatorial_ratio=ratio_i, mesh=mesh_i),
               alternative_pole=alt,
               all=[{kk: v for kk, v in r.items() if kk not in ("shape", "shape_w")} for r in done])
    if joint:
        ratio_j, mesh_j = mesh_of(joint["shape"])
        out["joint_polish"] = dict(joint, equatorial_ratio=ratio_j, mesh=mesh_j)
    if mirror:
        ratio_m, mesh_m = mesh_of(mirror["shape"])
        out["mirror"] = dict(mirror, equatorial_ratio=ratio_m, mesh=mesh_m)
    for d in (out["iterative"], out.get("joint_polish", {}), out.get("mirror", {}), out["alternative_pole"] or {}):
        d.pop("shape", None); d.pop("shape_w", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("designation"); ap.add_argument("period_hr", type=float)
    ap.add_argument("--mode", choices=["joint", "iterative"], default="joint")
    ap.add_argument("--yorp", action="store_true", help="free YORP d(omega)/dt in the final fits")
    ap.add_argument("--robust", type=float, default=None, help="Huber k (robust sigmas) for the final fits")
    ap.add_argument("--weights", choices=["noise", "equal", "scaled"], default=None,
                    help="iterative mode: rank poles by a weighted fixed-pole TESS+ATLAS fit (needs the CheerfulUser/DAMIT-convex convexinv)")
    ap.add_argument("--out", default="combined_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = bs.sn(a.designation)
    work = f"{a.out}/work_{tag}"
    os.makedirs(work, exist_ok=True)

    df = bs.prepare(pd.read_csv(f"{LCDIR}/{tag}_clean_lc.csv"))
    sv, ev = bs.build_geometry(df)
    blocks = bs.session_blocks(df)
    tess_rows = tess_points(df, sv, ev)
    raw = atlas_rows(pack(a.designation))
    raw = with_geometry(raw.assign(src="atlas"), *atlas_geometry(raw))
    g = gaia_rows(a.designation)
    if g is not None:
        raw = pd.concat([raw, g], ignore_index=True)
    sv_rows = survey_rows(a.designation)
    if sv_rows is not None:
        raw = pd.concat([raw, sv_rows], ignore_index=True)
    atlas, info = prepare_atlas(raw)
    atlas = atlas.reset_index(drop=True)
    sv_a, ev_a = atlas[["sx", "sy", "sz"]].values, atlas[["ex", "ey", "ez"]].values
    lcs = f"{work}/lcs.txt"
    n_tess, n_atlas = write_lcs(lcs, tess_rows, len(blocks), atlas, sv_a, ev_a)
    print(f"{a.designation}: {n_tess} TESS points in {len(blocks)} sessions; ATLAS {info}", flush=True)
    base = dict(designation=a.designation, period_start_hr=a.period_hr, atlas=info, n_tess=n_tess, n_atlas=n_atlas)
    if a.mode == "iterative":
        out = run_iterative(a, df, sv, ev, blocks, tess_rows, atlas, sv_a, ev_a, info, work, lcs,
                            weights=a.weights, tess_atlas_rows=(tess_rows, len(blocks), atlas, sv_a, ev_a))
        json.dump(dict(base, **out), open(f"{a.out}/{tag}.json", "w"))
        return

    sols = []
    for i, (l0, b0) in enumerate(START_POLES):
        cp, sp, pp, fp = (f"{work}/{k}{i}.txt" for k in "cspf")
        write_control(cp, l0, b0, a.period_hr)
        res, why = bs.run_pole(cp, lcs, sp, pp, fp)
        if res is None:
            print(f"  start ({l0},{b0}) failed: {why}", flush=True); continue
        c2 = chi2_of(fp, lcs)
        tok = open(pp).read().split()
        sols.append(dict(start=[l0, b0], lam=res[0], bet=res[1], per=res[2], chi2=c2, shape=sp, params=pp,
                         rotation_t0_jd=float(tok[3]), rotation_phi0_rad=float(np.radians(float(tok[4])))))
        print(f"  start ({l0:3d},{b0:+3d}) -> pole ({res[0]:6.1f},{res[1]:+5.1f}) P {res[2]:.6f} h chi2 {c2:.5f}",
              flush=True)
    best = min(sols, key=lambda s: s["chi2"])
    V, F = bs.minkowski(best["shape"])
    wmax, wmin = bs.caliper_widths(V)
    out = dict(base, mode="joint",
               best=dict(lam=best["lam"], bet=best["bet"], period_hr=best["per"], chi2=best["chi2"],
                         rotation_t0_jd=best["rotation_t0_jd"], rotation_phi0_rad=best["rotation_phi0_rad"],
                         equatorial_ratio=float(wmax / wmin)),
               all=[{k: v for k, v in s.items() if k not in ("shape", "params")} for s in sols],
               mesh={"v": V.ravel().tolist(), "f": [i for f in F for i in f], "fn": [len(f) for f in F]})
    json.dump(out, open(f"{a.out}/{tag}.json", "w"))
    print(f"BEST {a.designation}: pole ({best['lam']:.1f},{best['bet']:+.1f}) P {best['per']:.6f} h "
          f"chi2 {best['chi2']:.5f}, a/b {wmax / wmin:.2f}", flush=True)


if __name__ == "__main__":
    main()
