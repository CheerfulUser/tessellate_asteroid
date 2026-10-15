"""Injection-recovery test on BLANK-SKY paths: synthetic asteroids measured with the production forced photometry
along random paths through the reduced TESSELLATE frames, so the only signal is the injected one. Runs on ozstar.

Each synthetic object copies a real source asteroid's measured frames (store photometry: sectors, cuts, frame
indices, times, geometry), with every visit's path moved by a random offset inside its cut, at least AVOID_PX from
predicted asteroids (the cut's ephemeris): frames that leave the cut or pass within AVOID_PX of one are dropped,
and a moved visit is used when at least MIN_KEEP_FRAC of its frames remain. Along the moved path: forced PSF photometry
(asteroid_photometry.forced_psf_photometry, production settings; positions are exact, so the production per-track
offset fit is skipped) -> injected flux added -> detrend_pixel_phase -> flag_star_contamination -> the store's
photometry table -> the unchanged period pipeline (sector_report_pipeline: _store_frame, frames_to_points,
normalise, plot_asteroid).

Injected flux per frame: F = s_f e_ref 10^(-0.4 (mag_expected - m_ref)) (1 + A s(phase)), with s_f the per-frame
S/N drawn evenly over SNR_BANDS (log-uniform within), e_ref / m_ref the source's median e_flux / mag_expected, s a
template profile (zero mean, unit peak-to-peak) from injection_test.py's templates.csv and A its own measured
amplitude; source photon noise N(0, sqrt(F / T_EFF)) is added and folded into e_flux (flux in e-/s). Kinds:
rotator; blank (F = 0, BLANK_FRAC); constant (A = 0, CONST_FRAC).

Stages (RUN names the output folder asteroids_store/blank_injection/<RUN>):
  plan <n_src> <n_copy>   sources (random store objects with >= MIN_SRC_POINTS points), their photometry, the
                          synthetic objects' parameters and the list of cuts to measure
  phot <index> <total>    forced photometry + injection for every synthetic visit in this task's cuts ->
                          phot/bucket=<b>/ (b = object number % N_BUCKETS)
  pipe <bucket>           period pipeline on every object of one bucket -> pipe_<bucket>.csv
  check                   the controls (written before running, see CONTROLS) and recovery overview
  blind <n>               blind labelling plots (neutral names) + key

    RUN=test python3 -u blank_injection.py plan 30 10
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import sector_report_pipeline as p
from tessellate import asteroid_store as st

RUN = os.environ.get("RUN", "test")
OUT = f"/fred/oz335/rridden/asteroids_store/blank_injection/{RUN}"
TEMPLATES = "/fred/oz335/rridden/asteroids_store/injection/templates.csv"
TEMPLATE_HARM = 6
MIN_SRC_POINTS = 100
AVOID_PX = 9.0                 # one PSF stamp (asteroid_photometry.STAMP_SIZE)
EDGE_PX = 5.0                  # keep moved paths this far inside the cube
MAX_TRIES = 200
MIN_KEEP_FRAC = 0.8            # a moved visit is used when at least this fraction of its frames is clear
SNR_BANDS = [(0.1, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 20), (20, 50)]
# SNR_RANGE="lo,hi": per-frame S/N log-uniform over one range instead of the bands (e.g. a faint-only run)
SNR_RANGE = tuple(float(x) for x in os.environ["SNR_RANGE"].split(",")) if os.environ.get("SNR_RANGE") else None
# AMP_RANGE="lo,hi": peak-to-peak amplitude (relative flux) log-uniform over this range, the template's shape kept,
# instead of the template's own amplitude (templates come from bright, large asteroids, whose amplitudes run low)
AMP_RANGE = tuple(float(x) for x in os.environ["AMP_RANGE"].split(",")) if os.environ.get("AMP_RANGE") else None
P_MIN_HR, P_MAX_HR = 1.0, 500.0
BLANK_FRAC, CONST_FRAC = 0.10, 0.05
T_EFF = 480.0                  # s, effective exposure of a 600 s FFI frame (sectors 27-55)
N_BUCKETS = 200
TOL = 0.02
SEED = int(os.environ.get("SEED", 20261014))          # a new run needs its own seed (else the same sources)
NUM_OFFSET = int(os.environ.get("NUM_OFFSET", 0))     # first synthetic number, so runs merge without id clashes
EXCLUDE = os.environ.get("EXCLUDE")                   # objects.csv of earlier runs (comma-separated): sources not reused
SRC_COLUMNS = ["designation", "mjd", "e_flux", "x", "y", "ra", "dec", "r_au", "delta_au", "phase_deg", "mag_expected",
               "sector", "cam", "ccd", "cut", "part", "frame"]
DRIFT_MAX = 0.1   # recovered = phase drift |1/P_rec - 1/P_inj| x baseline below this many cycles (a 2 % tolerance let
                  # periods through whose fold smears over long baselines -- blind_0164: 0.7 % off, 2.1 cycles)
# controls, written down before the first run; check() reports PASS / FAIL
CONTROLS = dict(blank_median_sig=0.15,      # blanks: |median(flux_detrended / e_flux)| below this
                blank_pass_max=0.02,        # blanks: classifier pass rate at most this
                const_ratio=(0.85, 1.15),   # constants: median measured / injected flux (frames with F/e > 2)
                const_pass_max=0.03,        # constants (no rotation): classifier pass rate at most this
                rot_recovery_min=0.80)      # 2-maximum templates, s_f >= 5, period 3-30 h: recovery at least this


def fourier(phase, coef):
    m = np.full_like(phase, coef[0], dtype=float)
    for k in range(1, (len(coef) - 1) // 2 + 1):
        m += coef[2 * k - 1] * np.cos(2 * np.pi * k * phase) + coef[2 * k] * np.sin(2 * np.pi * k * phase)
    return m


def _src_photometry(d):
    df = st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=SRC_COLUMNS)
    return df[np.isfinite(df.e_flux) & (df.e_flux > 0)]


def plan(n_src, n_copy):
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    obj = st.load_objects(p.DATA_ROOT)
    cnt = obj.set_index("designation")["n_points"] if "n_points" in obj.columns else None
    pool = obj.designation[(cnt.reindex(obj.designation).values >= MIN_SRC_POINTS)] if cnt is not None else obj.designation
    if EXCLUDE:
        used = pd.concat([pd.read_csv(f, usecols=["source"]) for f in EXCLUDE.split(",")]).source
        pool = pool[~pool.isin(used)]
    src = list(rng.choice(pool.values, size=min(n_src, len(pool)), replace=False))
    with ProcessPoolExecutor(p.n_available_workers()) as ex:
        phot = pd.concat([d for d in ex.map(_src_photometry, src) if len(d)], ignore_index=True)
    # sorted by cut with small row groups, so each phot task reads only its own cuts' row groups
    phot.sort_values(["sector", "cam", "ccd", "cut", "part"]).to_parquet(f"{OUT}/src_phot.parquet", index=False,
                                                                          row_group_size=200_000)
    ref = phot.groupby("designation").agg(e_ref=("e_flux", "median"), m_ref=("mag_expected", "median"), n_src_points=("mjd", "size"))
    tp = pd.read_csv(TEMPLATES)
    rows = []
    for s in ref.index:
        for c in range(n_copy):
            u = rng.uniform()
            kind = "blank" if u < BLANK_FRAC else "constant" if u < BLANK_FRAC + CONST_FRAC else "rotator"
            lo, hi = SNR_RANGE if SNR_RANGE else SNR_BANDS[rng.integers(len(SNR_BANDS))]
            ti = int(rng.integers(len(tp)))
            num = NUM_OFFSET + len(rows)
            rows.append(dict(syn=f"SYN{num:06d}", num=num, source=s, copy=c, kind=kind,
                             s_frame=float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), template=ti,
                             amp_inj=(float(np.exp(rng.uniform(np.log(AMP_RANGE[0]), np.log(AMP_RANGE[1])))) if AMP_RANGE
                                      else float(tp.amp_rel.iloc[ti])) if kind == "rotator" else 0.0,
                             period_inj_hr=float(np.exp(rng.uniform(np.log(P_MIN_HR), np.log(P_MAX_HR)))),
                             phase0=float(rng.uniform()), e_ref=float(ref.e_ref[s]), m_ref=float(ref.m_ref[s])))
    objs = pd.DataFrame(rows)
    objs.to_csv(f"{OUT}/objects.csv", index=False)
    cuts = phot[["sector", "cam", "ccd", "cut", "part"]].drop_duplicates().sort_values(["sector", "cam", "ccd", "cut", "part"])
    cuts.to_csv(f"{OUT}/cuts.csv", index=False)
    print(f"{len(ref)} sources ({len(phot):,} measured frames), {len(objs)} synthetic objects "
          f"({objs.kind.value_counts().to_dict()}), {len(cuts)} cuts to measure")


_CORNERS = {}


def cut_corner(sector, cam, ccd):
    if (sector, cam, ccd) not in _CORNERS:
        from tessellate.dataprocessor import DataProcessor
        dp = DataProcessor(sector=int(sector), verbose=0, data_path=p.DATA_ROOT)
        _CORNERS[(sector, cam, ccd)] = dp.find_cuts(cam=int(cam), ccd=int(ccd), n=8, plot=False, verbose=0)[0]
    return _CORNERS[(sector, cam, ccd)]


def phot_task(index, total):
    from astropy.io import fits
    from astropy.wcs import WCS
    from tessellate.tools import load_table
    from tessellate.asteroid_photometry import (forced_psf_photometry, match_ephemeris_to_reduced_frames,
                                                detrend_pixel_phase, local_gaia_cat_to_stars, flag_star_contamination)
    cuts = pd.read_csv(f"{OUT}/cuts.csv").iloc[index::total]
    # only this task's cuts: the whole source table is several GB at full size
    import pyarrow.dataset as pds
    expr = None
    for _, c in cuts.iterrows():
        e = ((pds.field("sector") == int(c.sector)) & (pds.field("cam") == int(c.cam)) & (pds.field("ccd") == int(c.ccd))
             & (pds.field("cut") == int(c.cut)) & (pds.field("part") == int(c.part)))
        expr = e if expr is None else expr | e
    src = pds.dataset(f"{OUT}/src_phot.parquet").to_table(filter=expr).to_pandas() if expr is not None else pd.DataFrame()
    objs = pd.read_csv(f"{OUT}/objects.csv")
    tp = pd.read_csv(TEMPLATES)
    tcols = [f"c{j}" for j in range(2 * TEMPLATE_HARM + 1)]
    out_tables, n_vis, n_skip = [], 0, 0
    for _, c in cuts.iterrows():
        sector, cam, ccd, cut, part = (int(c[k]) for k in ("sector", "cam", "ccd", "cut", "part"))
        folder = (f"{p.DATA_ROOT}/Sector{sector}/Cam{cam}/Ccd{ccd}" + (f"/Part{part}" if part else "") + f"/Cut{cut}of64")
        base = f"sector{sector}_cam{cam}_ccd{ccd}_cut{cut}_of64"
        rng = np.random.default_rng([SEED, sector, cam, ccd, cut, part])
        try:
            cube = np.load(f"{folder}/{base}_ReducedFlux.npy")
            ny, nx = cube.shape[1:]
            eph = match_ephemeris_to_reduced_frames(load_table(f"{folder}/asteroids/{base}_Asteroids.csv"),
                                                    np.load(f"{folder}/{base}_Times.npy"))
            by_frame = {f: g[["x", "y"]].to_numpy() for f, g in eph.groupby("frame")}
            corner = cut_corner(sector, cam, ccd)[cut - 1]
            wcs = WCS(fits.open(f"{p.DATA_ROOT}/Sector{sector}/Cam{cam}/Ccd{ccd}/wcs/ref/corrected.fits")[1].header)
            gpath = f"{folder}/local_gaia_cat.csv"
            stars = (local_gaia_cat_to_stars(pd.read_csv(gpath), wcs, corner[0], corner[1]) if os.path.exists(gpath)
                     else pd.DataFrame(columns=["x", "y", "mag"]))
        except Exception as ex:
            print(f"  cut {sector}/{cam}/{ccd}/{cut}/{part}: {type(ex).__name__}: {ex}", flush=True)
            continue
        here = src[(src.sector == sector) & (src.cam == cam) & (src.ccd == ccd) & (src.cut == cut) & (src.part == part)]
        syn_eph = []
        for s, rows in here.groupby("designation"):
            x, y, fr = rows.x.to_numpy(float), rows.y.to_numpy(float), rows.frame.to_numpy(int)
            for _, o in objs[objs.source == s].iterrows():
                best = None
                for _ in range(MAX_TRIES):
                    # centroid kept inside the cut; frames that leave it, or come within AVOID_PX of any predicted
                    # asteroid, are dropped (blank sky stays blank); accepted when >= MIN_KEEP_FRAC remain
                    dx = rng.uniform(EDGE_PX - x.mean(), nx - 1 - EDGE_PX - x.mean())
                    dy = rng.uniform(EDGE_PX - y.mean(), ny - 1 - EDGE_PX - y.mean())
                    xs, ys = x + dx, y + dy
                    keep = (xs >= EDGE_PX) & (xs <= nx - 1 - EDGE_PX) & (ys >= EDGE_PX) & (ys <= ny - 1 - EDGE_PX)
                    for k in np.flatnonzero(keep):
                        q = by_frame.get(fr[k])
                        if q is not None and np.min(np.hypot(q[:, 0] - xs[k], q[:, 1] - ys[k])) <= AVOID_PX:
                            keep[k] = False
                    if best is None or keep.mean() > best[0]:
                        best = (keep.mean(), dx, dy, keep)
                    if keep.mean() >= MIN_KEEP_FRAC:
                        break
                if best is None or best[0] < MIN_KEEP_FRAC:
                    n_skip += 1
                    continue
                frac, dx, dy, keep = best
                n_vis += 1
                syn_eph.append(rows[keep].assign(designation=o.syn, x=x[keep] + dx, y=y[keep] + dy, dx=dx, dy=dy,
                                                 keep_frac=frac))
        if not syn_eph:
            continue
        se = pd.concat(syn_eph, ignore_index=True).rename(columns={"r_au": "r_helio_au", "phase_deg": "phase_angle_deg"})
        psf = forced_psf_photometry(cube, se[["designation", "frame", "x", "y", "mjd"]], sector, cam, ccd, corner[0], corner[1])
        psf = psf.merge(se[["designation", "frame", "mag_expected"]], on=["designation", "frame"], how="left")
        o_ = objs.set_index("syn").loc[psf.designation]
        s_ = np.array([fourier(np.array([((m * 24 / P) + ph) % 1]), tp.loc[t, tcols].values.astype(float))[0]
                       for m, P, ph, t in zip(psf.mjd.values, o_.period_inj_hr.values, o_.phase0.values, o_.template.values)])
        F = (o_.s_frame.values * o_.e_ref.values * 10 ** (-0.4 * (psf.mag_expected.values - o_.m_ref.values))
             * (1 + o_.amp_inj.values * s_))
        F = np.where(o_.kind.values == "blank", 0.0, F)
        shot = np.sqrt(np.clip(F, 0, None) / T_EFF)
        psf["flux_blank"] = psf.flux.values
        psf["flux_inj"] = F
        psf["flux"] = psf.flux.values + F + rng.normal(0, 1, len(F)) * shot
        psf["e_flux"] = np.hypot(psf.e_flux.values, shot)
        psf = detrend_pixel_phase(psf)
        psf = flag_star_contamination(psf, stars, flux_col="flux_detrended")
        tab = st.photometry_table(psf.drop(columns=["mag_expected"]), se, sector, cam, ccd, cut, part, shape=(ny, nx)).to_pandas()
        tab = tab.merge(psf[["designation", "frame", "flux_inj", "flux_blank"]], on=["designation", "frame"], how="left")
        tab = tab.merge(se[["designation", "keep_frac"]].drop_duplicates("designation"), on="designation", how="left")
        tab["bucket"] = (tab.designation.str[3:].astype(int) % N_BUCKETS).astype(int)
        out_tables.append(tab)
        del cube
    if out_tables:
        allt = pd.concat(out_tables, ignore_index=True)
        for b, g in allt.groupby("bucket"):
            os.makedirs(f"{OUT}/phot/bucket={b}", exist_ok=True)
            g.drop(columns=["bucket"]).to_parquet(f"{OUT}/phot/bucket={b}/task{index:04d}.parquet", index=False)
    print(f"task {index}: {len(cuts)} cuts, {n_vis} synthetic visits measured, {n_skip} skipped (< {MIN_KEEP_FRAC:.0%} "
          f"of frames clear)", flush=True)
    # the forked PSF workers and pyarrow's threads abort at interpreter teardown ("terminate called without an
    # active exception") after everything is written; leave without the teardown
    sys.stdout.flush()
    os._exit(0)


def pipe(bucket):
    objs = pd.read_csv(f"{OUT}/objects.csv").set_index("syn")
    files = glob.glob(f"{OUT}/phot/bucket={bucket}/*.parquet")
    if not files:
        print(f"bucket {bucket}: no photometry"); return
    ph = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = []
    for d, g in ph.groupby("designation"):
        o = objs.loc[d]
        info = dict(syn=d, source=o.source, kind=o.kind, s_frame=o.s_frame, template=o.template, amp_inj=o.amp_inj,
                    period_inj_hr=o.period_inj_hr, n_visits_inj=g.groupby(["sector", "cam", "ccd", "cut", "part"]).ngroups)
        sel = (g.flux_inj / g.e_flux) > 2
        info["const_ratio"] = float(np.median(g.flux_detrended[sel] / g.flux_inj[sel])) if sel.sum() >= 10 else np.nan
        info["blank_median_sig"] = float(np.median(g.flux_detrended / g.e_flux))
        try:
            track = p.normalise(p.frames_to_points(p._store_frame(g)))
            if len(track) < p.MIN_POINTS_FOR_PERIODOGRAM:
                rows.append(dict(info, error="too few points")); continue
            row = p.plot_asteroid(d, track, os.devnull)
            pr, P = row.get("period_hr", np.nan), o.period_inj_hr
            rot = o.kind == "rotator" and np.isfinite(pr)
            row.update(info, recovered=bool(rot and abs(pr / P - 1) < TOL),
                       recovered_half=bool(rot and abs(2 * pr / P - 1) < TOL),
                       recovered_double=bool(rot and abs(pr / (2 * P) - 1) < TOL))
            rows.append(row)
        except Exception as ex:
            rows.append(dict(info, error=f"{type(ex).__name__}: {ex}"))
    pd.DataFrame(rows).to_csv(f"{OUT}/pipe_{bucket:03d}.csv", index=False)
    print(f"bucket {bucket}: {len(rows)} objects")


def check():
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(f"{OUT}/pipe_*.csv"))], ignore_index=True)
    tp = pd.read_csv(TEMPLATES)[["template", "n_maxima"]]
    d = d.merge(tp, on="template", how="left")
    ok = d.get("error", pd.Series(np.nan, index=d.index)).isna()
    print(f"{len(d)} objects, {int(ok.sum())} through the pipeline; kinds {d[ok].kind.value_counts().to_dict()}")
    d = d[ok].copy()
    rot_ = (d.kind == "rotator") & np.isfinite(d.period_hr) & np.isfinite(d.baseline_hr)
    for col in ("recovered", "recovered_half", "recovered_double"):
        d[col + "_2pct"] = d[col]
    for col, f in (("recovered", 1.0), ("recovered_half", 0.5), ("recovered_double", 2.0)):
        d[col] = rot_ & (np.abs(1 / d.period_hr - 1 / (d.period_inj_hr * f)) * d.baseline_hr < DRIFT_MAX)
    print(f"labels: phase drift < {DRIFT_MAX} cycle over the baseline (a 2 % tolerance would call "
          f"{d.recovered_2pct.astype(bool).mean():.1%} recovered, coherence {d.recovered.mean():.1%})")
    rel = d.is_reliable.astype(bool)
    b, c, r = d[d.kind == "blank"], d[d.kind == "constant"], d[d.kind == "rotator"]
    res = []
    res.append(("blanks: |median flux/e_flux|", b.blank_median_sig.median(), abs(b.blank_median_sig.median()) < CONTROLS["blank_median_sig"]))
    res.append(("blanks: classifier pass rate", rel[b.index].mean(), rel[b.index].mean() <= CONTROLS["blank_pass_max"]))
    lo, hi = CONTROLS["const_ratio"]
    res.append(("constants: median measured/injected", c.const_ratio.median(), lo <= c.const_ratio.median() <= hi))
    res.append(("constants: classifier pass rate", rel[c.index].mean(), rel[c.index].mean() <= CONTROLS["const_pass_max"]))
    clean = r[(r.n_maxima == 2) & (r.s_frame >= 5) & r.period_inj_hr.between(3, 30)]
    res.append((f"rotators 2-max, s_f>=5, 3-30 h (n={len(clean)}): recovery", clean.recovered.mean(),
                clean.recovered.mean() >= CONTROLS["rot_recovery_min"]))
    for name, val, passed in res:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {val:.3f}")
    r = r.assign(sf_bin=pd.cut(r.s_frame, [0.1, 0.5, 1, 2, 5, 20, 50]), p_bin=pd.cut(r.period_inj_hr, [1, 3, 10, 30, 100, 500]),
                 **{c: r[c].astype(float) for c in ("recovered", "recovered_half", "recovered_double")})
    d.to_csv(f"{OUT}/synthetic_training_set.csv", index=False)
    print(r.groupby("sf_bin", observed=True).agg(n=("recovered", "size"), recovered=("recovered", "mean"),
                                                  half=("recovered_half", "mean"), double=("recovered_double", "mean"),
                                                  passes=("is_reliable", lambda x: x.astype(bool).mean())).round(3).to_string())
    print(r.groupby("p_bin", observed=True).recovered.mean().round(3).to_string())
    print(r.groupby(["sf_bin", "n_maxima"], observed=True).recovered.mean().unstack().round(2).to_string())


def blind(n):
    d = pd.read_csv(f"{OUT}/synthetic_training_set.csv", low_memory=False)
    rng_state = SEED
    n_b, n_c = int(round(BLANK_FRAC * n)), int(round(CONST_FRAC * n))
    rot = d[d.kind == "rotator"].assign(sf_bin=lambda x: pd.cut(x.s_frame, [0.1, 0.5, 1, 2, 5, 20, 50]))
    per = (n - n_b - n_c) // rot.sf_bin.nunique()
    sel = pd.concat([d[d.kind == "blank"].sample(min(n_b, int((d.kind == "blank").sum())), random_state=rng_state),
                     d[d.kind == "constant"].sample(min(n_c, int((d.kind == "constant").sum())), random_state=rng_state)]
                    + [g.sample(min(per, len(g)), random_state=rng_state) for _, g in rot.groupby("sf_bin", observed=True)])
    sel = sel.sample(frac=1, random_state=rng_state).reset_index(drop=True)
    sel["file"] = [f"blind_{i + 1:04d}.png" for i in range(len(sel))]
    os.makedirs(f"{OUT}/blind", exist_ok=True)
    sel[["file", "syn", "source", "kind", "s_frame", "template", "n_maxima", "amp_inj", "period_inj_hr", "recovered",
         "recovered_half", "recovered_double", "period_hr", "reliability_score", "is_reliable"]].to_csv(f"{OUT}/blind_key.csv", index=False)
    for _, r in sel.iterrows():
        b = int(r.syn[3:]) % N_BUCKETS
        g = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{OUT}/phot/bucket={b}/*.parquet")])
        g = g[g.designation == r.syn]
        p.plot_asteroid(r.syn, p.normalise(p.frames_to_points(p._store_frame(g))), f"{OUT}/blind/{r.file}")
    print(f"blind batch: {len(sel)} plots -> {OUT}/blind/, key {OUT}/blind_key.csv")


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "plan":
        plan(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "phot":
        phot_task(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "pipe":
        pipe(int(sys.argv[2]))
    elif a == "check":
        check()
    elif a == "blind":
        blind(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
