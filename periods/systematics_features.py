"""Features that separate a period made by systematics from a real (slow) rotation (agreed 2026-10-15). Computed
on the pipeline's frames (raw point mode: one point per frame; near-bright-star frames dropped, expected magnitude
divided out per visit, flux normalised by the visit median, the pipeline's pre-clip) at the adopted period P1.
Every fit has one free offset per visit, as the normalisation does; chi2_0 = offsets only.

  bkg_r2         variance explained by the background: one global coefficient on the visit-centred background
                 (in units of the visit's median flux), 1 - chi2(offsets + bkg) / chi2_0
  model_r2       the same for a FOURIER_K-harmonic model at P1
  model_gain     what the P1 model explains on top of the background, (chi2(bkg) - chi2(bkg + model)) / chi2_0
  bkg_share      bkg_r2 / r2(bkg + model): the share of the explained variance the background alone accounts for
  scale_dchi2    chi2(rotation as a constant fraction) - chi2(the same signal scaled by g = 10^(0.4 (mag - median
                 mag))) per point: > 0 favours an additive error in flux (larger fractional change when fainter)
  mag_range      the expected-magnitude range over the track (scale_dchi2 is only informative when it is large)
  pos_r2         variance explained by a quadratic in the cut pixel position (global coefficients)
  pos_gain       the P1 model on top of position, as model_gain
  orbit_near     min over k = 1..4 of |ln(P1 / (TESS_ORBIT_HR / k))| (scattered light repeats each spacecraft orbit)

  test <i> <n>       user labels, the LCDB holdout, the named failures -> asteroids_store/systematics/test_<i>.csv
  syn <run> <b>      the synthetic objects of one photometry bucket   -> systematics/syn_<run>_<b>.csv
  synall <i>         i -> (RUNS[i // 200], i % 200), for one array over every run
  lcdb <i> <n>       the LCDB training rows                            -> systematics/lcdbtrain_<i>.csv
  cand <i> <n>       the catalogue objects that can pass (event_dependence/candidates.csv) -> systematics/cand_<i>.csv
  report             AUC of each feature, all objects and the hard subset (period >= 50 h or 0 phase coverings)
Runs on ozstar.
"""
import glob
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

import sector_report_pipeline as p

S = "/fred/oz335/rridden/asteroids_store"
OUT = f"{S}/systematics"
FOURIER_K = 4
TESS_ORBIT_HR = 13.7 * 24
EXTRA = ["background", "x", "y"]
FEATS = ["bkg_r2", "model_r2", "model_gain", "bkg_share", "scale_dchi2", "mag_range", "pos_r2", "pos_gain", "orbit_near"]
NAMED = ["(275) Sapientia", "(651174) 2012 XY7", "(75275) 1999 XG19", "(30545) 2001 OT35", "(22056) 2000 AU31"]
SCAN = 0.08829


def frames(phot):
    """Store photometry -> frame-level points as the pipeline sees them, keeping background, position, magnitude."""
    near = (phot["contamination_flags"].to_numpy() & p.NEAR_BRIGHT_STAR_BIT) > 0
    df = phot.loc[~near].reset_index(drop=True).copy()
    key = ["sector", "cam", "ccd", "cut", "part"]
    df["visit"] = df[key].astype(str).agg("_".join, axis=1)
    mag = df.mag_expected.astype(float)
    ref = df.groupby("visit").mag_expected.transform("median").astype(float)
    corr = np.where(mag.notna() & ref.notna(), 10 ** (0.4 * (mag - ref)), 1.0)
    f = df.flux_detrended.astype(float) * corr
    e = df.e_flux.astype(float) * corr
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & np.isfinite(df.background.astype(float))
    df, f, e = df[ok].reset_index(drop=True), f[ok].to_numpy(), e[ok].to_numpy()
    med = pd.Series(f).groupby(df.visit.values).transform("median").to_numpy()
    good = med > 0
    df, f, e, med = df[good].reset_index(drop=True), f[good], e[good], med[good]
    out = pd.DataFrame(dict(visit=df.visit.values, mjd=df.mjd.astype(float).values, rel_flux=f / med, rel_flux_err=e / med,
                            bkg=df.background.astype(float).values, x=df.x.astype(float).values, y=df.y.astype(float).values,
                            mag=df.mag_expected.astype(float).values, vmed=med))
    out = out[p.preclip_mask(out)].sort_values("mjd").reset_index(drop=True)
    return out


def chi2(cols, y, w):
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    return float(np.sum(((y - A @ c) * w) ** 2))


def features(tr, P1):
    y, w = tr.rel_flux.to_numpy(), 1 / tr.rel_flux_err.to_numpy()
    vis = pd.factorize(tr.visit)[0]
    off = [(vis == k).astype(float) for k in range(vis.max() + 1)]
    ph = (tr.mjd.to_numpy() * 24 / P1) % 1
    four = []
    for j in range(1, FOURIER_K + 1):
        four += [np.cos(2 * np.pi * j * ph), np.sin(2 * np.pi * j * ph)]
    bn = (tr.bkg - tr.groupby("visit").bkg.transform("median")).to_numpy() / tr.vmed.to_numpy()
    x0 = tr.x - tr.x.mean(); y0 = tr.y - tr.y.mean()
    sx, sy = max(x0.std(), 1e-6), max(y0.std(), 1e-6)
    x0, y0 = (x0 / sx).to_numpy(), (y0 / sy).to_numpy()
    pos = [x0, y0, x0 ** 2, y0 ** 2, x0 * y0]
    c0 = chi2(off, y, w)
    cb = chi2(off + [bn], y, w)
    cm = chi2(off + four, y, w)
    cbm = chi2(off + [bn] + four, y, w)
    cp = chi2(off + pos, y, w)
    cpm = chi2(off + pos + four, y, w)
    g = 10 ** (0.4 * (tr.mag.to_numpy() - np.nanmedian(tr.mag.to_numpy())))
    g = np.where(np.isfinite(g), g, 1.0)
    cs = chi2(off + [g * b for b in four], y, w)
    r2b, r2m, r2bm = 1 - cb / c0, 1 - cm / c0, 1 - cbm / c0
    orbit = min(abs(np.log(P1 / (TESS_ORBIT_HR / k))) for k in range(1, 5))
    return dict(bkg_r2=r2b, model_r2=r2m, model_gain=(cb - cbm) / c0, bkg_share=r2b / r2bm if r2bm > 0 else np.nan,
                scale_dchi2=(cm - cs) / len(y), mag_range=float(np.nanmax(tr.mag) - np.nanmin(tr.mag)),
                pos_r2=1 - cp / c0, pos_gain=(cp - cpm) / c0, orbit_near=orbit, n_frames=len(y))


def from_store(args):
    d, P1 = args
    from tessellate import asteroid_store as st
    try:
        tr = frames(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS + EXTRA))
        return dict(designation=d, **features(tr, P1))
    except Exception as ex:
        return dict(designation=d, error=f"{type(ex).__name__}: {ex}")


def targets():
    import event_dependence as ev
    return ev.targets()           # LCDB holdout (truth = reachable), user labels (truth = good), the two examples


def test(index, total):
    os.makedirs(OUT, exist_ok=True)
    t = targets()
    rep = pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False, usecols=["designation", "period_hr", "amp_snr", "n_phase_coverings"]).drop_duplicates("designation")
    x = rep[rep.designation.isin(NAMED)].assign(set="example", truth=False)
    t = pd.concat([t[t.set != "example"], x], ignore_index=True)
    t = t[t.period_hr.notna()].iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=20) as pool:
        r = pd.DataFrame(pool.map(from_store, list(zip(t.designation, t.period_hr))))
    t.merge(r, on="designation").to_csv(f"{OUT}/test_{index:03d}.csv", index=False)
    print(f"task {index}: {len(r)}")


def syn(run, bucket):
    os.makedirs(OUT, exist_ok=True)
    B = f"{S}/blank_injection"
    table = "synthetic_training_set_all_detect_cut_art.csv"
    d = pd.read_csv(f"{B}/{table}", low_memory=False, usecols=["syn", "run", "kind", "reach", "period_hr", "error"])
    d = d[(d.run == run) & d.error.isna() & d.period_hr.notna()].set_index("syn")
    ph = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{B}/{run}/phot/bucket={bucket}/*.parquet")], ignore_index=True)
    rows = []
    for s, g in ph.groupby("designation"):
        if s not in d.index:
            continue
        try:
            rows.append(dict(syn=s, **features(frames(g), float(d.loc[s, "period_hr"]))))
        except Exception as ex:
            print(f"  {s}: {type(ex).__name__}: {ex}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/syn_{run}_{bucket:03d}.csv", index=False)
    print(f"{run} {bucket}: {len(rows)}")


RUNS = ("full", "full2", "amp", "faint", "cut1", "art1")


def lcdb(index, total):
    """The LCDB training objects (round3_rows), at their recomputed period."""
    import train_round3_ozstar as r3
    os.makedirs(OUT, exist_ok=True)
    rl = pd.concat([pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("designation", "period_hr", "error"))
                    for f in glob.glob(f"{r3.ROWS}/lcdb_*.csv")]).drop_duplicates("designation")
    rl = rl[rl.get("error", pd.Series(np.nan, index=rl.index)).isna() & rl.period_hr.notna()].iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=20) as pool:
        r = pd.DataFrame(pool.map(from_store, list(zip(rl.designation, rl.period_hr))))
    r.to_csv(f"{OUT}/lcdbtrain_{index:03d}.csv", index=False)
    print(f"lcdb task {index}: {len(r)}")


def cand(index, total):
    """The catalogue objects that can pass (event_dependence candidates.csv, as for the drops)."""
    os.makedirs(OUT, exist_ok=True)
    c = pd.read_csv("/fred/oz335/rridden/asteroids_store/event_dependence/candidates.csv").iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=20) as pool:
        r = pd.DataFrame(pool.map(from_store, list(zip(c.designation, c.period_hr))))
    r.to_csv(f"{OUT}/cand_{index:04d}.csv", index=False)
    print(f"cand task {index}: {len(r)}")


def report():
    from sklearn.metrics import roc_auc_score
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/test_*.csv")], ignore_index=True)
    d = d[d.get("error", pd.Series(np.nan, index=d.index)).isna()]
    pd.set_option("display.width", 250)
    print(d[d.set == "example"][["designation", "period_hr", "n_phase_coverings"] + FEATS].round(3).to_string(index=False))
    hard = (d.period_hr >= 50) | (d.n_phase_coverings == 0)
    rows = []
    for s, g in d[d.set != "example"].groupby("set"):
        for lab, m in (("all", np.ones(len(g), bool)), ("hard", hard[g.index].values)):
            y = g.truth.astype(bool).values[m]
            if len(np.unique(y)) < 2:
                continue
            r = dict(set=s, subset=lab, n=int(m.sum()), positive=float(y.mean()))
            for f in FEATS:
                x = g[f].values[m].astype(float)
                x = np.where(np.isfinite(x), x, np.nanmedian(x))
                a = roc_auc_score(y, x)
                r[f] = max(a, 1 - a) * (1 if a >= 0.5 else -1)     # sign: + higher = good, - lower = good
            rows.append(r)
    print("\nAUC per feature (sign: + higher value = good / reachable, - lower value = good)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    sy = glob.glob(f"{OUT}/syn_*.csv")
    if sy:
        s = pd.concat([pd.read_csv(f) for f in sy], ignore_index=True)
        B = f"{S}/blank_injection/synthetic_training_set_all_detect_cut_art.csv"
        lab = pd.read_csv(B, low_memory=False, usecols=["syn", "run", "kind", "reach", "art"]).merge(s, on="syn")
        print(f"\nsynthetic ({len(lab)}): feature medians by group")
        lab["group"] = np.where(lab.kind != "rotator", "blank/constant", np.where(lab.reach.astype(bool), "rotator reached", "rotator missed"))
        lab["group"] += np.where(lab.run.astype(str).str.startswith("art"), " +artefacts", "")
        print(lab.groupby("group")[FEATS].median().round(3).to_string())
        print("\nreal (all test objects) medians: " + ", ".join(f"{f} {d[f].median():.3f}" for f in FEATS))


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "test":
        test(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "syn":
        syn(sys.argv[2], int(sys.argv[3]))
    elif a == "synall":                               # array index -> (run, bucket) over RUNS x 200 buckets
        i = int(sys.argv[2])
        syn(RUNS[i // 200], i % 200)
    elif a == "lcdb":
        lcdb(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "cand":
        cand(int(sys.argv[2]), int(sys.argv[3]))
    else:
        report()
