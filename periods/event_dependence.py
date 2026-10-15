"""Does a period rest on one short stretch of data? For the pipeline's cleaned points (pre-clipped, per-visit
normalised) and its adopted period: the 2-term Lomb-Scargle power at the adopted frequency, then the same with each
CHUNK_D-day chunk removed in turn (chunks with >= MIN_CHUNK_PTS points); max_<n>day_power_drop = the largest fractional
drop. A real period is built from many cycles on many days (removing one day changes the power a little); a period made
by one event (e.g. a bright artefact) collapses when that day is removed.

  test <index> <total>   for XY7, Sapientia, the LCDB holdout (lcdb_real_test) and the user-labelled objects
                         -> asteroids_store/event_dependence/test_<index>.csv
  report                 separation: LCDB-confirmed vs contradicted, user good vs bad, by amp_snr; the two examples
Runs on ozstar.
"""
import glob
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

import sector_report_pipeline as p

S = "/fred/oz335/rridden/asteroids_store"
OUT = f"{S}/event_dependence"
CHUNKS_D = (1.0, 3.0)
MIN_CHUNK_PTS = 5
SCAN = 0.08829


def power_drops(t, y, e, period_hr):
    f = 24 / period_hr
    full = float(LombScargle(t, y, e, nterms=2).power(np.array([f]))[0])
    out = {"lc_power_full": full}
    for c in CHUNKS_D:
        idx = np.floor((t - t.min()) / c).astype(int)
        drops = []
        for k in np.unique(idx):
            m = idx == k
            if m.sum() < MIN_CHUNK_PTS or (~m).sum() < 30:
                continue
            pk = float(LombScargle(t[~m], y[~m], e[~m], nterms=2).power(np.array([f]))[0])
            drops.append(1 - pk / full if full > 0 else np.nan)
        out[f"max_{int(c)}day_power_drop"] = float(np.nanmax(drops)) if drops else np.nan
        out[f"n_{int(c)}day_chunks"] = len(drops)
    return out


def from_store(args):
    d, period_hr = args
    from tessellate import asteroid_store as st
    try:
        tr = p.normalise(p.frames_to_points(p._store_frame(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))))
        tr = tr.sort_values("mjd").reset_index(drop=True)
        tr = tr[p.preclip_mask(tr)]
        return dict(designation=d, **power_drops(tr.mjd.values, tr.rel_flux.values, tr.rel_flux_err.values, period_hr))
    except Exception as ex:
        return dict(designation=d, error=f"{type(ex).__name__}: {ex}")


def targets():
    import lcdb_real_test as lt
    import train_round3_ozstar as r3
    rep = pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False,
                      usecols=["designation", "period_hr", "amp_snr", "n_phase_coverings"]).drop_duplicates("designation")
    T = lt.targets().merge(rep, on="designation").assign(set="lcdb_holdout")
    T["truth"] = np.any([((T.period_hr / T.Period) / q - 1).abs() < SCAN for q in (1, 2, 0.5)], axis=0)
    U = r3.user_labels()[["designation", "label_final"]].merge(rep, on="designation").assign(set="user")
    U["truth"] = U.label_final == "good"
    X = rep[rep.designation.isin(["(275) Sapientia", "(651174) 2012 XY7"])].assign(set="example", truth=False)
    return pd.concat([T, U, X], ignore_index=True)[["designation", "period_hr", "amp_snr", "n_phase_coverings", "set", "truth"]]


def test(index, total):
    os.makedirs(OUT, exist_ok=True)
    t = targets()
    t = t[t.period_hr.notna()].iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=10) as pool:
        r = pd.DataFrame(pool.map(from_store, list(zip(t.designation, t.period_hr))))
    t.merge(r, on="designation").to_csv(f"{OUT}/test_{index:03d}.csv", index=False)
    print(f"task {index}: {len(r)} objects")


def report():
    from sklearn.metrics import roc_auc_score
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/test_*.csv")], ignore_index=True)
    d = d[d.get("error", pd.Series(np.nan, index=d.index)).isna()]
    print(d[d.set == "example"][["designation", "period_hr", "max_1day_power_drop", "max_3day_power_drop", "n_1day_chunks", "n_phase_coverings"]].round(3).to_string(index=False))
    for s, g in d[d.set != "example"].groupby("set"):
        y = g.truth.astype(bool)
        print(f"\n{s} ({len(g)}, {y.mean():.0%} positive): median 1-day / 3-day drop, positive {g.max_1day_power_drop[y].median():.3f} / "
              f"{g.max_3day_power_drop[y].median():.3f}; negative {g.max_1day_power_drop[~y].median():.3f} / {g.max_3day_power_drop[~y].median():.3f}")
        for c in ("max_1day_power_drop", "max_3day_power_drop"):
            x = g[c].fillna(g[c].median())
            print(f"   AUC (low drop = positive) {c}: {roc_auc_score(y, -x):.3f}; by amp_snr "
                  + ", ".join(f"{lab}: {roc_auc_score(y[m], -x[m]):.3f}" for lab, m in (("<1", g.amp_snr < 1), (">=1", g.amp_snr >= 1)) if y[m].nunique() == 2))
        hi = g.max_1day_power_drop > 0.5
        print(f"   objects with a 1-day drop > 0.5: {hi.mean():.1%}; positive among them {y[hi].mean():.0%} (vs {y.mean():.0%} overall)")


def syn(run, bucket):
    """Synthetic objects of one photometry bucket, at the pipeline's adopted period."""
    os.makedirs(OUT, exist_ok=True)
    B = f"{S}/blank_injection"
    ts = pd.read_csv(f"{B}/synthetic_training_set_all.csv", low_memory=False, usecols=["syn", "run", "period_hr", "error"])
    ts = ts[(ts.run == run) & ts.error.isna() & ts.period_hr.notna()].set_index("syn")
    files = glob.glob(f"{B}/{run}/phot/bucket={bucket}/*.parquet")
    rows = []
    if files:
        ph = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        for d, g in ph.groupby("designation"):
            if d not in ts.index:
                continue
            try:
                tr = p.normalise(p.frames_to_points(p._store_frame(g))).sort_values("mjd").reset_index(drop=True)
                tr = tr[p.preclip_mask(tr)]
                rows.append(dict(syn=d, **power_drops(tr.mjd.values, tr.rel_flux.values, tr.rel_flux_err.values, float(ts.loc[d, "period_hr"]))))
            except Exception as ex:
                print(f"  {d}: {type(ex).__name__}: {ex}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/syn_{run}_{bucket:03d}.csv", index=False)
    print(f"{run} {bucket}: {len(rows)}")


def lcdb(index, total):
    """The LCDB training objects (round3_rows), at their recomputed period."""
    os.makedirs(OUT, exist_ok=True)
    import train_round3_ozstar as r3
    rl = pd.concat([pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("designation", "period_hr", "error"))
                    for f in glob.glob(f"{r3.ROWS}/lcdb_*.csv")]).drop_duplicates("designation")
    rl = rl[rl.get("error", pd.Series(np.nan, index=rl.index)).isna() & rl.period_hr.notna()].iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=10) as pool:
        r = pd.DataFrame(pool.map(from_store, list(zip(rl.designation, rl.period_hr))))
    r.to_csv(f"{OUT}/lcdbtrain_{index:03d}.csv", index=False)
    print(f"lcdb task {index}: {len(r)}")


CAND_MIN_SCORE = 0.15


def candidates():
    """Report objects that could pass the C + L + event-dependence model: C + L score >= CAND_MIN_SCORE (the drops can
    only matter near the threshold) or currently reliable (for what it would drop)."""
    import joblib
    import synthetic_feature_test as sft
    b = joblib.load("/fred/oz335/TESSdata/asteroid_store/reliability_classifier_C_L.joblib")
    rep = sft.derive(pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False))
    rep = rep[rep.period_hr.notna()].drop_duplicates("designation")
    sc = b["model"].predict_proba(sft.X(rep, b["features"]))[:, 1]
    keep = (sc >= CAND_MIN_SCORE) | rep.is_reliable.astype(str).str.lower().eq("true").values
    have = set()
    for f in glob.glob(f"{OUT}/test_*.csv") + glob.glob(f"{OUT}/lcdbtrain_*.csv"):
        have |= set(pd.read_csv(f, usecols=["designation"]).designation)
    c = rep[keep & ~rep.designation.isin(have).values][["designation", "period_hr"]]
    c.to_csv(f"{OUT}/candidates.csv", index=False)
    print(f"{int(keep.sum()):,} candidates ({len(c):,} still to compute) -> {OUT}/candidates.csv")


def cand(index, total):
    c = pd.read_csv(f"{OUT}/candidates.csv").iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=20) as pool:
        r = pd.DataFrame(pool.map(from_store, list(zip(c.designation, c.period_hr))))
    r.to_csv(f"{OUT}/cand_{index:04d}.csv", index=False)
    print(f"cand task {index}: {len(r)}")


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "syn":
        syn(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "candidates":
        candidates()
    elif sys.argv[1] == "cand":
        cand(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "lcdb":
        lcdb(int(sys.argv[2]), int(sys.argv[3]))
    else:
        report()
