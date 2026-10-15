"""Test the hybrid Vavilov-style period determination (vavilov_periods.py) on the user's labelled objects
(label_batches/labels_final.csv): candidates from a 1-term Lomb-Scargle on the pre-clipped points (the N_CANDIDATES
strongest distinct peaks), decided by multi-harmonic fits. Writes asteroids_store/vavilov_test.csv and prints
  - period agreement with what the user accepted (good labels) and how often bad labels get a different period;
  - how well each statistic separates good from bad labels (AUC);
  - classifier cross-validation on the user labels with the pipeline features alone and with the Vavilov statistics.
    python3 -u vavilov_test.py [N]        # first N labelled objects only (timing test)
"""
import os, sys, time
import numpy as np, pandas as pd
import multiprocessing as mp
import sector_report_pipeline as p
import vavilov_periods as vp
from tessellate import asteroid_store as st

S = "/fred/oz335/rridden/asteroids_store"


def one(d):
    t0 = time.time()
    try:
        tr = p.normalise(p.frames_to_points(p._store_frame(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))))
        tr = tr.sort_values("mjd").reset_index(drop=True)
        tr = tr[p.preclip_mask(tr)] if len(tr) else tr     # positional mask: computed and applied on the sorted frame
        t, y, e, v = tr.mjd.values, tr.rel_flux.values, tr.rel_flux_err.values, tr.visit.values
        if len(t) < p.MIN_POINTS_FOR_PERIODOGRAM:
            return dict(designation=d, error="too few points")
        cands, width = vp.candidates(t, y, e, p.MIN_PERIOD_DAYS, p.MAX_PERIOD_DAYS, p.SAMPLES_PER_PEAK)
        res = vp.determine(t, y, e, v, cands, width)
        return dict(designation=d, seconds=round(time.time() - t0, 1), n_points=len(t), **(res or {"error": "no fit"}))
    except Exception as ex:
        return dict(designation=d, error=f"{type(ex).__name__}: {ex}")


def main():
    lab = pd.read_csv(f"{S}/label_batches/labels_final.csv")
    if len(sys.argv) > 1:
        lab = lab.head(int(sys.argv[1]))
    fn = f"{S}/vavilov_test{'_' + sys.argv[1] if len(sys.argv) > 1 else ''}.csv"
    lab = lab.drop_duplicates("designation")
    if os.environ.get("VAV_REUSE") == "1":          # analysis only, from the saved fits
        return analyse(pd.read_csv(fn).drop_duplicates("designation"))
    rows = []
    # workers recycled every few objects so memory from the largest light curves is returned; partial results
    # written every 50 objects so a failure keeps what finished
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=5) as pool:
        for k, row in enumerate(pool.imap_unordered(one, lab.designation), 1):
            rows.append(row)
            if k % 50 == 0:
                lab.merge(pd.DataFrame(rows), on="designation").to_csv(fn + ".partial", index=False)
                print(f"  {k}/{len(lab)} done", flush=True)
    out = lab.merge(pd.DataFrame(rows), on="designation")
    out.to_csv(fn, index=False)
    analyse(out)


def analyse(out):
    ok = out[out.get("error", pd.Series(np.nan, index=out.index)).isna()] if "error" in out else out
    print(f"{len(out)} objects, {len(ok)} fitted; seconds per object median {out.seconds.median():.1f}, max {out.seconds.max():.1f}")
    if len(ok) < 20:
        print(ok[["designation", "label_final", "labelled_period_hr", "vav_period_hr", "vav_order", "vav_alt_period_hr",
                  "vav_alt_dbic", "vav_n_rivals", "vav_best_rival_dbic", "vav_n_apparitions", "vav_scale_max",
                  "seconds"]].round(3).to_string(index=False))
        return
    r = ok.vav_period_hr / ok.labelled_period_hr
    same = (r - 1).abs() < 0.02
    harm = sum((r / q - 1).abs().lt(0.02) for q in (2, 0.5, 3, 1 / 3, 1.5, 2 / 3)).astype(bool)
    alt_same = (ok.vav_alt_period_hr / ok.labelled_period_hr - 1).abs().lt(0.02)
    good = ok.label_final == "good"
    print(f"good labels: same period {same[good].mean():.1%}, harmonic (2, 3, 3/2 and inverses) {harm[good].mean():.1%}, "
          f"other {(~same & ~harm)[good].mean():.1%} of {int(good.sum())}; accepted period = harmonic alternative "
          f"{(alt_same & ~same)[good].mean():.1%}")
    print(f"bad labels:  same period {same[~good].mean():.1%}, harmonic {harm[~good].mean():.1%}, other {(~same & ~harm)[~good].mean():.1%} of {int((~good).sum())}")
    m = good & harm & ~same
    print(f"good labels where the adopted period is a harmonic: harmonic-alternative dBIC quartiles "
          f"{np.nanpercentile(ok.vav_alt_dbic[m], [25, 50, 75]).round(1) if m.any() else '-'}; where it agrees: "
          f"{np.nanpercentile(ok.vav_alt_dbic[good & same], [25, 50, 75]).round(1)}")
    from sklearn.metrics import roc_auc_score
    y = good.astype(int).values
    for c, sign in [("vav_log10F_flat", 1), ("vav_best_rival_dbic", 1), ("vav_p_beats_rivals", 1), ("vav_n_rivals", -1),
                    ("vav_chi2_red", -1), ("vav_frac_rejected", -1), ("vav_order", 1),
                    ("vav_alt_dbic", 1), ("vav_scale_max", -1)]:
        x = ok[c].astype(float).replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median() if x.notna().any() else 0).values
        print(f"  AUC {c:22s} {roc_auc_score(y, sign * x):.3f}")
    import train_raw_ozstar as tr
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold
    rep = pd.read_csv(tr.REPORT, low_memory=False, usecols=["designation"] + tr.F).drop_duplicates("designation").set_index("designation")
    d = ok.drop(columns=[c for c in rep.columns if c in ok.columns]).join(rep, on="designation")
    V = ["vav_log10F_flat", "vav_best_rival_dbic", "vav_p_beats_rivals", "vav_n_rivals", "vav_alt_dbic", "vav_scale_max",
         "vav_chi2_red", "vav_frac_rejected", "vav_order"]
    for name, feats in [("pipeline features", tr.F), ("pipeline + Vavilov", tr.F + V), ("Vavilov only", V)]:
        X = d[feats].astype(float).replace([np.inf, -np.inf], np.nan).values
        oof, cnt = np.zeros(len(d)), np.zeros(len(d))
        for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(X, y):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0,
                                               min_samples_leaf=20, random_state=0).fit(X[a], y[a])
            oof[b] += m.predict_proba(X[b])[:, 1]; cnt[b] += 1
        print(f"  classifier CV on user labels, {name:20s}: AUC {roc_auc_score(y, oof / cnt):.3f}")


if __name__ == "__main__":
    main()
