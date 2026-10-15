"""Real-data truth for the classifiers, independent of the user's eye and of the synthetic set: LCDB U = 3 objects whose
period is LCDB's own (period_source LCDB_native -- a category the deployed classifier's LCDB training rows never
included), not in the user labels, with a TESS period. Truth: the TESS period is reachable from the LCDB period
(within +-SCAN of P_LCDB x {1, 2, 1/2}), the same question the synthetic label asks.

  targets                the object list (once) -> asteroids_store/lcdb_test_targets.csv
  feat <index> <total>   pipeline summary rows for these objects (plot_asteroid, plots off) -> asteroids_store/lcdb_test_rows/
  eval                   AUC, precision and recall at each model's own threshold, overall and by amp_snr: the deployed
                         round-3 classifier; synthetic A, B (soft labels, synthetic only) and C (B + the user labels),
                         refit on the full synthetic set with thresholds from synthetic grouped CV (synthetic_feature_test)
Runs on ozstar.
"""
import glob
import multiprocessing as mp
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import sector_report_pipeline as p
import train_round3_ozstar as r3
import synthetic_feature_test as sft

S = "/fred/oz335/rridden/asteroids_store"
ROWS = f"{S}/lcdb_test_rows"
SCAN = 0.08829


TARGETS = f"{S}/lcdb_test_targets.csv"


def make_targets():
    """Computed once (building the training-row list rewrites raw_training_table.csv, so not per task)."""
    lc = pd.read_csv("/fred/oz335/rridden/asteroids/lcdb_updated.csv", low_memory=False)
    lc["designation"] = lc.designation.where(lc.designation.notna(), "(" + lc.Number.astype("Int64").astype(str) + ") " + lc.Name.astype(str))
    lc = lc[(lc.U_clean == 3) & (lc.period_source == "LCDB_native")][["designation", "Period"]].drop_duplicates("designation")
    rep = pd.read_csv(p.SUMMARY_PATH, low_memory=False, usecols=["designation", "period_hr"]).drop_duplicates("designation")
    t = lc.merge(rep[rep.period_hr.notna()][["designation"]], on="designation")
    used = set(r3.user_labels().designation) | set(r3.lcdb_rows().designation)
    t = t[~t.designation.isin(used)].reset_index(drop=True)
    t.to_csv(TARGETS, index=False)
    print(f"{len(t)} targets -> {TARGETS}")


def targets():
    return pd.read_csv(TARGETS)


def feat(index, total):
    from tessellate import asteroid_store as st
    os.makedirs(ROWS, exist_ok=True)
    des = list(targets().designation)[index::total]
    objects = st.load_objects(p.DATA_ROOT)
    objects = objects[objects.designation.isin(des)]
    h = {d: float(x) for d, x in zip(objects.designation, objects.magnitude_H) if np.isfinite(x)}
    sbdb = p.load_sbdb_physical_properties()
    with mp.get_context("fork").Pool(p.n_available_workers(), initializer=p._pool_worker_init, initargs=(h, *sbdb),
                                     maxtasksperchild=10) as pool:
        rows = list(pool.imap_unordered(r3._one, des))
    pd.DataFrame(rows).to_csv(f"{ROWS}/rows_{index:03d}.csv", index=False)
    print(f"task {index}: {len(rows)} objects")


def evaluate():
    t = targets()
    rows = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob(f"{ROWS}/rows_*.csv")]).drop_duplicates("designation")
    rows = rows[rows.get("error", pd.Series(np.nan, index=rows.index)).isna() & rows.period_hr.notna()]
    d = sft.derive(t.merge(rows, on="designation"))
    ratio = d.period_hr / d.Period
    y = np.any([(ratio / q - 1).abs() < SCAN for q in (1, 2, 0.5)], axis=0)
    print(f"LCDB U=3 native objects, unseen by every model: {len(d)}; TESS period reachable from LCDB {y.mean():.1%}")

    syn = sft.derive(pd.read_csv(f"{sft.BI}/synthetic_training_set_all_detect.csv", low_memory=False))
    syn = syn[syn.error.isna() & syn.period_hr.notna()].reset_index(drop=True)
    w_pos, w_neg = sft.soft_weights(syn)
    lab = pd.concat([pd.read_csv(f"{S}/label_batches/labels_final.csv")[["designation", "label_final"]]]
                    + [pd.read_csv(f"{S}/label_batches/labels_{n}.csv").assign(label_final=lambda x: np.where(x.good, "good", "bad"))[["designation", "label_final"]]
                       for n in ("faint2", "faint3")]).drop_duplicates("designation")
    corr = pd.read_csv(f"{S}/label_batches/label_corrections.csv")
    h = sft.derive(pd.read_csv(f"{S}/bin_range_test.csv", low_memory=False).drop_duplicates("designation").merge(lab, on="designation"))
    h = h[h.period_hr.notna()].reset_index(drop=True)
    h.loc[h.designation.isin(corr.designation), "label_final"] = "good"
    yh = (h.label_final == "good").values.astype(float)
    wu = (w_pos.sum() + w_neg.sum()) / len(h)
    base = list(dict.fromkeys(list(joblib.load(f"{sft.TD}/reliability_classifier.joblib")["features"]) + ["fold_vn"]))
    folds = list(GroupKFold(n_splits=5).split(syn, groups=syn.source))
    scores, ths = {}, {}
    dep = joblib.load(f"{sft.TD}/reliability_classifier.joblib")
    scores["deployed round 3"] = dep["model"].predict_proba(sft.X(d, dep["features"]))[:, 1]
    ths["deployed round 3"] = dep["threshold"]
    for name, feats, hybrid in (("synthetic A", base, False), ("synthetic B", base + sft.EXTRA, False), ("C = B + user labels", base + sft.EXTRA, True)):
        Xs, Xu = sft.X(syn, feats), sft.X(h, feats)
        extra = (Xu, yh, np.full(len(h), wu)) if hybrid else ()
        oof = np.zeros(len(syn))
        for a, t_ in folds:
            oof[t_] = sft.fit(Xs[a], w_pos[a], w_neg[a], *extra).predict_proba(Xs[t_])[:, 1]
        _, ths[name], _, _, _ = sft.weighted_eval(oof, w_pos, w_neg)
        scores[name] = sft.fit(Xs, w_pos, w_neg, *extra).predict_proba(sft.X(d, feats))[:, 1]
    band = pd.cut(d.amp_snr, [0, 1, 2, 4, 1e9])
    print(f"\n{'model':22s} {'AUC':>6s} {'thresh':>7s} {'precision':>9s} {'recall':>7s}   precision / recall by amp_snr (0-1, 1-2, 2-4, >4)")
    for name, s in scores.items():
        sel = s >= ths[name]
        by = []
        for b in band.cat.categories:
            m = (band == b).values
            pr_ = y[m & sel].mean() if (m & sel).any() else np.nan
            rc_ = sel[m & y].mean() if (m & y).any() else np.nan
            by.append(f"{pr_:.0%}/{rc_:.0%}")
        print(f"{name:22s} {roc_auc_score(y, s):6.3f} {ths[name]:7.3f} {y[sel].mean():9.1%} {sel[y].mean():7.1%}   " + "  ".join(by))
    print("objects per amp_snr band:", band.value_counts().sort_index().to_dict(), "; reachable per band:",
          {str(b): round(float(y[(band == b).values].mean()), 3) for b in band.cat.categories})


if __name__ == "__main__":
    if sys.argv[1] == "targets":
        make_targets()
    elif sys.argv[1] == "feat":
        feat(int(sys.argv[2]), int(sys.argv[3]))
    else:
        evaluate()
