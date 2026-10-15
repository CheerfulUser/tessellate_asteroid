"""Round-3 reliability classifier: the deployed U+L recipe (user labels + LCDB re-attached at weight 0.5, train_round1)
with fold_vn added, and the user's faint labels (labels_faint2 / labels_faint3) joined to labels_final. fold_vn is
not in the production report yet, so every training row is recomputed with the pipeline (plot_asteroid, plots off):
the user-labelled rows from bin_range_test.csv, the LCDB rows here.

  feat <index> <total>   recompute summary rows for the LCDB training objects -> asteroids_store/round3_rows/
  train                  CV on the user labels (repeated 5-fold; LCDB rows always in the training folds) for
                         U+L with the pipeline features and with fold_vn; threshold = lowest score with out-of-fold
                         precision >= 90 %; the fold_vn model saved as reliability_classifier_round3.joblib (the
                         chain deploys it, this script does not)
A user label applies only while the recomputed period is within 2 % of the period the user saw; an LCDB label is
whether the recomputed period matches the vetted LCDB period (period_updated).
"""
import glob
import multiprocessing as mp
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import sector_report_pipeline as p
import train_raw_ozstar as tr

S = "/fred/oz335/rridden/asteroids_store"
ROWS = f"{S}/round3_rows"
OUT = "/fred/oz335/TESSdata/asteroid_store/reliability_classifier_round3.joblib"
PREC_TARGET = 0.90
F = list(tr.F)
F3 = F + ["fold_vn"]


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0,
                                          min_samples_leaf=20, random_state=0)


def X(d, feats):
    return d[feats].astype(float).replace([np.inf, -np.inf], np.nan).values


def lcdb_rows():
    """The deployed recipe's LCDB training objects and weights (train_raw_ozstar.build_table, weight x 0.5) with the
    vetted LCDB period; the label is set later from the recomputed period (match within tr.TOL)."""
    t = tr.build_table()
    L = t[t.split == "lcdb"][["designation", "w"]].assign(w=lambda x: x.w * 0.5)
    lc = pd.read_csv(tr.LCDB, low_memory=False, usecols=["designation", "period_updated"]).dropna().drop_duplicates("designation")
    return L.merge(lc, on="designation")


def _one(d):
    from tessellate import asteroid_store as st
    try:
        phot = st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS)
        track = p.normalise(p.frames_to_points(p._store_frame(phot)))
        return p.plot_asteroid(d, track, os.devnull, h_mag=p._worker_h_lookup.get(d),
                               sbdb_entry=p.match_sbdb(d, p._worker_sbdb_by_number, p._worker_sbdb_by_designation))
    except Exception as ex:
        return dict(designation=d, error=f"{type(ex).__name__}: {ex}")


def feat(index, total):
    from tessellate import asteroid_store as st
    os.makedirs(ROWS, exist_ok=True)
    des = sorted(lcdb_rows().designation.unique())[index::total]
    objects = st.load_objects(p.DATA_ROOT)
    objects = objects[objects.designation.isin(des)]
    h = {d: float(x) for d, x in zip(objects.designation, objects.magnitude_H) if np.isfinite(x)}
    sbdb = p.load_sbdb_physical_properties()
    with mp.get_context("fork").Pool(p.n_available_workers(), initializer=p._pool_worker_init, initargs=(h, *sbdb),
                                     maxtasksperchild=10) as pool:
        rows = list(pool.imap_unordered(_one, des))
    pd.DataFrame(rows).to_csv(f"{ROWS}/lcdb_{index:03d}.csv", index=False)
    print(f"task {index}: {len(rows)} LCDB objects")


def user_labels():
    a = pd.read_csv(f"{S}/label_batches/labels_final.csv").drop_duplicates("designation")[["designation", "label_final", "labelled_period_hr", "folder"]]
    out = [a]
    for name in ("faint2", "faint3"):
        b = pd.read_csv(f"{S}/label_batches/labels_{name}.csv")
        out.append(b.assign(label_final=np.where(b.good, "good", "bad"), labelled_period_hr=b.period_hr, folder=name)[
            ["designation", "label_final", "labelled_period_hr", "folder"]])
    return pd.concat(out, ignore_index=True).drop_duplicates("designation")


def train():
    rows_u = pd.read_csv(f"{S}/bin_range_test.csv", low_memory=False).drop_duplicates("designation")
    u = user_labels().merge(rows_u, on="designation")
    keep = (u.period_hr / u.labelled_period_hr - 1).abs() < 0.02
    print(f"user labels {len(u)} with recomputed rows; period still as labelled {int(keep.sum())}")
    u = u[keep].assign(y=(u.label_final == "good").astype(float), w=1.0).reset_index(drop=True)
    L = lcdb_rows()
    rl = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob(f"{ROWS}/lcdb_*.csv")]).drop_duplicates("designation")
    rl = rl[rl.get("error", pd.Series(np.nan, index=rl.index)).isna()]
    L = L.merge(rl, on="designation")
    L = L[np.isfinite(L.period_hr) & ~L.designation.isin(u.designation)]
    L = L.assign(y=tr.match(L.period_hr, L.period_updated).astype(float))
    print(f"LCDB rows with recomputed features: {len(L)}, {L.y.mean():.1%} match the vetted LCDB period")
    yu = u.y.values
    faint = u.folder.isin(["faint2", "faint3"]).values
    cur = u.reliability_score.astype(float).fillna(0).values
    print(f"deployed classifier on these labels: AUC {roc_auc_score(yu, cur):.3f} (faint {roc_auc_score(yu[faint], cur[faint]):.3f})")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    res = {}
    for name, feats in (("U+L pipeline features", F), ("U+L + fold_vn", F3)):
        oof = np.zeros(len(u)); cnt = np.zeros(len(u))
        for tri, tei in cv.split(u, yu):
            fit = pd.concat([u.iloc[tri], L])
            oof[tei] += model().fit(X(fit, feats), fit.y.values, sample_weight=fit.w.values).predict_proba(X(u.iloc[tei], feats))[:, 1]
            cnt[tei] += 1
        oof /= cnt
        th = next((float(t_) for t_ in np.sort(np.unique(oof)) if yu[oof >= t_].mean() >= PREC_TARGET), 1.0)
        sel = oof >= th
        print(f" {name:22s}: CV AUC {roc_auc_score(yu, oof):.3f} (faint {roc_auc_score(yu[faint], oof[faint]):.3f}); "
              f"threshold {th:.3f} -> precision {yu[sel].mean():.1%}, recall {sel[yu == 1].mean():.1%} "
              f"(faint recall {sel[faint & (yu == 1)].mean():.1%}), agreement {(sel == (yu == 1)).mean():.1%}")
        res[name] = (feats, th)
    feats, th = res["U+L + fold_vn"]
    fit = pd.concat([u, L])
    mdl = model().fit(X(fit, feats), fit.y.values, sample_weight=fit.w.values)
    joblib.dump({"model": mdl, "features": feats, "calibrator": None, "threshold": th,
                 "floors": {"n_points": 100, "aov_F": 1.5}, "trained_on": "U+L (labels_final + faint2 + faint3; LCDB w 0.5)",
                 "note": "round 3: fold_vn added; rows recomputed with the pipeline; trained on ozstar"}, OUT)
    print(f"saved {OUT} (threshold {th:.3f}; deployed by the chain, not by this script)")


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "feat":
        feat(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "train":
        train()
