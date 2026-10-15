"""Retrain the reliability classifier on the user's visual labels of the raw-frame store run (label_batches,
labels_final.csv: 599 objects judged on their plots, 2026-10-09), using the features the pipeline already writes.

Variants (all judged on the user's labels by repeated 5-fold cross-validation, since those are the reference):
  U      user labels only
  U+L    + LCDB (lcdb_updated.csv v4 sources) re-attached to the new period through period_updated, weight 0.5
  U+L+H  + the old hand labels re-attached as in train_raw_ozstar.py, weight 1
  ...+A  + the cross-sector agreement labels (label_v2: all sectors agree and match pooled = 1, two confident
         sectors disagree = 0), weight 0.5 -- almost all positive, so the user's bad labels supply the negatives
Extra (L, H) rows are always in the training folds, never scored. The current deployed classifier's score is
reported on the same labels for comparison. A user label applies only while the report period is within 2% of the
period the user saw. Threshold: lowest score with out-of-fold precision >= PREC_TARGET.

    python3 -u train_round1_ozstar.py
"""
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
import train_raw_ozstar as tr

S = "/fred/oz335/rridden/asteroids_store"
LABELS = f"{S}/label_batches/labels_final.csv"
OUT = "/fred/oz335/TESSdata/asteroid_store/reliability_classifier_round1.joblib"
PREC_TARGET = 0.90
F = tr.F


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0,
                                          min_samples_leaf=20, random_state=0)


def X(d):
    return d[F].astype(float).replace([np.inf, -np.inf], np.nan).values


def main():
    rep = pd.read_csv(tr.REPORT, low_memory=False, usecols=["designation", "period_hr", "reliability_score"] + F)
    rep = rep.drop_duplicates("designation").set_index("designation")
    u = pd.read_csv(LABELS).join(rep, on="designation")
    keep = (u.period_hr / u.labelled_period_hr - 1).abs() < 0.02
    print(f"user labels {len(u)}; period still as labelled {int(keep.sum())}")
    u = u[keep].assign(y=(u.label_final == "good").astype(float), w=1.0, source="user")
    t = tr.build_table()                                   # hand (dev/test), lcdb, agree on the current report
    extra = {"L": t[t.split == "lcdb"].assign(w=t.w * 0.5), "H": t[t.split.isin(["dev", "test"])],
             "A": t[t.split == "agree"]}
    for k in extra:
        extra[k] = extra[k][~extra[k].designation.isin(u.designation)]
    yu = u.y.values
    cur = u.reliability_score.values
    print(f"current classifier on user labels: AUC {roc_auc_score(yu, cur):.3f}, agreement at 0.564 "
          f"{((cur >= 0.564) == (yu == 1)).mean():.1%}")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    results = {}
    for name, add in [("U", []), ("U+L", ["L"]), ("U+L+H", ["L", "H"]), ("U+A", ["A"]), ("U+L+A", ["L", "A"]),
                      ("U+L+H+A", ["L", "H", "A"])]:
        rest = pd.concat([extra[k] for k in add]) if add else u.iloc[:0]
        oof = np.zeros(len(u)); cnt = np.zeros(len(u))
        for tri, tei in cv.split(u, yu):
            fit = pd.concat([u.iloc[tri], rest])
            oof[tei] += model().fit(X(fit), fit.y.values, sample_weight=fit.w.values).predict_proba(X(u.iloc[tei]))[:, 1]
            cnt[tei] += 1
        oof /= cnt
        auc = roc_auc_score(yu, oof)
        th = next((float(t_) for t_ in np.sort(np.unique(oof)) if yu[oof >= t_].mean() >= PREC_TARGET), 1.0)
        p = oof >= th
        prec, rec = yu[p].mean() if p.any() else np.nan, p[yu == 1].mean()
        by = {f: f"{((oof[u.folder == f] >= th) == (yu[u.folder == f] == 1)).mean():.0%}" for f in u.folder.unique()}
        print(f" {name:6s} ({len(u) + len(rest):,} rows): CV AUC {auc:.3f}; threshold {th:.3f} -> precision {prec:.1%}, "
              f"recall {rec:.1%}, agreement {(p == (yu == 1)).mean():.1%}; by folder {by}")
        results[name] = (auc, th, rest)
    best = max(results, key=lambda k: results[k][0])
    auc, th, rest = results[best]
    fit = pd.concat([u, rest])
    mdl = model().fit(X(fit), fit.y.values, sample_weight=fit.w.values)
    joblib.dump({"model": mdl, "features": F, "calibrator": None, "threshold": th,
                 "floors": {"n_points": 100, "aov_F": 1.5}, "trained_on": best,
                 "note": "user visual labels round 1 (raw-frame store, pre-clip run) + extras; trained on ozstar"}, OUT)
    imp = None
    print(f"\nsaved {best} -> {OUT} (not deployed)")


if __name__ == "__main__":
    main()
