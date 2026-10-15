"""Reliability classifier trained on synthetic truth (blank-path injections, blank_injection.py): label 1 when the
pipeline's period is phase-coherent with the injected one over the object's baseline (drift < 0.1 cycle), else 0
(blank paths, constant sources, wrong periods, half / double periods). Features: the deployed classifier's plus
fold_vn. Cross-validation grouped by source asteroid (the copies of one source share frames and sky).

Compares, without deploying anything:
  1. synthetic CV: this model vs the deployed classifier (its own reliability_score / is_reliable on the same runs),
     overall and by per-frame S/N band
  2. the user's hand labels (labels_final + labels_faint2, features from bin_range_test.csv): AUC of the synthetic
     model against the user, vs a model cross-validated on the user labels themselves
  3. the 297 blind plots (truth known): precision / recall of the user, the deployed classifier and this model
Saves the model (all synthetic data) to asteroid_store/reliability_classifier_synthetic.joblib -- NOT deployed.
Runs on ozstar.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold

S = "/fred/oz335/rridden/asteroids_store"
TD = "/fred/oz335/TESSdata/asteroid_store"
SYN = os.environ.get("SYN", f"{S}/blank_injection/synthetic_training_set_all.csv")   # all runs incl. the faint one
DEPLOYED = f"{TD}/reliability_classifier.joblib"
LABEL = os.environ.get("LABEL", "recovered")   # recovered (phase-coherent over the baseline) or recovered_2pct
OUT = f"{TD}/reliability_classifier_synthetic{'' if LABEL == 'recovered' else '_' + LABEL}.joblib"
PREC_TARGET = 0.90
HGB = dict(max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0, min_samples_leaf=20, random_state=0)


def X(df, feats):
    return df[feats].astype(float).replace([np.inf, -np.inf], np.nan).values


def threshold_for_precision(y, s, target=PREC_TARGET):
    o = np.argsort(-s)
    tp = np.cumsum(y[o]); prec = tp / np.arange(1, len(y) + 1)
    ok = np.flatnonzero(prec >= target)
    return float(s[o][ok.max()]) if len(ok) else float(s.max())


def pr(y, sel):
    return (y & sel).sum() / max(sel.sum(), 1), (y & sel).sum() / max(y.sum(), 1)


def main():
    feats = list(dict.fromkeys(list(joblib.load(DEPLOYED)["features"]) + ["fold_vn"]))   # deployed may already have it
    d = pd.read_csv(SYN, low_memory=False)
    d = d[d.get("error", pd.Series(np.nan, index=d.index)).isna() & d.period_hr.notna()].reset_index(drop=True)
    y = d[LABEL].astype(str).str.lower().eq("true").values
    print(f"synthetic set: {len(d):,} objects, label {LABEL}: {y.mean():.1%} positive; features {feats}")

    # 1. grouped CV on the synthetic truth
    oof = np.zeros(len(d))
    for a, b in GroupKFold(n_splits=5).split(d, y, groups=d.source):
        oof[b] = HistGradientBoostingClassifier(**HGB).fit(X(d.iloc[a], feats), y[a]).predict_proba(X(d.iloc[b], feats))[:, 1]
    th = threshold_for_precision(y, oof)
    dep_s = d.reliability_score.astype(float).fillna(0).values
    dep_sel = d.is_reliable.astype(str).str.lower().eq("true").values
    syn_sel = oof >= th
    print(f"\n1. synthetic truth (grouped CV): AUC synthetic model {roc_auc_score(y, oof):.3f}, deployed score at the time of "
          f"the injection runs (round 2) {roc_auc_score(y, dep_s):.3f}")
    p1, r1 = pr(y, syn_sel); p2, r2 = pr(y, dep_sel)
    print(f"   synthetic model at {PREC_TARGET:.0%} precision (threshold {th:.3f}): precision {p1:.1%}, recall {r1:.1%}")
    print(f"   deployed classifier (is_reliable):            precision {p2:.1%}, recall {r2:.1%}")
    band = pd.cut(d.s_frame, [0.1, 0.5, 1, 2, 5, 20, 50])
    rows = []
    for bnd, idx in d.groupby(band, observed=True).groups.items():
        i = np.asarray(idx)
        ps, rs = pr(y[i], syn_sel[i]); pd_, rd = pr(y[i], dep_sel[i])
        rows.append(dict(s_frame=str(bnd), n=len(i), positive=y[i].mean(), syn_precision=ps, syn_recall=rs,
                         dep_precision=pd_, dep_recall=rd))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    fails = d.kind.isin(["blank", "constant"]).values
    print(f"   pass rate on blank / constant: synthetic {syn_sel[fails].mean():.1%}, deployed {dep_sel[fails].mean():.1%}")

    model = HistGradientBoostingClassifier(**HGB).fit(X(d, feats), y)
    joblib.dump(dict(model=model, features=feats, threshold=th, trained_on=f"{SYN} ({len(d)} objects)",
                     note="synthetic truth (phase-coherent recovery of injected periods); grouped-CV threshold for "
                          f"{PREC_TARGET:.0%} precision; NOT deployed"), OUT)

    # 2. the user's hand labels
    lab = pd.concat([pd.read_csv(f"{S}/label_batches/labels_final.csv").drop_duplicates("designation")[["designation", "label_final"]].assign(src="round1"),
                     pd.read_csv(f"{S}/label_batches/labels_faint2.csv").assign(label_final=lambda x: np.where(x.good, "good", "bad"), src="faint2")[["designation", "label_final", "src"]],
                     pd.read_csv(f"{S}/label_batches/labels_faint3.csv").assign(label_final=lambda x: np.where(x.good, "good", "bad"), src="faint3")[["designation", "label_final", "src"]]])
    lab = lab.drop_duplicates("designation")
    h = pd.read_csv(f"{S}/bin_range_test.csv", low_memory=False).drop_duplicates("designation").merge(lab, on="designation")
    h = h[h.period_hr.notna()].reset_index(drop=True)
    yh = (h.label_final == "good").values
    faint = h.src.isin(["faint2", "faint3"]).values
    s_syn = model.predict_proba(X(h, feats))[:, 1]
    own = np.zeros(len(h)); cnt = np.zeros(len(h))
    for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(h, yh):
        own[b] += HistGradientBoostingClassifier(**HGB).fit(X(h.iloc[a], feats), yh[a]).predict_proba(X(h.iloc[b], feats))[:, 1]; cnt[b] += 1
    own /= cnt
    print(f"\n2. user hand labels ({len(h)}, {yh.mean():.1%} good; {int(faint.sum())} faint):")
    for name, s in [("synthetic model (never saw these labels)", s_syn), ("model CV-trained on the user labels", own),
                    ("deployed classifier score", h.reliability_score.astype(float).fillna(0).values)]:
        print(f"   {name:42s} AUC all {roc_auc_score(yh, s):.3f}, faint {roc_auc_score(yh[faint], s[faint]):.3f}")
    sel = s_syn >= th
    agree = np.mean(sel == yh)
    print(f"   synthetic model at its threshold: agrees with the user on {agree:.1%}; passes {sel[yh].mean():.1%} of user-good, "
          f"{sel[~yh].mean():.1%} of user-bad")

    # 3. the blind plots (truth known)
    bk = pd.read_csv(f"{S}/blank_injection/full/blind_key.csv")[["file", "syn"]]
    bu = pd.read_csv(f"{S}/blank_injection/full/blind_results_user.csv")[["file", "kept"]]
    bl = bk.merge(bu, on="file").merge(d[["syn"]].assign(i=np.arange(len(d))), on="syn")
    i = bl.i.values
    yt = y[i]
    print("   (blind truth = the training label: " + LABEL + ")")
    print(f"\n3. blind plots ({len(bl)}, {yt.mean():.1%} coherent): precision / recall")
    for name, sel in [("you", bl.kept.astype(bool).values), ("deployed classifier", dep_sel[i]), ("synthetic model (CV)", syn_sel[i])]:
        p_, r_ = pr(yt, sel)
        print(f"   {name:22s} {p_:.1%} / {r_:.1%}")
    print(f"\nsaved {OUT} (not deployed)")


if __name__ == "__main__":
    main()
