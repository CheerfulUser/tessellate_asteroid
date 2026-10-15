"""Which parameters is the synthetic classifier missing, and does adding the user's labels help? Soft labels as
synthetic_classifier_soft.py (reachable by the shape stage x P(real recovery | S/N_exp)). Feature sets:
  A  the deployed features + fold_vn
  B  A + log10 period, best_rival_ratio, n_rival_peaks (competing non-harmonic peaks), integrated S/N
     (amp_snr x sqrt(N)), noise level (fitted peak-to-peak / amp_snr = median relative error), chi2_red_fourier,
     bin_range_sig -- quantities the label depends on that A lacks
  C  B trained on the synthetic soft labels plus the user's labels (total user weight = total synthetic weight)
Judged on: synthetic grouped CV (weighted truth; C with the user labels always in training); AUC on the user's labels
(A, B: model trained on synthetic only; C: repeated 5-fold CV over the user labels); the blind plots (A, B: grouped
CV scores; C: trained without the blind objects' sources). Runs on ozstar.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold

S = "/fred/oz335/rridden/asteroids_store"
BI = f"{S}/blank_injection"
TD = "/fred/oz335/TESSdata/asteroid_store"
N_BINS = 30
PREC_TARGET = 0.90
HGB = dict(max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0, min_samples_leaf=20, random_state=0)
EXTRA = ["logP", "best_rival_ratio", "n_rival_peaks", "int_snr", "noise_est", "chi2_red_fourier", "bin_range_sig"]


def derive(df):
    df = df.copy()
    df["logP"] = np.log10(df.period_hr.astype(float))
    df["int_snr"] = df.amp_snr.astype(float) * np.sqrt(df.n_points.astype(float))
    df["noise_est"] = 2 * (1 - df.model_min_rel_flux.astype(float)).clip(lower=1e-6) / df.amp_snr.astype(float)
    return df


def X(df, feats):
    return df[feats].astype(float).replace([np.inf, -np.inf], np.nan).values


def soft_weights(d):
    tp = pd.read_csv(f"{S}/injection/templates.csv")
    rms = np.sqrt((tp[[f"c{j}" for j in range(1, 13)]].values ** 2).sum(1) / 2)
    snr_exp = d.s_frame * d.amp_inj * rms[d.template.astype(int).values] * np.sqrt(d.n_points)
    rot = (d.kind == "rotator").values
    reach = d.reach.astype(bool).values
    null = d.stand_in.fillna(False).astype(bool).values
    c = float(d.reach_chance[null].astype(bool).mean())
    edges = np.unique(np.quantile(snr_exp[rot], np.linspace(0, 1, N_BINS + 1)))
    b = np.clip(np.searchsorted(edges, snr_exp, side="right") - 1, 0, len(edges) - 2)
    r = pd.Series(reach[rot]).groupby(b[rot]).mean().reindex(range(len(edges) - 1)).values
    wbin = np.clip((r - c) / (r * (1 - c)), 0, 1)
    w_pos = np.where(rot & reach, wbin[b], 0.0)
    return w_pos, np.where(rot & reach, 1 - w_pos, 1.0)


def fit(Xs, ws_pos, ws_neg, Xu=None, yu=None, wu=None):
    Xa = np.vstack([Xs, Xs]); ya = np.r_[np.ones(len(Xs)), np.zeros(len(Xs))]; wa = np.r_[ws_pos, ws_neg]
    if Xu is not None:
        Xa = np.vstack([Xa, Xu]); ya = np.r_[ya, yu]; wa = np.r_[wa, wu]
    k = wa > 0
    return HistGradientBoostingClassifier(**HGB).fit(Xa[k], ya[k], sample_weight=wa[k])


def weighted_eval(score, w_pos, w_neg):
    s2 = np.r_[score, score]; y2 = np.r_[np.ones(len(score)), np.zeros(len(score))]; w2 = np.r_[w_pos, w_neg]
    k = w2 > 0
    auc = roc_auc_score(y2[k], s2[k], sample_weight=w2[k])
    o = np.argsort(-score); cp = np.cumsum(w_pos[o]); cn = np.cumsum(w_neg[o])
    ok = np.flatnonzero(cp / np.maximum(cp + cn, 1e-9) >= PREC_TARGET)
    th = float(score[o][ok.max()]) if len(ok) else 1.0
    sel = score >= th
    return auc, th, w_pos[sel].sum() / max(w_pos[sel].sum() + w_neg[sel].sum(), 1e-9), w_pos[sel].sum() / w_pos.sum(), sel


def main():
    base = list(dict.fromkeys(list(joblib.load(f"{TD}/reliability_classifier.joblib")["features"]) + ["fold_vn"]))
    d = derive(pd.read_csv(f"{BI}/synthetic_training_set_all_detect.csv", low_memory=False))
    d = d[d.error.isna() & d.period_hr.notna()].reset_index(drop=True)
    w_pos, w_neg = soft_weights(d)
    lab = pd.concat([pd.read_csv(f"{S}/label_batches/labels_final.csv")[["designation", "label_final"]].assign(src="round1")]
                    + [pd.read_csv(f"{S}/label_batches/labels_{n}.csv").assign(label_final=lambda x: np.where(x.good, "good", "bad"), src=n)[["designation", "label_final", "src"]]
                       for n in ("faint2", "faint3")]).drop_duplicates("designation")
    corr = pd.read_csv(f"{S}/label_batches/label_corrections.csv")
    h = derive(pd.read_csv(f"{S}/bin_range_test.csv", low_memory=False).drop_duplicates("designation").merge(lab, on="designation"))
    h = h[h.period_hr.notna()].reset_index(drop=True)
    h.loc[h.designation.isin(corr.designation), "label_final"] = "good"
    yh = (h.label_final == "good").values.astype(float)
    faint = h.src.isin(["faint2", "faint3"]).values
    w_user_each = (w_pos.sum() + w_neg.sum()) / len(h)          # total user weight = total synthetic weight
    bk = pd.read_csv(f"{BI}/full/blind_key.csv")[["file", "syn"]]
    bu = pd.read_csv(f"{BI}/full/blind_results_user.csv")[["file", "kept"]]
    bl = bk.merge(bu, on="file").merge(d[["syn", "source"]].assign(i=np.arange(len(d))), on="syn")
    bi = bl.i.values
    folds = list(GroupKFold(n_splits=5).split(d, groups=d.source))
    print(f"synthetic {len(d):,} (effective positives {w_pos.sum():,.0f}); user labels {len(h)} ({yh.mean():.1%} good, "
          f"{len(corr)} corrected to good); each user label weighted {w_user_each:.1f}")
    for name, feats, hybrid in (("A current features", base, False), ("B + missing parameters", base + EXTRA, False),
                                ("C = B + user labels", base + EXTRA, True)):
        Xs, Xu = X(d, feats), X(h, feats)
        oof = np.zeros(len(d))
        for a, t_ in folds:
            m = fit(Xs[a], w_pos[a], w_neg[a], *((Xu, yh, np.full(len(h), w_user_each)) if hybrid else ()))
            oof[t_] = m.predict_proba(Xs[t_])[:, 1]
        auc, th, prec, rec, sel = weighted_eval(oof, w_pos, w_neg)
        if hybrid:
            su = np.zeros(len(h)); cnt = np.zeros(len(h))
            for tri, tei in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(h, yh):
                m = fit(Xs, w_pos, w_neg, Xu[tri], yh[tri], np.full(len(tri), w_user_each))
                su[tei] += m.predict_proba(Xu[tei])[:, 1]; cnt[tei] += 1
            su /= cnt
            keep = ~d.source.isin(bl.source).values
            m = fit(Xs[keep], w_pos[keep], w_neg[keep], Xu, yh, np.full(len(h), w_user_each))
            sb = m.predict_proba(Xs[bi])[:, 1]
        else:
            su = fit(Xs, w_pos, w_neg).predict_proba(Xu)[:, 1]
            sb = oof[bi]
        pu = su >= th
        bsel = sb >= th
        bp = w_pos[bi][bsel].sum() / max(w_pos[bi][bsel].sum() + w_neg[bi][bsel].sum(), 1e-9); br = w_pos[bi][bsel].sum() / w_pos[bi].sum()
        print(f"\n{name}:\n  synthetic CV: AUC {auc:.3f}; at 90% precision recall {rec:.1%} (threshold {th:.3f}); "
              f"blank/constant pass {sel[d.kind.isin(['blank', 'constant']).values].mean():.1%}; faintest band "
              f"(s_f < 0.5) recall {w_pos[sel & (d.s_frame < 0.5).values].sum() / w_pos[(d.s_frame < 0.5).values].sum():.1%}")
        print(f"  user labels: AUC all {roc_auc_score(yh, su):.3f}, faint {roc_auc_score(yh[faint], su[faint]):.3f}; at threshold passes "
              f"{pu[yh == 1].mean():.1%} of user-good, {pu[yh == 0].mean():.1%} of user-bad")
        print(f"  blind plots: precision {bp:.1%}, recall {br:.1%}")
        if name.startswith("B"):
            m = fit(Xs, w_pos, w_neg)
            from sklearn.inspection import permutation_importance
            sub = np.random.default_rng(0).choice(len(d), 15000, replace=False)
            yb = (w_pos[sub] >= 0.5).astype(int)
            pi = permutation_importance(m, Xs[sub], yb, scoring="roc_auc", n_repeats=3, random_state=0)
            o = np.argsort(pi.importances_mean)[::-1][:10]
            print("  top features (drop in AUC when shuffled): " + ", ".join(f"{feats[j]} {pi.importances_mean[j]:.3f}" for j in o))
    print("\n(reference: deployed round-3 classifier, CV on the user labels: AUC 0.936 / faint 0.917)")


if __name__ == "__main__":
    main()
