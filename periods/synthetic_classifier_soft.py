"""Synthetic reliability classifier with soft labels. The question it answers is the one left to it once harmonics,
aliases and precision go to the shape stage: is the pipeline's period one the shape stage can reach the truth from,
because the data contain the signal? A reachable period (within +-SCAN of P_inj x {1, 2, 1/2}) can also be a
coincidence -- noise lands in those windows CHANCE of the time (blank paths with stand-in periods) -- so each reachable
object is a real recovery only with probability (reach rate r = r_real + (1 - r_real) CHANCE)
    w = max(0, (r - CHANCE) / (r (1 - CHANCE)))
where the reach rate is measured among rotators of similar expected integrated S/N
    S/N_exp = s_frame x A x rms(s) x sqrt(N)
(per-frame S/N, template amplitude, unit-amplitude template rms, points): known from the injection, never measured
from the noisy data and never a feature. Training: reachable objects enter as positive with weight w and negative
with weight 1 - w; everything else negative. No threshold is chosen for detection.
Evaluations as synthetic_classifier.py (synthetic truth and blind plots weighted the same way). Runs on ozstar.
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
OUT = f"{TD}/reliability_classifier_synthetic_soft.joblib"
N_BINS = 30
PREC_TARGET = 0.90
HGB = dict(max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0, min_samples_leaf=20, random_state=0)


def X(df, feats):
    return df[feats].astype(float).replace([np.inf, -np.inf], np.nan).values


def wprec_rec(w_pos, w_neg, sel):
    tp = w_pos[sel].sum()
    return tp / max(w_pos[sel].sum() + w_neg[sel].sum(), 1e-9), tp / max(w_pos.sum(), 1e-9)


def main():
    feats = list(dict.fromkeys(list(joblib.load(f"{TD}/reliability_classifier.joblib")["features"]) + ["fold_vn"]))
    d = pd.read_csv(f"{BI}/synthetic_training_set_all_detect.csv", low_memory=False)
    d = d[d.error.isna() & d.period_hr.notna()].reset_index(drop=True)
    tp = pd.read_csv(f"{S}/injection/templates.csv")
    cc = [f"c{j}" for j in range(1, 13)]
    rms = np.sqrt((tp[cc].values ** 2).sum(1) / 2)
    d["snr_exp"] = d.s_frame * d.amp_inj * rms[d.template.astype(int).values] * np.sqrt(d.n_points)
    rot = (d.kind == "rotator").values
    reach = d.reach.astype(bool).values
    null = d.stand_in.fillna(False).astype(bool).values
    chance = float(d.reach_chance[null].astype(bool).mean())
    edges = np.unique(np.quantile(d.snr_exp[rot], np.linspace(0, 1, N_BINS + 1)))
    b = np.clip(np.searchsorted(edges, d.snr_exp, side="right") - 1, 0, len(edges) - 2)
    rate = pd.Series(reach[rot]).groupby(b[rot]).mean()
    # observed reach r = r_real + (1 - r_real) c  ->  P(real | reached) = (r - c) / (r (1 - c))
    r_ = rate.reindex(range(len(edges) - 1)).values
    wbin = np.clip((r_ - chance) / (r_ * (1 - chance)), 0, 1)
    d["w_real"] = np.where(rot & reach, wbin[b], 0.0)
    print(f"chance reach rate {chance:.3f}; per S/N_exp bin: reach rate -> weight")
    for i in range(0, len(edges) - 1, 3):
        print(f"   S/N_exp {edges[i]:7.2f}-{edges[i + 1]:7.2f}: reach {rate.get(i, np.nan):.3f} -> w {wbin[i]:.2f}")
    w_pos = d.w_real.values
    w_neg = np.where(rot & reach, 1 - w_pos, 1.0)
    print(f"objects {len(d):,}; effective positives {w_pos.sum():,.0f} ({w_pos.sum() / len(d):.1%})")

    def stacked(idx):
        Xi = X(d.iloc[idx], feats)
        return (np.vstack([Xi, Xi]), np.concatenate([np.ones(len(idx)), np.zeros(len(idx))]),
                np.concatenate([w_pos[idx], w_neg[idx]]))

    oof = np.zeros(len(d))
    for a, t_ in GroupKFold(n_splits=5).split(d, rot, groups=d.source):
        Xa, ya, wa = stacked(a)
        keep = wa > 0
        oof[t_] = HistGradientBoostingClassifier(**HGB).fit(Xa[keep], ya[keep], sample_weight=wa[keep]).predict_proba(X(d.iloc[t_], feats))[:, 1]
    Xs = np.concatenate([oof, oof]); ys = np.concatenate([np.ones(len(d)), np.zeros(len(d))]); ws = np.concatenate([w_pos, w_neg])
    k = ws > 0
    auc = roc_auc_score(ys[k], Xs[k], sample_weight=ws[k])
    order = np.argsort(-oof)
    cp = np.cumsum(w_pos[order]); cn = np.cumsum(w_neg[order])
    prec = cp / np.maximum(cp + cn, 1e-9)
    ok = np.flatnonzero(prec >= PREC_TARGET)
    th = float(oof[order][ok.max()]) if len(ok) else 1.0
    sel = oof >= th
    p_, r_ = wprec_rec(w_pos, w_neg, sel)
    dep = d.is_reliable.astype(str).str.lower().eq("true").values
    pd_, rd_ = wprec_rec(w_pos, w_neg, dep)
    print(f"\n1. synthetic (grouped CV, weighted truth): AUC {auc:.3f}; at {PREC_TARGET:.0%} precision threshold {th:.3f}: "
          f"precision {p_:.1%}, recall {r_:.1%}; deployed (round 2 at injection time): precision {pd_:.1%}, recall {rd_:.1%}")
    rows = []
    for bnd, idx in d.groupby(pd.cut(d.s_frame, [0.1, 0.5, 1, 2, 5, 20, 50]), observed=True).groups.items():
        i = np.asarray(idx)
        a1, a2 = wprec_rec(w_pos[i], w_neg[i], sel[i]); b1, b2 = wprec_rec(w_pos[i], w_neg[i], dep[i])
        rows.append(dict(s_frame=str(bnd), real_recoveries=w_pos[i].sum() / len(i), syn_precision=a1, syn_recall=a2,
                         dep_precision=b1, dep_recall=b2))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    fails = d.kind.isin(["blank", "constant"]).values
    print(f"   pass rate on blank / constant: synthetic {sel[fails].mean():.1%}, deployed {dep[fails].mean():.1%}")

    Xa, ya, wa = stacked(np.arange(len(d)))
    keep = wa > 0
    model = HistGradientBoostingClassifier(**HGB).fit(Xa[keep], ya[keep], sample_weight=wa[keep])
    joblib.dump(dict(model=model, features=feats, threshold=th, chance=chance,
                     note="soft labels: reachable by the shape stage x P(real recovery | S/N_exp); NOT deployed"), OUT)

    lab = pd.concat([pd.read_csv(f"{S}/label_batches/labels_final.csv")[["designation", "label_final"]].assign(src="round1")]
                    + [pd.read_csv(f"{S}/label_batches/labels_{n}.csv").assign(label_final=lambda x: np.where(x.good, "good", "bad"), src=n)[["designation", "label_final", "src"]]
                       for n in ("faint2", "faint3")]).drop_duplicates("designation")
    h = pd.read_csv(f"{S}/bin_range_test.csv", low_memory=False).drop_duplicates("designation").merge(lab, on="designation")
    h = h[h.period_hr.notna()].reset_index(drop=True)
    yh = (h.label_final == "good").values
    faint = h.src.isin(["faint2", "faint3"]).values
    s = model.predict_proba(X(h, feats))[:, 1]
    print(f"\n2. user labels ({len(h)}, {yh.mean():.1%} good, {int(faint.sum())} faint): AUC all {roc_auc_score(yh, s):.3f}, "
          f"faint {roc_auc_score(yh[faint], s[faint]):.3f} (deployed round 3 CV 0.936 / 0.917); at threshold passes "
          f"{(s >= th)[yh].mean():.1%} of user-good, {(s >= th)[~yh].mean():.1%} of user-bad; agreement {np.mean((s >= th) == yh):.1%}")

    bk = pd.read_csv(f"{BI}/full/blind_key.csv")[["file", "syn"]]
    bu = pd.read_csv(f"{BI}/full/blind_results_user.csv")[["file", "kept"]]
    bl = bk.merge(bu, on="file").merge(d[["syn"]].assign(i=np.arange(len(d))), on="syn")
    i = bl.i.values
    print(f"\n3. blind plots ({len(bl)}, weighted positives {w_pos[i].sum():.1f}): precision / recall")
    for name, sl in (("you", bl.kept.astype(bool).values), ("deployed classifier", dep[i]), ("synthetic model (CV)", sel[i])):
        a1, a2 = wprec_rec(w_pos[i], w_neg[i], sl)
        print(f"   {name:22s} {a1:.1%} / {a2:.1%}")
    print(f"saved {OUT} (not deployed)")


if __name__ == "__main__":
    main()
