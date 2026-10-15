"""Examples of the synthetic classifier's decisions (reliability_classifier_synthetic_recovered_2pct.joblib), plotted
with the pipeline's own figure (plot_asteroid) into one folder, the verdict in each file name:
  <category>__score<score>__<truth>__<designation>.png
Synthetic objects (truth known; held-out scores from grouped CV by source): correct period passed, wrong period
passed, correct period rejected, wrong period rejected, blank path rejected. The user's real labelled objects (final
model): user-good passed, user-good rejected, user-bad passed. Plus summary.png: score distributions by truth, and
permutation importance (drop in held-out AUC when each feature is shuffled).
NOTE: each plot's own title shows the DEPLOYED classifier's verdict, not this model's -- the file name has this one.
Runs on ozstar.
    ASTEROID_SAVE_PLOTS=1 python3 -u synthetic_examples.py
"""
import matplotlib
matplotlib.use("Agg")
import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import sector_report_pipeline as p
from tessellate import asteroid_store as st

S = "/fred/oz335/rridden/asteroids_store"
BI = f"{S}/blank_injection"
OUT = f"{S}/synthetic_examples"
N_EACH = 6
SEED = 5
HGB = dict(max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0, min_samples_leaf=20, random_state=0)


def X(df, feats):
    return df[feats].astype(float).replace([np.inf, -np.inf], np.nan).values


def syn_track(row):
    b = int(row.syn[3:]) % 200
    g = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{BI}/{row.run}/phot/bucket={b}/*.parquet")])
    return p.normalise(p.frames_to_points(p._store_frame(g[g.designation == row.syn])))


def real_track(d):
    return p.normalise(p.frames_to_points(p._store_frame(
        st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))))


def main():
    os.makedirs(OUT, exist_ok=True)
    b = joblib.load("/fred/oz335/TESSdata/asteroid_store/reliability_classifier_synthetic_recovered_2pct.joblib")
    feats, th = b["features"], b["threshold"]
    d = pd.read_csv(f"{BI}/synthetic_training_set_all.csv", low_memory=False)
    d = d[d.error.isna() & d.period_hr.notna()].reset_index(drop=True)
    y = d.recovered_2pct.astype(str).str.lower().eq("true").values
    oof = np.zeros(len(d))
    imps = []
    for k, (a, t_) in enumerate(GroupKFold(n_splits=5).split(d, y, groups=d.source)):
        m = HistGradientBoostingClassifier(**HGB).fit(X(d.iloc[a], feats), y[a])
        oof[t_] = m.predict_proba(X(d.iloc[t_], feats))[:, 1]
        if k == 0:
            sub = np.random.default_rng(SEED).choice(t_, size=min(15000, len(t_)), replace=False)
            imps = permutation_importance(m, X(d.iloc[sub], feats), y[sub], scoring="roc_auc", n_repeats=5, random_state=SEED)
    d["score"] = oof
    print(f"held-out AUC {roc_auc_score(y, oof):.3f}, threshold {th:.3f}")

    # summary figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(0, 1, 41)
    for lab, sel, col in (("Correct period", y, "tab:blue"), ("Wrong period (rotator)", ~y & (d.kind == "rotator").values, "tab:red"),
                          ("Blank or constant", d.kind.isin(["blank", "constant"]).values, "0.4")):
        ax[0].hist(d.score[sel], bins=bins, histtype="step", lw=2, color=col, label=f"{lab} ({sel.sum():,})", density=True)
    ax[0].axvline(th, color="k", ls="--", label=f"Threshold {th:.2f}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Synthetic classifier score (held out)"); ax[0].set_ylabel("Density")
    ax[0].legend(fontsize=8)
    o = np.argsort(imps.importances_mean)
    ax[1].barh(np.array(feats)[o], imps.importances_mean[o], xerr=imps.importances_std[o], color="tab:blue")
    ax[1].set_xlabel("Drop in held-out AUC when shuffled")
    ax[1].set_title("What the model relies on", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    passed = d.score >= th
    rot = d.kind == "rotator"
    cats = {"A_correct_passed": d[y & passed], "B_wrong_passed": d[~y & passed & rot],
            "C_correct_rejected": d[y & ~passed], "D_wrong_rejected": d[~y & ~passed & rot],
            "E_blank_rejected": d[(d.kind == "blank") & ~passed]}
    for name, g in cats.items():
        for _, r in g.sample(min(N_EACH, len(g)), random_state=SEED).iterrows():
            truth = f"inj{r.period_inj_hr:.2f}h_got{r.period_hr:.2f}h_Sf{r.s_frame:.2f}" if r.kind == "rotator" else r.kind
            try:
                p.plot_asteroid(r.syn, syn_track(r), f"{OUT}/{name}__score{r.score:.2f}__{truth}__{r.syn}.png")
            except Exception as ex:
                print(f"  {r.syn}: {type(ex).__name__}: {ex}")
        print(f"{name}: {len(g):,} objects, plotted {min(N_EACH, len(g))}")

    # real labelled objects, final model
    lab = pd.concat([pd.read_csv(f"{S}/label_batches/labels_final.csv")[["designation", "label_final"]]]
                    + [pd.read_csv(f"{S}/label_batches/labels_{n}.csv").assign(label_final=lambda x: np.where(x.good, "good", "bad"))[["designation", "label_final"]]
                       for n in ("faint2", "faint3")]).drop_duplicates("designation")
    h = pd.read_csv(f"{S}/bin_range_test.csv", low_memory=False).drop_duplicates("designation").merge(lab, on="designation")
    h = h[h.period_hr.notna()].reset_index(drop=True)
    h["score"] = b["model"].predict_proba(X(h, feats))[:, 1]
    good, ps = h.label_final == "good", h.score >= th
    for name, g in {"F_real_good_passed": h[good & ps], "G_real_good_rejected": h[good & ~ps], "H_real_bad_passed": h[~good & ps]}.items():
        for _, r in g.sample(min(N_EACH, len(g)), random_state=SEED).iterrows():
            try:
                p.plot_asteroid(r.designation, real_track(r.designation),
                                f"{OUT}/{name}__score{r.score:.2f}__user_{r.label_final}__{p.safe_name(r.designation)}.png")
            except Exception as ex:
                print(f"  {r.designation}: {type(ex).__name__}: {ex}")
        print(f"{name}: {len(g):,} objects, plotted {min(N_EACH, len(g))}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
