"""Classifier C + L (synthetic soft labels + reliability-unweighted user labels + LCDB training rows, feature set B):
train the final model (threshold from synthetic grouped CV at 90 % weighted precision), score the current report,
and plot real examples into one folder with the pipeline's figure:
  A LCDB holdout, confirmed by LCDB and passed      B LCDB holdout, contradicted but passed
  C LCDB holdout, confirmed but rejected            D catalogue: gained vs the deployed classifier, amp_snr < 1
  E catalogue: gained, amp_snr >= 1                 F catalogue: lost vs the deployed classifier
  G user-good rejected                              H user-bad passed
File names: <category>__CL<score>__dep<deployed score>__<extra>__<designation>.png (each plot's own title shows the
DEPLOYED classifier's verdict). Prints how the reliable catalogue would change. Model saved, NOT deployed.
Runs on ozstar.   ASTEROID_SAVE_PLOTS=1 python3 -u c_l_examples.py
"""
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import sector_report_pipeline as p
import synthetic_feature_test as sft
import strengthen_c as sc
import lcdb_real_test as lt
from tessellate import asteroid_store as st

S = sft.S
MODEL = os.environ.get("MODEL", "")                # C_L_ev or C_L_ev<TAG>: a saved C + L + event-dependence model
OUT = f"{S}/c_l_examples" + ("_" + MODEL[4:].lower() if MODEL.startswith("C_L_ev") else "")
N_EACH = 6
SEED = 7


def track(d):
    return p.normalise(p.frames_to_points(p._store_frame(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))))


def main():
    os.makedirs(OUT, exist_ok=True)
    base = list(dict.fromkeys(list(joblib.load(f"{sft.TD}/reliability_classifier.joblib")["features"]) + ["fold_vn"]))
    feats = base + sft.EXTRA
    syn = sft.derive(pd.read_csv(f"{sft.BI}/synthetic_training_set_all_detect.csv", low_memory=False))
    syn = syn[syn.error.isna() & syn.period_hr.notna()].reset_index(drop=True)
    w_pos, w_neg = sft.soft_weights(syn)
    Xs = sft.X(syn, feats)
    lab = pd.concat([pd.read_csv(f"{S}/label_batches/labels_final.csv")[["designation", "label_final"]]]
                    + [pd.read_csv(f"{S}/label_batches/labels_{n}.csv").assign(label_final=lambda x: np.where(x.good, "good", "bad"))[["designation", "label_final"]]
                       for n in ("faint2", "faint3")]).drop_duplicates("designation")
    corr = pd.read_csv(f"{S}/label_batches/label_corrections.csv")
    h = sft.derive(pd.read_csv(f"{S}/bin_range_test.csv", low_memory=False).drop_duplicates("designation").merge(lab, on="designation"))
    h = h[h.period_hr.notna()].reset_index(drop=True)
    h.loc[h.designation.isin(corr.designation), "label_final"] = "good"
    yh = (h.label_final == "good").values.astype(float)
    Xu = sft.X(h, feats)
    wu = (w_pos.sum() + w_neg.sum()) / len(h)
    L = sc.r3.lcdb_rows()
    rl = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob(f"{sc.r3.ROWS}/lcdb_*.csv")]).drop_duplicates("designation")
    rl = rl[rl.get("error", pd.Series(np.nan, index=rl.index)).isna()]
    L = sft.derive(L.merge(rl, on="designation"))
    L = L[L.period_hr.notna() & ~L.designation.isin(h.designation)].reset_index(drop=True)
    parts_fixed = [(Xu, yh, np.full(len(h), wu)), (sft.X(L, feats), sc.reach(L.period_hr, L.period_updated).astype(float), wu * L.w.values)]
    syn_part = lambda idx: [(np.vstack([Xs[idx], Xs[idx]]), np.r_[np.ones(len(idx)), np.zeros(len(idx))], np.r_[w_pos[idx], w_neg[idx]])]
    oof = np.zeros(len(syn))
    for a, b_ in GroupKFold(n_splits=5).split(syn, groups=syn.source):
        oof[b_] = sc.fit(sc.BASE_HGB, syn_part(a) + parts_fixed).predict_proba(Xs[b_])[:, 1]
    _, th, prec, rec, _ = sft.weighted_eval(oof, w_pos, w_neg)
    model = sc.fit(sc.BASE_HGB, syn_part(np.arange(len(syn))) + parts_fixed)
    joblib.dump(dict(model=model, features=feats, threshold=th, note="C + L: synthetic soft labels + user labels (with corrections) "
                     "+ LCDB rows labelled by shape-stage reachability; NOT deployed"), f"{sft.TD}/reliability_classifier_C_L.joblib")
    print(f"C + L: threshold {th:.3f} (synthetic CV precision {prec:.1%}, recall {rec:.1%})")
    if MODEL.startswith("C_L_ev"):                     # the saved C + L + event-dependence model (strengthen_c.py)
        bb = joblib.load(f"{sft.TD}/reliability_classifier_{MODEL}.joblib")
        model, feats, th = bb["model"], bb["features"], bb["threshold"]
        print(f"using C + L + event dependence: threshold {th:.3f}, features {feats}")

    rep = sft.derive(pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False))
    rep = rep[rep.period_hr.notna()].drop_duplicates("designation").reset_index(drop=True)
    if MODEL.startswith("C_L_ev"):
        ED = f"{S}/event_dependence"
        ev = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ED}/cand_*.csv") + glob.glob(f"{ED}/test_*.csv") + glob.glob(f"{ED}/lcdbtrain_*.csv")])
        ev = ev.dropna(subset=["max_1day_power_drop"]).drop_duplicates("designation").set_index("designation")
        for f_ in ("max_1day_power_drop", "max_3day_power_drop"):
            rep[f_] = rep.designation.map(ev[f_])
        print(f"event-dependence drops available for {rep.max_1day_power_drop.notna().sum():,} report objects (the rest cannot pass)")
        if any(f_ in feats for f_ in sc.SYSF):          # systematics features (systematics_features.py)
            SD = f"{S}/systematics"
            sr = pd.concat([pd.read_csv(f) for f in glob.glob(f"{SD}/cand_*.csv") + glob.glob(f"{SD}/test_*.csv") + glob.glob(f"{SD}/lcdbtrain_*.csv")])
            sr = sr[sr.get("error", pd.Series(np.nan, index=sr.index)).isna()].drop_duplicates("designation").set_index("designation")
            for f_ in sc.SYSF:
                rep[f_] = rep.designation.map(sr[f_])
            print(f"systematics features available for {rep.model_gain.notna().sum():,} report objects (the rest cannot pass)")
    rep["cl"] = model.predict_proba(sft.X(rep, feats))[:, 1]
    if MODEL.startswith("C_L_ev"):
        rep.loc[rep.max_1day_power_drop.isna(), "cl"] = 0.0
        if "model_gain" in rep:
            rep.loc[rep.model_gain.isna(), "cl"] = 0.0
    # the pipeline's prerequisites (sector_report_pipeline.plot_asteroid prereq), from the report's own columns
    npnt = rep.n_points.astype(float)
    gates = ((npnt >= p.RELIABLE_MIN_POINTS) & rep.n_cycles_observed.ge(p.RELIABLE_MIN_CYCLES)
             & np.isfinite(rep.aov_F.astype(float)) & np.isfinite(rep.amp_snr.astype(float))
             & (npnt >= p.RELIABILITY_MIN_POINTS_FLOOR) & rep.aov_F.ge(p.RELIABILITY_MIN_AOVF_FLOOR)
             & (rep.fold_min_upper.ge(0) | rep.max_visit_motion_px.lt(p.STATIONARY_MAX_PX))
             & ~rep.period_at_search_edge.astype(str).str.lower().eq("true"))
    check = rep.is_reliable.astype(str).str.lower().eq("true") & ~gates
    print(f"gate reproduction check: deployed-reliable objects failing the reproduced gates: {int(check.sum())} (must be 0)")
    rep["cl_pass"] = gates & (rep.cl >= th)
    dep = rep.is_reliable.astype(str).str.lower().eq("true")
    b = pd.cut(rep.amp_snr, [0, 1, 2, 4, 1e9])
    print(f"\nreliable: deployed {int(dep.sum()):,}; C + L (with the hard gates) {int(rep.cl_pass.sum()):,}; kept {int((dep & rep.cl_pass).sum()):,}, "
          f"gained {int((~dep & rep.cl_pass).sum()):,}, lost {int((dep & ~rep.cl_pass).sum()):,}")
    print(pd.DataFrame({"deployed": dep.groupby(b, observed=True).sum(), "C+L": rep.cl_pass.groupby(b, observed=True).sum(),
                        "gained": (~dep & rep.cl_pass).groupby(b, observed=True).sum(), "lost": (dep & ~rep.cl_pass).groupby(b, observed=True).sum()}).to_string())

    T = lt.targets().merge(rep[["designation", "period_hr", "cl", "cl_pass", "reliability_score", "amp_snr"]], on="designation")
    T["confirmed"] = sc.reach(T.period_hr, T.Period)
    hh = h[["designation", "label_final"]].merge(rep[["designation", "cl", "cl_pass", "reliability_score", "amp_snr"]], on="designation")
    cats = {"A_LCDB_confirmed_passed": T[T.confirmed & T.cl_pass], "B_LCDB_contradicted_passed": T[~T.confirmed & T.cl_pass],
            "C_LCDB_confirmed_rejected": T[T.confirmed & ~T.cl_pass],
            "D_catalogue_gained_faint": rep[~dep & rep.cl_pass & (rep.amp_snr < 1)], "E_catalogue_gained_bright": rep[~dep & rep.cl_pass & (rep.amp_snr >= 1)],
            "F_catalogue_lost": rep[dep & ~rep.cl_pass],
            "G_user_good_rejected": hh[(hh.label_final == "good") & ~hh.cl_pass], "H_user_bad_passed": hh[(hh.label_final != "good") & hh.cl_pass]}
    only = os.environ.get("CATS")                      # e.g. CATS=G,H to plot only those categories
    for name, g in cats.items():
        if only and name[0] not in only.split(","):
            continue
        for _, r in g.sample(min(N_EACH, len(g)), random_state=SEED).iterrows():
            extra = f"LCDB{r.Period:.2f}h" if "Period" in r else f"ampsnr{r.amp_snr:.2f}" if "amp_snr" in r else ""
            fn = f"{OUT}/{name}__CL{r.cl:.2f}__dep{float(r.reliability_score):.2f}__{extra}__{p.safe_name(r.designation)}.png"
            try:
                p.plot_asteroid(r.designation, track(r.designation), fn)
            except Exception as ex:
                print(f"  {r.designation}: {type(ex).__name__}: {ex}")
        print(f"{name}: {len(g):,} objects, plotted {min(N_EACH, len(g))}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
