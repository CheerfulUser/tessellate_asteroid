"""Can the hybrid classifier C (synthetic soft labels + user labels, feature set B; synthetic_feature_test.py) be made
stronger? Variants, each a single change, then all together:
  C0   C as before
  +L   + the LCDB training objects (round3_rows, recomputed pipeline rows), labelled by whether the TESS period is
       reachable by the shape stage from the vetted LCDB period (+-8.829 % of P x {1, 2, 1/2}); weight = the user
       per-row weight x the deployed recipe's LCDB weight (0.5 x category weight)
  +R   user labels weighted by their measured reliability in the blind test, by per-frame S/N band: a "good" label by
       the share of kept plots that were truly reachable, a "bad" label by the share of rejected plots that were not;
       per-frame S/N of a real object = 1 / its median relative error (noise_est)
  +D   synthetic examples reweighted toward the real catalogue: a classifier separating synthetic from real objects
       (DOM_N random report rows with a period) gives w = p / (1 - p), normalised, clipped at DOM_CLIP
  +H   deeper trees: max_leaf_nodes 63, 800 iterations, learning rate 0.05, min_samples_leaf 40
Judged on synthetic grouped CV (weighted AUC, recall at 90 % precision), 5-fold CV on the user labels (all and the
reliable bright part), and once on the LCDB holdout (lcdb_real_test.py: 898 LCDB U=3 native objects no model saw).
Runs on ozstar.
"""
import glob

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

import synthetic_feature_test as sft
import train_round3_ozstar as r3
import lcdb_real_test as lt

S = sft.S
SCAN = 0.08829
DOM_N = 60000
DOM_CLIP = 20.0
BASE_HGB = dict(max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0, min_samples_leaf=20, random_state=0)
DEEP_HGB = dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0, min_samples_leaf=40, random_state=0)
SYSF = ["model_gain", "bkg_share", "pos_gain", "bkg_r2", "scale_dchi2"]   # systematics_features.py (agreed 2026-10-15)
SN_EDGES = [0, 0.5, 1, 2, 5, np.inf]


def reach(period, ref):
    r = period / ref
    return np.any([(r / q - 1).abs() < SCAN for q in (1, 2, 0.5)], axis=0)


def fit(params, parts):
    Xa = np.vstack([p_[0] for p_ in parts]); ya = np.concatenate([p_[1] for p_ in parts]); wa = np.concatenate([p_[2] for p_ in parts])
    k = wa > 0
    return HistGradientBoostingClassifier(**params).fit(Xa[k], ya[k], sample_weight=wa[k])


def blind_reliability():
    """Per per-frame-S/N band: P(truly reachable | kept), P(not reachable | rejected) from the blind plots."""
    bk = pd.read_csv(f"{sft.BI}/full/blind_key.csv")
    bu = pd.read_csv(f"{sft.BI}/full/blind_results_user.csv")[["file", "kept"]]
    b = bk.merge(bu, on="file")
    rot = b.kind == "rotator"
    truth = rot & reach(b.period_hr, b.period_inj_hr)
    band = np.where(rot, np.digitize(b.s_frame, SN_EDGES) - 1, -1)   # rotators only (blanks have no S/N band)
    ppv, npv = [], []
    for i in range(len(SN_EDGES) - 1):
        m = band == i
        k_, r_ = m & b.kept.astype(bool), m & ~b.kept.astype(bool)
        ppv.append(float(truth[k_].mean()) if k_.sum() >= 3 else 1.0)
        npv.append(float((~truth[r_]).mean()) if r_.sum() >= 3 else 1.0)
    return np.array(ppv), np.array(npv)


def main():
    base = list(dict.fromkeys(list(joblib.load(f"{sft.TD}/reliability_classifier.joblib")["features"]) + ["fold_vn"]))
    feats = base + sft.EXTRA
    import os
    syn = sft.derive(pd.read_csv(os.environ.get("SYN", f"{sft.BI}/synthetic_training_set_all_detect.csv"), low_memory=False))
    syn = syn[syn.error.isna() & syn.period_hr.notna()].reset_index(drop=True)
    # cut-data and artefact copies (cut_augment.py, artefact_augment.py) reach less often at a given S/N_exp: their
    # soft weights come from their own reach rates, so they do not dilute the full-data positives' weights
    grp = syn.run.astype(str).str.extract(r"^(cut|art)")[0].fillna("full").values   # artefact copies likewise
    w_pos, w_neg = np.zeros(len(syn)), np.zeros(len(syn))
    for g_ in np.unique(grp):
        m = grp == g_
        w_pos[m], w_neg[m] = sft.soft_weights(syn[m].reset_index(drop=True))
    print("synthetic rows by group: " + str(pd.Series(grp).value_counts().to_dict()))
    TAG = os.environ.get("TAG", "")                     # appended to the saved models' names
    Xs = sft.X(syn, feats)
    # user labels (with corrections)
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
    sn_real = 1 / h.noise_est.astype(float).clip(lower=1e-6)
    ppv, npv = blind_reliability()
    band = np.clip(np.digitize(sn_real, SN_EDGES) - 1, 0, len(SN_EDGES) - 2)
    rel = np.where(yh == 1, ppv[band], npv[band])
    bright = (sn_real >= 2).values
    print("blind-test reliability by per-frame S/N band " + str(SN_EDGES) + ": kept " + str(np.round(ppv, 2)) + ", rejected " + str(np.round(npv, 2)))
    # LCDB training rows
    L = r3.lcdb_rows()
    rl = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob(f"{r3.ROWS}/lcdb_*.csv")]).drop_duplicates("designation")
    rl = rl[rl.get("error", pd.Series(np.nan, index=rl.index)).isna()]
    L = sft.derive(L.merge(rl, on="designation"))
    L = L[L.period_hr.notna() & ~L.designation.isin(h.designation)].reset_index(drop=True)
    yL = reach(L.period_hr, L.period_updated).astype(float)
    XL, wL = sft.X(L, feats), wu * L.w.values
    print(f"synthetic {len(syn):,}; user labels {len(h)} ({int(bright.sum())} bright, per-frame S/N >= 2); LCDB rows {len(L)} ({yL.mean():.1%} reachable)")
    # domain weights
    rep = pd.read_csv(sft.S + "/all_sector_asteroid_report.csv", low_memory=False)
    NAMED = ["(275) Sapientia", "(651174) 2012 XY7", "(75275) 1999 XG19", "(30545) 2001 OT35", "(22056) 2000 AU31"]
    named = sft.derive(rep[rep.designation.isin(NAMED)].drop_duplicates("designation"))
    rep = sft.derive(rep[rep.period_hr.notna()].sample(min(DOM_N, int(rep.period_hr.notna().sum())), random_state=0))
    Xr = sft.X(rep, feats)
    dom = HistGradientBoostingClassifier(**BASE_HGB).fit(np.vstack([Xs, Xr]), np.r_[np.zeros(len(Xs)), np.ones(len(Xr))])
    pr = np.clip(dom.predict_proba(Xs)[:, 1], 1e-4, 1 - 1e-4)
    wd = pr / (1 - pr)
    wd = np.clip(wd / np.mean(wd), 0, DOM_CLIP)
    print(f"domain classifier (synthetic vs real) separation AUC "
          f"{roc_auc_score(np.r_[np.zeros(len(Xs)), np.ones(len(Xr))], dom.predict_proba(np.vstack([Xs, Xr]))[:, 1]):.3f}; "
          f"synthetic weight quantiles 10/50/90/99 {np.round(np.quantile(wd, [.1, .5, .9, .99]), 2)}")
    # LCDB holdout
    t = lt.targets()
    rows = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob(f"{lt.ROWS}/rows_*.csv")]).drop_duplicates("designation")
    rows = rows[rows.get("error", pd.Series(np.nan, index=rows.index)).isna() & rows.period_hr.notna()]
    T = sft.derive(t.merge(rows, on="designation"))
    yT = reach(T.period_hr, T.Period)
    XT = sft.X(T, feats)
    faintT = (T.amp_snr < 1).values

    gfolds = list(GroupKFold(n_splits=5).split(syn, groups=syn.source))
    ufolds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(h, yh))
    if os.environ.get("VARIANTS") == "npc":             # C + L with and without n_phase_coverings
        variants = {"+L": dict(L=1), "+L+npc": dict(L=1, NPC=1)}
    elif os.environ.get("VARIANTS") in ("evdep", "sys"):  # + event-dependence features (and coverings / systematics)
        variants = ({"+L": dict(L=1), "+L+ev": dict(L=1, EV=1), "+L+ev+npc": dict(L=1, EV=1, NPC=1)} if os.environ["VARIANTS"] == "evdep"
                    else {"+L+ev": dict(L=1, EV=1), "+L+ev+sys": dict(L=1, EV=1, SYS=1)})
        ED = "/fred/oz335/rridden/asteroids_store/event_dependence"
        evr = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ED}/test_*.csv") + glob.glob(f"{ED}/lcdbtrain_*.csv") + glob.glob(f"{ED}/cand_*.csv")])
        evr = evr.dropna(subset=["max_1day_power_drop"]).drop_duplicates("designation").set_index("designation")
        evs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ED}/syn_*.csv")]).drop_duplicates("syn").set_index("syn")
        EVF = ["max_1day_power_drop", "max_3day_power_drop"]
        for df_, key, src in ((syn, "syn", evs), (h, "designation", evr), (L, "designation", evr), (T, "designation", evr), (named, "designation", evr)):
            for f_ in EVF:
                df_[f_] = df_[key].map(src[f_]).values
        print("event-dependence coverage: synthetic %.3f, user %.3f, LCDB rows %.3f, holdout %.3f" % tuple(
            df_.max_1day_power_drop.notna().mean() for df_ in (syn, h, L, T)))
        if os.environ["VARIANTS"] == "sys":           # systematics_features.py
            SD = "/fred/oz335/rridden/asteroids_store/systematics"
            sr = pd.concat([pd.read_csv(f) for f in glob.glob(f"{SD}/test_*.csv") + glob.glob(f"{SD}/lcdbtrain_*.csv") + glob.glob(f"{SD}/cand_*.csv")])
            sr = sr[sr.get("error", pd.Series(np.nan, index=sr.index)).isna()].drop_duplicates("designation").set_index("designation")
            ss = pd.concat([pd.read_csv(f) for f in glob.glob(f"{SD}/syn_*.csv")]).drop_duplicates("syn").set_index("syn")
            for df_, key, src in ((syn, "syn", ss), (h, "designation", sr), (L, "designation", sr), (T, "designation", sr), (named, "designation", sr)):
                for f_ in SYSF:
                    df_[f_] = df_[key].map(src[f_]).values
            print("systematics coverage: synthetic %.3f, user %.3f, LCDB rows %.3f, holdout %.3f" % tuple(
                df_.model_gain.notna().mean() for df_ in (syn, h, L, T)))
    else:
        variants = {"C0": dict(), "+L": dict(L=1), "+R": dict(R=1), "+D": dict(D=1), "+H": dict(H=1), "all": dict(L=1, R=1, D=1, H=1)}
    def flist(v):
        return (feats + (["n_phase_coverings"] if v.get("NPC") else []) + (EVF if v.get("EV") else [])
                + (SYSF if v.get("SYS") else []))
    print(f"\n{'variant':8s} {'synAUC':>7s} {'synRec90':>8s} {'userAUC':>8s} {'bright':>7s} | LCDB holdout: {'AUC':>6s} {'prec':>6s} {'rec':>6s}  faint (amp_snr<1) prec/rec")
    for name, v in variants.items():
        params = DEEP_HGB if v.get("H") else BASE_HGB
        if v.get("NPC") or v.get("EV") or v.get("SYS"):  # rebuild the matrices with the added features
            f2 = flist(v)
            Xs, Xu, XL, XT = sft.X(syn, f2), sft.X(h, f2), sft.X(L, f2), sft.X(T, f2)
            print(f"   n_phase_coverings present: synthetic {syn.n_phase_coverings.notna().mean():.3f}, user {h.n_phase_coverings.notna().mean():.3f}, "
                  f"LCDB rows {L.n_phase_coverings.notna().mean():.3f}, holdout {T.n_phase_coverings.notna().mean():.3f}")
        sp, sn_ = (w_pos * wd, w_neg * wd) if v.get("D") else (w_pos, w_neg)
        uw = wu * (rel if v.get("R") else np.ones(len(h)))
        syn_part = lambda idx: [(np.vstack([Xs[idx], Xs[idx]]), np.r_[np.ones(len(idx)), np.zeros(len(idx))], np.r_[sp[idx], sn_[idx]])]
        extra_L = [(XL, yL, wL)] if v.get("L") else []
        oof = np.zeros(len(syn))
        for a, b_ in gfolds:
            oof[b_] = fit(params, syn_part(a) + [(Xu, yh, uw)] + extra_L).predict_proba(Xs[b_])[:, 1]
        auc_s, th, _, rec_s, _ = sft.weighted_eval(oof, w_pos, w_neg)
        allidx = np.arange(len(syn))
        su = np.zeros(len(h))
        for a, b_ in ufolds:
            su[b_] = fit(params, syn_part(allidx) + [(Xu[a], yh[a], uw[a])] + extra_L).predict_proba(Xu[b_])[:, 1]
        m = fit(params, syn_part(allidx) + [(Xu, yh, uw)] + extra_L)
        sT = m.predict_proba(XT)[:, 1]
        sel = sT >= th
        fp = yT[faintT & sel].mean() if (faintT & sel).any() else np.nan
        fr = sel[faintT & yT].mean()
        print(f"{name:8s} {auc_s:7.3f} {rec_s:8.1%} {roc_auc_score(yh, su):8.3f} {roc_auc_score(yh[bright], su[bright]):7.3f} | "
              f"{'':14s}{roc_auc_score(yT, sT):6.3f} {yT[sel].mean():6.1%} {sel[yT].mean():6.1%}  {fp:.0%}/{fr:.0%}  (threshold {th:.3f})")
        npc0 = (T.n_phase_coverings.astype(float) == 0).values
        print(f"         holdout objects with 0 phase coverings passed: {int((sel & npc0).sum())} of {int(npc0.sum())} "
              f"(reachable among them {yT[sel & npc0].mean() if (sel & npc0).any() else float('nan'):.0%})")
        if v.get("NPC") or v.get("EV") or v.get("SYS") or name == "+L":
            f_ = flist(v)
            for dsg in NAMED:
                row = named[named.designation == dsg]
                if len(row):
                    print(f"         {dsg}: score {m.predict_proba(sft.X(row, f_))[:, 1][0]:.2f} (threshold {th:.2f})")
            joblib.dump(dict(model=m, features=f_, threshold=th),
                        f"{sft.TD}/reliability_classifier_C_L{'_ev' if v.get('EV') else ''}{'_npc' if v.get('NPC') else ''}"
                        f"{'_sys' if v.get('SYS') else ''}{TAG}.joblib")
        if name == "all":
            joblib.dump(dict(model=m, features=feats, threshold=th, note="C + LCDB rows + reliability-weighted user labels + "
                             "domain reweighting + deep trees; NOT deployed"), f"{sft.TD}/reliability_classifier_C_all{TAG}.joblib")
    dep = joblib.load(f"{sft.TD}/reliability_classifier.joblib")
    sd = dep["model"].predict_proba(sft.X(T, dep["features"]))[:, 1]
    sel = sd >= dep["threshold"]
    print(f"{'deployed':8s} {'':7s} {'':8s} {'':8s} {'':7s} | {'':14s}{roc_auc_score(yT, sd):6.3f} {yT[sel].mean():6.1%} {sel[yT].mean():6.1%}  "
          f"{yT[faintT & sel].mean():.0%}/{sel[faintT & yT].mean():.0%}")


if __name__ == "__main__":
    main()
