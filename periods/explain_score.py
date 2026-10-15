"""Why does C + L + event dependence pass (or reject) an object? For each designation: its feature values, their
percentile among the report objects that pass the hard gates, and the score change when that one feature is set to the
median of gate-passing objects the model rejects (largest drops = what lifts it over the threshold).
Runs on ozstar.   python3 -u explain_score.py "(75275) 1999 XG19" ...
"""
import glob
import sys

import joblib
import numpy as np
import pandas as pd

import sector_report_pipeline as p
import synthetic_feature_test as sft

S = sft.S
ED = f"{S}/event_dependence"


def cl_ev_scores():
    """The report (objects with event-dependence drops), its C + L + event-dependence scores and hard-gate mask."""
    b = joblib.load(f"{sft.TD}/reliability_classifier_C_L_ev.joblib")
    feats = b["features"]
    rep = sft.derive(pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False))
    rep = rep[rep.period_hr.notna()].drop_duplicates("designation").reset_index(drop=True)
    ev = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ED}/cand_*.csv") + glob.glob(f"{ED}/test_*.csv") + glob.glob(f"{ED}/lcdbtrain_*.csv")])
    ev = ev.dropna(subset=["max_1day_power_drop"]).drop_duplicates("designation").set_index("designation")
    for f_ in ("max_1day_power_drop", "max_3day_power_drop"):
        rep[f_] = rep.designation.map(ev[f_])
    rep = rep[rep.max_1day_power_drop.notna()].reset_index(drop=True)
    X = sft.X(rep, feats)
    rep["cl"] = b["model"].predict_proba(X)[:, 1]
    npnt = rep.n_points.astype(float)
    gates = ((npnt >= p.RELIABLE_MIN_POINTS) & rep.n_cycles_observed.ge(p.RELIABLE_MIN_CYCLES)
             & np.isfinite(rep.aov_F.astype(float)) & np.isfinite(rep.amp_snr.astype(float))
             & (npnt >= p.RELIABILITY_MIN_POINTS_FLOOR) & rep.aov_F.ge(p.RELIABILITY_MIN_AOVF_FLOOR)
             & (rep.fold_min_upper.ge(0) | rep.max_visit_motion_px.lt(p.STATIONARY_MAX_PX))
             & ~rep.period_at_search_edge.astype(str).str.lower().eq("true")).values
    rep["cl_pass"] = gates & (rep.cl.values >= b["threshold"])
    return b, rep, X, gates


def main(des):
    b, rep, X, gates = cl_ev_scores()
    feats, th = b["features"], b["threshold"]
    ref = np.nanmedian(X[gates & (rep.cl.values < th)], axis=0)
    for d in des:
        i = np.flatnonzero(rep.designation.values == d)
        if not len(i):
            print(f"{d}: not found (or no event-dependence drops)")
            continue
        i = i[0]
        x = X[i]
        print(f"\n{d}: score {rep.cl[i]:.3f} (threshold {th:.3f}), deployed {float(rep.reliability_score[i]):.3f}, "
              f"period {rep.period_hr[i]:.3f} h, gates {'pass' if gates[i] else 'FAIL'}")
        rows = []
        for j, f_ in enumerate(feats):
            xx = x.copy()
            xx[j] = ref[j]
            s = b["model"].predict_proba(xx[None])[0, 1]
            col = X[gates, j]
            pct = 100 * np.mean(col[np.isfinite(col)] < x[j]) if np.isfinite(x[j]) else np.nan
            rows.append(dict(feature=f_, value=x[j], pct_gated=pct, rejected_median=ref[j], score_if_typical=s, change=s - rep.cl[i]))
        print(pd.DataFrame(rows).sort_values("change").round(3).to_string(index=False))
        extra = ["n_phase_coverings", "max_visit_motion_px", "fold_min_upper", "amplitude", "n_sectors"]
        print(rep.loc[i, [c for c in extra if c in rep]].to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
