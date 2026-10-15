"""Category B of c_l_examples (LCDB holdout objects contradicted by the LCDB period but passed by C + L + event
dependence): what relation does the pipeline period have to the LCDB period?
  harmonic   P_tess / P_lcdb within SCAN of a small-integer ratio m/k (k, m <= 4) other than 1, 2, 1/2
  orbit      the frequency difference is within 0.1 cycles of an integer number of cycles per TESS orbit (13.7 d)
  daily      the frequency difference is within 0.05 c/d of an integer c/d
  unrelated  none of these
Runs on ozstar.   python3 -u b_alias_check.py
"""
import glob
from fractions import Fraction

import joblib
import numpy as np
import pandas as pd

import sector_report_pipeline as p
import synthetic_feature_test as sft
import strengthen_c as sc
import lcdb_real_test as lt

S = sft.S
ED = f"{S}/event_dependence"
ORBIT_D = 13.7


def relation(pt, pl):
    r = pt / pl
    for k in range(1, 5):
        for m in range(1, 5):
            q = m / k
            if Fraction(m, k).denominator == k and q not in (1, 2, 0.5) and abs(r / q - 1) < sc.SCAN:
                return f"harmonic {m}/{k}"
    df = abs(24 / pt - 24 / pl)
    if round(df) >= 1 and abs(df - round(df)) < 0.05:
        return f"daily {round(df)} c/d"
    n = df * ORBIT_D
    if round(n) >= 1 and abs(n - round(n)) < 0.1:
        return f"orbit {round(n)} c/orbit"
    return "unrelated"


def main():
    b = joblib.load(f"{sft.TD}/reliability_classifier_C_L_ev.joblib")
    rep = sft.derive(pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False))
    rep = rep[rep.period_hr.notna()].drop_duplicates("designation").reset_index(drop=True)
    ev = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ED}/cand_*.csv") + glob.glob(f"{ED}/test_*.csv") + glob.glob(f"{ED}/lcdbtrain_*.csv")])
    ev = ev.dropna(subset=["max_1day_power_drop"]).drop_duplicates("designation").set_index("designation")
    for f_ in ("max_1day_power_drop", "max_3day_power_drop"):
        rep[f_] = rep.designation.map(ev[f_])
    rep["cl"] = b["model"].predict_proba(sft.X(rep, b["features"]))[:, 1]
    rep.loc[rep.max_1day_power_drop.isna(), "cl"] = 0.0
    npnt = rep.n_points.astype(float)
    gates = ((npnt >= p.RELIABLE_MIN_POINTS) & rep.n_cycles_observed.ge(p.RELIABLE_MIN_CYCLES)
             & np.isfinite(rep.aov_F.astype(float)) & np.isfinite(rep.amp_snr.astype(float))
             & (npnt >= p.RELIABILITY_MIN_POINTS_FLOOR) & rep.aov_F.ge(p.RELIABILITY_MIN_AOVF_FLOOR)
             & (rep.fold_min_upper.ge(0) | rep.max_visit_motion_px.lt(p.STATIONARY_MAX_PX))
             & ~rep.period_at_search_edge.astype(str).str.lower().eq("true"))
    rep["cl_pass"] = gates & (rep.cl >= b["threshold"])
    T = lt.targets().merge(rep, on="designation")
    B = T[~sc.reach(T.period_hr, T.Period) & T.cl_pass].copy()
    B["ratio"] = B.period_hr / B.Period
    B["relation"] = [relation(a, c) for a, c in zip(B.period_hr, B.Period)]
    B["dep_pass"] = B.is_reliable.astype(str).str.lower().eq("true")
    cols = ["designation", "Period", "period_hr", "ratio", "relation", "cl", "dep_pass", "amp_snr", "n_cycles_observed",
            "n_phase_coverings", "max_1day_power_drop", "best_rival_ratio"]
    print(B.sort_values("ratio")[cols].round(3).to_string(index=False))
    print("\n" + B.relation.str.split().str[0].value_counts().to_string())
    B[cols].to_csv(f"{S}/c_l_ev_B_relations.csv", index=False)


if __name__ == "__main__":
    main()
