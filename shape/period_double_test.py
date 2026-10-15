"""Validation of choosing between the periodogram peak P1 and 2 P1 by shape-model fit quality, against literature
periods. Three steps (run on ozstar):

  targets   reliable objects of the raw-frame store report with an independent literature period: LCDB U = 3 with
            period_source LCDB_native (not TESS-derived), 150 the pipeline left at P1 (double_peaked False) and 150
            it doubled, plus fixed checks (DAMIT periods of the four occultation objects; Nagainis & Eglitis 2026 for
            3081, 3473, 3716, 4303). Writes shapes/pdouble/targets.csv and, for the combined (TESS + sparse) fit,
            targets_comb_P1.csv / targets_comb_2P1.csv (the fixed checks plus 6 LCDB objects of each stratum).
  fit       TESS-only fixed-pole fits (batch_shapes.process, production settings) at P1 and 2 P1 for every
            SHAPE_TOTAL-th target; each fit's point_rms_min (per-point residual rms of the best orientation, which
            compares across periods, unlike the binned folded rms). Writes shapes/pdouble/fit_<index>.csv.
  summary   accuracy against the literature period (within TOL) of the pipeline's adopted period and of the
            shape choice (lower point_rms_min), by stratum, with near-ties (relative difference < NEAR_TIE); and,
            where the combined fits exist, the same choice by the joint fit's weighted chi2 (iterative chi2_w).

    python3 shape/period_double_test.py targets
    python3 shape/period_double_test.py fit <index> <total>
    python3 shape/period_double_test.py summary
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
S = "/fred/oz335/rridden/asteroids_store"
D = f"{S}/shapes/pdouble"
REPORT = f"{S}/all_sector_asteroid_report.csv"
LCDB = "/fred/oz335/rridden/asteroids/lcdb_updated.csv"
LCDIR = f"{S}/clean_lightcurves"
TOL = 0.02
NEAR_TIE = 0.01
N_PER_STRATUM = 150
N_COMB_PER_STRATUM = 6
SEED = 3
# literature periods (h) for the fixed checks: DAMIT (occultation objects), Nagainis & Eglitis 2026
FIXED = {"(22) Kalliope": 4.148199, "(130) Elektra": 5.224664, "(165) Loreley": 7.224378, "(554) Peraga": 13.712,
         "(3081) Martinuboh": 8.0073, "(3473) Sapporo": 9.0740, "(3716) Petzval": 10.4743, "(4303) Savitskij": 6.1363}


def sn(d):
    import batch_shapes as bs
    return bs.sn(d)


def targets():
    os.makedirs(D, exist_ok=True)
    r = pd.read_csv(REPORT, low_memory=False, usecols=["designation", "period_hr", "period_hr_naive", "double_peaked",
                                                        "is_reliable", "reliability_score"]).drop_duplicates("designation")
    r = r[r.is_reliable.astype(bool) & r.period_hr_naive.notna()]
    r = r[[os.path.exists(f"{LCDIR}/{sn(d)}_clean_lc.csv") for d in r.designation]]
    lc = pd.read_csv(LCDB, low_memory=False)
    # most rows carry Number and Name with an empty designation column: build "(number) name" as the report writes it
    built = "(" + lc.Number.astype("Int64").astype(str) + ") " + lc.Name.astype(str)
    lc["designation"] = lc.designation.where(lc.designation.notna(), built)
    lc = lc[(lc.U_clean == 3) & (lc.period_source == "LCDB_native")][["designation", "Period"]]
    m = r.merge(lc.rename(columns={"Period": "truth_hr"}), on="designation")
    m["source"] = "LCDB U=3"
    dp = m.double_peaked.astype(str).str.lower() == "true"
    samp = pd.concat([m[~dp].sample(min(N_PER_STRATUM, int((~dp).sum())), random_state=SEED).assign(stratum="pipeline P1"),
                      m[dp].sample(min(N_PER_STRATUM, int(dp.sum())), random_state=SEED).assign(stratum="pipeline 2P1")])
    fx = r[r.designation.isin(FIXED)].assign(truth_hr=lambda x: x.designation.map(FIXED), source="DAMIT / Nagainis 2026",
                                             stratum="fixed check")
    t = pd.concat([fx, samp[~samp.designation.isin(fx.designation)]], ignore_index=True)
    t.to_csv(f"{D}/targets.csv", index=False)
    comb = pd.concat([fx] + [samp[samp.stratum == s].head(N_COMB_PER_STRATUM) for s in ("pipeline P1", "pipeline 2P1")])
    for k, f in (("P1", 1.0), ("2P1", 2.0)):
        comb.assign(period_hr=comb.period_hr_naive * f)[["designation", "period_hr"]].to_csv(f"{D}/targets_comb_{k}.csv", index=False)
    print(f"targets: {len(t)} ({len(fx)} fixed checks of {len(FIXED)}; LCDB U=3 reliable pool {len(m)}: "
          f"P1 {int((~dp).sum())}, 2P1 {int(dp.sum())}); combined subset {len(comb)}")


def fit(index, total):
    import batch_shapes as bs
    t = pd.read_csv(f"{D}/targets.csv").iloc[index::total]
    rows = []
    for _, r in t.iterrows():
        row = r.to_dict()
        for k, f in (("P1", 1.0), ("2P1", 2.0)):
            work, out = f"{D}/work_{k}_{index}", f"{D}/out_{k}"
            os.makedirs(work, exist_ok=True)
            rec = bs.process(r.designation, r.period_hr_naive * f, f"{LCDIR}/{sn(r.designation)}_clean_lc.csv", work, out)
            for c in ("ok", "model_period_hr", "point_rms_min", "point_rms_median", "orientation_rms_min",
                      "equatorial_ratio", "polar_ratio", "seconds", "error"):
                row[f"{c}_{k}"] = rec.get(c)
        rows.append(row)
        print(f"  {r.designation}: P1 {row['point_rms_min_P1']} 2P1 {row['point_rms_min_2P1']}", flush=True)
    pd.DataFrame(rows).to_csv(f"{D}/fit_{index:03d}.csv", index=False)


def match(p, truth):
    return np.abs(p / truth - 1) < TOL


def summary():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{D}/fit_*.csv"))], ignore_index=True)
    ok = d.ok_P1.astype(bool) & d.ok_2P1.astype(bool) & d.point_rms_min_P1.notna() & d.point_rms_min_2P1.notna()
    print(f"{len(d)} objects, {int(ok.sum())} with both fits")
    d = d[ok].copy()
    in_pair = match(d.period_hr_naive, d.truth_hr) | match(2 * d.period_hr_naive, d.truth_hr)
    print(f"literature period is P1 or 2P1 for {int(in_pair.sum())}; other (excluded below): {int((~in_pair).sum())}")
    d = d[in_pair].copy()
    d["rel_diff"] = (d.point_rms_min_P1 - d.point_rms_min_2P1) / d[["point_rms_min_P1", "point_rms_min_2P1"]].min(axis=1)
    d["shape_choice"] = np.where(d.rel_diff > 0, 2 * d.period_hr_naive, d.period_hr_naive)
    d["pipeline_ok"] = match(d.period_hr, d.truth_hr)
    d["shape_ok"] = match(d.shape_choice, d.truth_hr)
    d["near_tie"] = d.rel_diff.abs() < NEAR_TIE
    d["truth_is_2P1"] = match(2 * d.period_hr_naive, d.truth_hr)
    for s, g in d.groupby("stratum"):
        print(f"  {s:14s} n {len(g):3d}: truth = 2P1 for {g.truth_is_2P1.mean():.0%}; pipeline correct {g.pipeline_ok.mean():.1%}, "
              f"shape choice correct {g.shape_ok.mean():.1%}; near-ties {g.near_tie.mean():.1%} "
              f"(shape correct outside ties {g[~g.near_tie].shape_ok.mean():.1%})")
    print(f"  all            n {len(d):3d}: pipeline correct {d.pipeline_ok.mean():.1%}, shape choice correct {d.shape_ok.mean():.1%}; "
          f"outside near-ties {d[~d.near_tie].shape_ok.mean():.1%} of {int((~d.near_tie).sum())}")
    print("fixed checks:")
    print(d[d.stratum == "fixed check"][["designation", "truth_hr", "period_hr_naive", "period_hr", "point_rms_min_P1",
                                         "point_rms_min_2P1", "rel_diff", "pipeline_ok", "shape_ok"]].round(5).to_string(index=False))
    d.to_csv(f"{D}/summary.csv", index=False)
    comb = []
    for des in d.designation:
        tag = sn(des)
        ch = {}
        for k in ("P1", "2P1"):
            f = f"{D}/comb_{k}/{tag}.json"
            if os.path.exists(f):
                it = json.load(open(f)).get("iterative", {})
                ch[k] = (it.get("chi2_w"), it.get("period_joint_hr"))
        if len(ch) == 2 and None not in (ch["P1"][0], ch["2P1"][0]):
            row = d[d.designation == des].iloc[0]
            pick = ch["2P1"][1] if ch["2P1"][0] < ch["P1"][0] else ch["P1"][1]
            comb.append(dict(designation=des, stratum=row.stratum, truth_hr=row.truth_hr, chi2_w_P1=ch["P1"][0],
                             chi2_w_2P1=ch["2P1"][0], comb_choice=pick, comb_ok=bool(match(pick, row.truth_hr)),
                             shape_ok=bool(row.shape_ok), pipeline_ok=bool(row.pipeline_ok)))
    if comb:
        c = pd.DataFrame(comb)
        print(f"combined (TESS + sparse) choice by weighted chi2, {len(c)} objects: correct {c.comb_ok.mean():.0%}; "
              f"TESS-only shape choice on the same objects {c.shape_ok.mean():.0%}; pipeline {c.pipeline_ok.mean():.0%}")
        print(c.round(6).to_string(index=False))
        c.to_csv(f"{D}/summary_combined.csv", index=False)


if __name__ == "__main__":
    {"targets": targets, "summary": summary}.get(sys.argv[1], lambda: fit(int(sys.argv[2]), int(sys.argv[3])))()
