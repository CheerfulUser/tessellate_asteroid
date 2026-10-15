"""Faint labelling round: objects with per-frame S/N < 1 (largest visit avg_sig in the store tracks) that have a
period and pass the physical gates (fold, edge), excluding everything already labelled (labels_final.csv and every
label_batches/labels_faint*.csv): N/5 the deployed classifier passes (random) and 4N/5 it fails, N/5 from each
quartile of the failing scores. Writes asteroids_store/label_batches/<name>.csv (designations, shuffled) and
<name>_sample.csv (with scores and strata). Runs on ozstar.
    python3 -u faint_label_batch.py faint3 2000
"""
import glob, sys
import numpy as np, pandas as pd
import sector_report_pipeline as p

S = "/fred/oz335/rridden/asteroids_store"
OUT = f"{S}/label_batches"
SEED = 11


def main():
    r = pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False)
    r = r[r.period_hr.notna()].drop_duplicates("designation")
    gated = r[(r.fold_min_upper >= 0) & ~r.period_at_search_edge.astype(bool) & r.reliability_score.notna()]
    t = pd.concat([pd.read_parquet(f, columns=["designation", "avg_sig", "n_points"])
                   for f in glob.glob("/fred/oz335/TESSdata/asteroid_store/tracks/sector*.parquet")])
    fsn = t[t.n_points > 0].groupby("designation").avg_sig.max().rename("frame_sn")
    faint = gated.join(fsn, on="designation")
    faint = faint[faint.frame_sn < 1]
    name, n = sys.argv[1], int(sys.argv[2])
    done = pd.concat([pd.read_csv(f"{OUT}/labels_final.csv").designation]
                     + [pd.read_csv(f).designation for f in glob.glob(f"{OUT}/labels_faint*.csv")])
    faint = faint[~faint.designation.isin(done)]
    th = p.reliability_threshold()
    passed, failed = faint[faint.reliability_score >= th], faint[faint.reliability_score < th]
    q = pd.qcut(failed.reliability_score, 4, labels=False)
    sel = pd.concat([passed.sample(min(n // 5, len(passed)), random_state=SEED).assign(stratum="pass")]
                    + [failed[q == k].sample(n // 5, random_state=SEED).assign(stratum=f"fail_q{k + 1}") for k in range(4)])
    sel = sel.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sel[["designation", "period_hr", "reliability_score", "frame_sn", "stratum"]].to_csv(f"{OUT}/{name}_sample.csv", index=False)
    sel[["designation"]].to_csv(f"{OUT}/{name}.csv", index=False)
    edges = failed.reliability_score.quantile([0, 0.25, 0.5, 0.75, 1]).round(3).tolist()
    print(f"faint pool (gated, unlabelled) {len(faint):,}: passes {len(passed):,}, fails {len(failed):,}; threshold {th:.3f}; "
          f"failing-score quartile edges {edges}")
    print(sel.stratum.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
