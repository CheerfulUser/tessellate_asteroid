"""Shape-model target list from the reliable objects of the current store period report (designation, period_hr,
n_points), shuffled so every batch_shapes array task gets a mix of sizes.
    python3 make_shape_targets.py <out.csv>"""
import sys
import pandas as pd
r = pd.read_csv("/fred/oz335/rridden/asteroids_store/all_sector_asteroid_report.csv", low_memory=False,
                usecols=["designation", "period_hr", "is_reliable", "n_points"])
r = r[r.is_reliable.astype(bool) & r.period_hr.notna()].sample(frac=1, random_state=0)
r[["designation", "period_hr", "n_points"]].to_csv(sys.argv[1], index=False)
print(len(r), "targets ->", sys.argv[1])
