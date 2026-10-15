"""Write the cleaned points (pre-clipped, per-visit normalised) of the labelled objects whose hybrid Vavilov period
(vavilov_test.csv) is unrelated to the accepted one, for local comparison plots (vavilov_compare_plot.py).
    python3 -u vavilov_extract_points.py good_unrelated     # -> asteroids_store/vav_compare/good_unrelated/
"""
import os, sys
import numpy as np, pandas as pd
import sector_report_pipeline as p
from tessellate import asteroid_store as st

S = "/fred/oz335/rridden/asteroids_store"
HARM_Q = (2, 0.5, 3, 1 / 3, 1.5, 2 / 3)


def main():
    sel = sys.argv[1]
    out = f"{S}/vav_compare/{sel}"
    os.makedirs(out, exist_ok=True)
    v = pd.read_csv(f"{S}/vavilov_test.csv").drop_duplicates("designation")
    r = v.vav_period_hr / v.labelled_period_hr
    same = (r - 1).abs() < 0.02
    harm = sum((r / q - 1).abs().lt(0.02) for q in HARM_Q).astype(bool)
    good = v.label_final == "good"
    v = v[good & ~same & ~harm] if sel == "good_unrelated" else v[good & harm & ~same]
    v.to_csv(f"{out}/vavilov_results.csv", index=False)
    for d in v.designation:
        tr = p.normalise(p.frames_to_points(p._store_frame(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))))
        tr = tr.sort_values("mjd").reset_index(drop=True)
        tr = tr[p.preclip_mask(tr)]
        name = d.replace("(", "").replace(")", "").replace(" ", "_")
        tr[["mjd", "rel_flux", "rel_flux_err", "visit"]].to_csv(f"{out}/{name}_points.csv", index=False)
    print(f"{len(v)} objects -> {out}")


if __name__ == "__main__":
    main()
