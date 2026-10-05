"""Does the fixed-pole scan actually choose a pole, or is the choice degenerate -- and does it
prefer poles near the ecliptic plane? For a random sample of catalogue objects, run the same
12-orientation scan as batch_shapes.process (fixed code: ecliptic frame, relative lcs, light-time)
and record every candidate's ecliptic latitude and folded rms, not just the winner.

Motivation: against occultation chords, the scan's chosen poles for (130) Elektra and (22) Kalliope
were in the ecliptic plane (beta = 0) while the measured DAMIT poles put Elektra's near the
ecliptic south pole, rotating the predicted silhouette by ~90 deg.

Run on ozstar from /fred/oz335/rridden/asteroids as a SLURM array:
    python pole_scan_sample.py <task> <ntasks> <nsample> <outdir>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import batch_shapes as bs  # noqa: E402

task, ntasks, nsample, outdir = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
work = f"{outdir}/work_{task}"
os.makedirs(work, exist_ok=True)
targets = pd.read_csv("shapes/targets.csv", low_memory=False).sample(nsample, random_state=7)
mine = targets.iloc[task::ntasks]

rows = []
for t in mine.itertuples():
    tag = bs.sn(t.designation)
    lc = f"clean_lightcurves/{tag}_clean_lc.csv"
    try:
        df = bs.prepare(pd.read_csv(lc))
        if len(df) < 200:
            continue
        sv, ev = bs.build_geometry(df)
        blocks = bs.session_blocks(df)
        lcs = f"{work}/{tag}_lcs.txt"
        bs.write_lcs(df, sv, ev, blocks, lcs)
        en = (ev / np.linalg.norm(ev, axis=1, keepdims=True)).mean(0)
        target_beta = float(np.degrees(np.arcsin(en[2] / np.linalg.norm(en))))
        for i, (lam, bet) in enumerate(bs.perpendicular_poles(ev)):
            cp, sp, pp, fp = (f"{work}/{tag}_{k}{i}.txt" for k in "cspf")
            bs.write_control(cp, lam, bet, t.period_hr)
            res, why = bs.run_pole(cp, lcs, sp, pp, fp)
            brms = bs.folded_rms(df, np.loadtxt(fp), res[2], blocks) if res else np.nan
            rows.append(dict(designation=t.designation, cand=i, lam=lam, bet=bet, brms=brms,
                             observer_ecl_lat=target_beta, failed=why if res is None else ""))
    except Exception as e:  # keep going; record nothing for this object
        print(f"{tag}: {type(e).__name__}: {e}", flush=True)
    finally:
        for f in os.listdir(work):
            if f.startswith(tag + "_"):
                os.remove(os.path.join(work, f))
    print(f"{tag} done", flush=True)
pd.DataFrame(rows).to_csv(f"{outdir}/pole_scan_{task:03d}.csv", index=False)
