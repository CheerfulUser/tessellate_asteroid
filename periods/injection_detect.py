"""Classifier label for the synthetic set when identification (harmonic, alias) and precision are left to the shape
stage: the object's data contain the injected signal (detection) AND the pipeline's period is one the shape stage can
reach the truth from. Runs on ozstar.

  power <run> <bucket>   detection statistic for every object of one photometry bucket: the 1-term Lomb-Scargle power
                         (pre-clipped points, as the pipeline's naive search) at k f_inj, k = 1..4, maximised over
                         +-0.5 peak width, converted to a single-frequency false-alarm probability (1 - z)^((N-3)/2)
                         times the trials (K_MAX x LOCAL); blank / constant objects get a stand-in period drawn like the
                         injections (log-uniform P_MIN..P_MAX h, seeded by object number) to give the null
                         -> blank_injection/detect/<run>_<bucket>.parquet
  label                  detected = FAP below the level only NULL_FRAC of blank paths reach (empirical, so correlated
                         noise is included); reach = pipeline period within +-SCAN of P_inj x {1, 2, 1/2} (the
                         combined fit's period-scan range around the P / 2P candidates the shape stage tests);
                         label reach_detect = reach & detected. Chance rate of reach from the blanks' stand-in periods,
                         by per-frame S/N. -> blank_injection/synthetic_training_set_all_detect.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

import sector_report_pipeline as p

B = "/fred/oz335/rridden/asteroids_store/blank_injection"
K_MAX = 4
LOCAL = 11
SCAN = 0.08829
NULL_FRAC = 0.01
P_MIN, P_MAX = 1.0, 500.0
HARM = (1.0, 2.0, 0.5)


def power(run, bucket):
    os.makedirs(f"{B}/detect", exist_ok=True)
    objs = pd.read_csv(f"{B}/{run}/objects.csv").set_index("syn")
    files = glob.glob(f"{B}/{run}/phot/bucket={bucket}/*.parquet")
    if not files:
        print(f"{run} {bucket}: no photometry"); return
    ph = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows = []
    for d, g in ph.groupby("designation"):
        o = objs.loc[d]
        try:
            tr = p.normalise(p.frames_to_points(p._store_frame(g))).sort_values("mjd").reset_index(drop=True)
            tr = tr[p.preclip_mask(tr)]
            t, y, e = tr.mjd.values, tr.rel_flux.values, tr.rel_flux_err.values
            if len(t) < p.MIN_POINTS_FOR_PERIODOGRAM:
                continue
            if o.kind == "rotator":
                P_true, stand_in = o.period_inj_hr, False
            else:
                rng = np.random.default_rng(int(o.num))
                P_true, stand_in = float(np.exp(rng.uniform(np.log(P_MIN), np.log(P_MAX)))), True
            f0 = 24 / P_true
            width = 1 / np.ptp(t)
            fr = np.concatenate([k * f0 + np.linspace(-0.5, 0.5, LOCAL) * width for k in range(1, K_MAX + 1)])
            fr = fr[(fr > 1 / p.MAX_PERIOD_DAYS) & (fr < 1 / p.MIN_PERIOD_DAYS)]
            if not len(fr):
                continue
            z = float(np.max(LombScargle(t, y, e).power(fr)))
            n = len(t)
            fap = min(1.0, (1 - min(z, 1 - 1e-15)) ** ((n - 3) / 2) * K_MAX * LOCAL)
            rows.append(dict(syn=d, run=run, P_ref_hr=P_true, stand_in=stand_in, det_power=z, det_fap=fap, n_det=n))
        except Exception as ex:
            print(f"  {d}: {type(ex).__name__}: {ex}", flush=True)
    pd.DataFrame(rows).to_parquet(f"{B}/detect/{run}_{bucket:03d}.parquet", index=False)
    print(f"{run} bucket {bucket}: {len(rows)} objects", flush=True)


def label():
    det = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{B}/detect/*.parquet")], ignore_index=True)
    d = pd.read_csv(f"{B}/synthetic_training_set_all.csv", low_memory=False)
    d = d.merge(det.drop(columns=["run"]), on="syn", how="left")
    ok = d.error.isna() & d.period_hr.notna() & d.det_fap.notna()
    blanks = ok & (d.kind == "blank")
    thr = float(np.quantile(np.log10(np.clip(d.det_fap[blanks], 1e-300, 1)), NULL_FRAC))
    d["detected"] = ok & (np.log10(np.clip(d.det_fap, 1e-300, 1)) < thr)
    ratio = d.period_hr / d.P_ref_hr
    within = np.any([(ratio / q - 1).abs() < SCAN for q in HARM], axis=0)
    d["reach"] = ok & (d.kind == "rotator") & within
    d["reach_chance"] = ok & d.stand_in.fillna(False).astype(bool) & within      # blanks / constants: coincidence
    d["reach_detect"] = d.reach & d.detected
    d.to_csv(f"{B}/synthetic_training_set_all_detect.csv", index=False)
    rot, null = d[ok & (d.kind == "rotator")], d[ok & d.stand_in.fillna(False).astype(bool)]
    print(f"detection threshold: log10 FAP < {thr:.2f} (only {NULL_FRAC:.0%} of {int(blanks.sum()):,} blank paths reach it); "
          f"constants detected {d[ok & (d.kind == 'constant')].detected.mean():.1%}")
    band = lambda x: pd.cut(x.s_frame, [0.1, 0.5, 1, 2, 5, 20, 50])
    t = pd.DataFrame({"rotators: detected": rot.groupby(band(rot), observed=True).detected.mean(),
                      "rotators: reach": rot.groupby(band(rot), observed=True).reach.mean(),
                      "chance reach (stand-in periods)": null.groupby(band(null), observed=True).reach_chance.mean(),
                      "label reach & detected": rot.groupby(band(rot), observed=True).reach_detect.mean()})
    print(t.round(3).to_string())
    print(f"all rotators: detected {rot.detected.mean():.1%}, reach {rot.reach.mean():.1%}, label {rot.reach_detect.mean():.1%}; "
          f"chance reach overall {null.reach_chance.mean():.1%}; reach but not detected {(rot.reach & ~rot.detected).mean():.1%}")


if __name__ == "__main__":
    if sys.argv[1] == "power":
        power(sys.argv[2], int(sys.argv[3]))
    else:
        label()
