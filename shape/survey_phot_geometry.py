"""Viewing geometry for the MPC survey photometry (convert_mpc_obs_to_parquet.py) of the shape-model targets,
for the TESS + sparse-survey inversion. Same orbits and light-time correction as gaia_transit_geometry.py
(MPCORB integrated with ASSIST); the observer is Earth's centre, as for ATLAS (the stations' offset from it,
< 5e-5 au, is far below what the inversion resolves).

Input (SURVEY_DIR, default /fred/oz335/TESSdata/mpc/mpc_obs): survey_phot/bucket_XXX.parquet.
Output, per observation (asteroid-centric, ecliptic J2000), with the photometry columns:
  sx, sy, sz (to the Sun), ex, ey, ez (to Earth), R, delta (au), SOE (phase angle, deg), MJD_lc
  (light-time-corrected epoch, UTC MJD)
  task mode:   python survey_phot_geometry.py <numbers.txt> <task> <ntasks>  -> SURVEY_DIR/geometry_tasks/
  merge mode:  python survey_phot_geometry.py --merge -> SURVEY_DIR/survey_phot_geometry/bucket_XXX.parquet
                                                         (bucket = number % 256)
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gaia_transit_geometry as gg  # noqa: E402

SURVEY_DIR = os.environ.get("SURVEY_DIR", "/fred/oz335/TESSdata/mpc/mpc_obs")
GEOM_TASKS = os.environ.get("GEOM_TASKS", f"{SURVEY_DIR}/geometry_tasks")
NB = 256


def run_task(numbers_file, task, ntasks):
    from astropy.time import Time
    from tessellate import asteroid_prediction as ap
    todo = sorted({int(x) for x in open(numbers_file).read().split()})[task::ntasks]
    mpc = ap.load_mpcorb(gg.DATA_DIR)
    by_packed = mpc.set_index("designation_packed")
    ephem = ap.load_assist_ephem(gg.DATA_DIR)
    out, t0, bad, empty = [], time.time(), 0, 0
    for i, n in enumerate(todo):
        try:
            ph = pd.read_parquet(f"{SURVEY_DIR}/survey_phot/bucket_{n % NB:03d}.parquet", filters=[("number", "==", n)])
            if not len(ph):
                empty += 1
                continue
            jd = ph.mjd_utc.values + 2400000.5
            T = Time(jd, format="jd", scale="utc").tdb.jd - ephem.jd_ref
            earth = np.array([[p.x, p.y, p.z] for p in (ephem.get_particle("Earth", t) for t in T)])
            ast, sun, lt = gg.geometry(ap, by_packed.loc[gg.pack_number(n)], n, jd, earth)
            sv, ev = (sun - ast) @ gg.ICRS_TO_ECL.T, (earth - ast) @ gg.ICRS_TO_ECL.T
            R, D = np.linalg.norm(sv, axis=1), np.linalg.norm(ev, axis=1)
            out.append(ph.assign(MJD_lc=ph.mjd_utc.values - lt, sx=sv[:, 0], sy=sv[:, 1], sz=sv[:, 2],
                                 ex=ev[:, 0], ey=ev[:, 1], ez=ev[:, 2], R=R, delta=D,
                                 SOE=np.degrees(np.arccos(np.clip(np.sum(sv * ev, 1) / (R * D), -1, 1)))))
        except Exception as e:
            bad += 1
            print(f"  ({n}): {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)} objects, {(time.time() - t0) / 60:.1f} min", flush=True)
    os.makedirs(GEOM_TASKS, exist_ok=True)
    if out:
        pd.concat(out, ignore_index=True).to_parquet(f"{GEOM_TASKS}/task_{task:04d}.parquet", index=False)
    print(f"task {task}: {len(todo) - bad - empty} objects done, {empty} without survey photometry, {bad} failed, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


def merge():
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{GEOM_TASKS}/task_*.parquet"))], ignore_index=True)
    os.makedirs(f"{SURVEY_DIR}/survey_phot_geometry", exist_ok=True)
    for b, part in d.groupby(d.number % NB):
        part.to_parquet(f"{SURVEY_DIR}/survey_phot_geometry/bucket_{b:03d}.parquet", index=False)
    print(f"merged {len(d):,} observations of {d.number.nunique():,} objects; phase angle "
          f"{d.SOE.min():.1f}-{d.SOE.max():.1f} deg, R {d.R.min():.2f}-{d.R.max():.2f} au", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge()
    else:
        run_task(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
