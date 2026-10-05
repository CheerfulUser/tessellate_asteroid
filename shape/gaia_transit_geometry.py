"""Viewing geometry for every Gaia DR3 transit of the catalogue asteroids, for the TESS + ATLAS + Gaia
inversion (combined_tess_atlas.py reads it instead of querying JPL Horizons per object).

Input (GAIA_DIR, default /fred/oz335/TESSdata/mpc/gaia): gaia_dr3_sso/bucket_XXX.parquet, the per-transit
Gaia DR3 photometry with Gaia's barycentric position (convert_gaia_sso_to_parquet.py).
Orbits: MPCORB integrated with ASSIST (planets + its 16 perturbing asteroids), using TESSELLATE's
asteroid_prediction machinery (TESSELLATE_DIR on sys.path; ASSIST/MPCORB data in DATA_DIR). The 16
perturbers themselves are read from ASSIST's ephemeris, as asteroid_prediction does. Each transit is
light-time corrected to Gaia's own position. Checked against JPL Horizons on (130), (165), (554), (22):
positions agree to <= 300 km, distance to 4e-7, Sun direction to < 0.2 arcsec (a two-body MPCORB orbit
was off by up to 0.08 au and 1.8 deg).

Output, per transit (asteroid-centric, ecliptic J2000): sx, sy, sz (to the Sun), ex, ey, ez (to Gaia),
R, delta (au), SOE (phase angle, deg), MJD_lc (light-time-corrected epoch, UTC MJD).
  task mode:   python gaia_transit_geometry.py <numbers.txt> <task> <ntasks>  -> GAIA_DIR/geometry_tasks/
  merge mode:  python gaia_transit_geometry.py --merge -> GAIA_DIR/gaia_dr3_geometry/bucket_XXX.parquet
                                                          (bucket = number % 256, like the transits)
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

GAIA_DIR = os.environ.get("GAIA_DIR", "/fred/oz335/TESSdata/mpc/gaia")
DATA_DIR = os.environ.get("DATA_DIR", "/fred/oz335/TESSdata/mpc")
sys.path.insert(0, os.environ.get("TESSELLATE_DIR", "/fred/oz335/rridden/code/TESSELLATE"))
GEOM_TASKS = os.environ.get("GEOM_TASKS", f"{GAIA_DIR}/geometry_tasks")   # per-task output (test runs elsewhere)
NB = 256
C_AU_PER_DAY = 173.1446326846693
_OBL = np.radians(23.4392911)                     # J2000 mean obliquity, as batch_shapes
ICRS_TO_ECL = np.array([[1, 0, 0], [0, np.cos(_OBL), np.sin(_OBL)], [0, -np.sin(_OBL), np.cos(_OBL)]])
B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def pack_number(n):
    if n < 100000:
        return f"{n:05d}"
    if n < 620000:
        return B62[n // 10000] + f"{n % 10000:04d}"
    n -= 620000
    s = ""
    for _ in range(4):
        s = B62[n % 62] + s
        n //= 62
    return "~" + s


def geometry(ap, row, number, jd_utc, gaia_bary):
    """Asteroid and Sun barycentric (ICRF, au) at the light-time-corrected emission time of each transit."""
    import assist
    import rebound
    from astropy.time import Time
    ephem = ap.load_assist_ephem(DATA_DIR)
    pidx = ap.assist_perturber_indices(DATA_DIR).get(number) if number in ap.ASSIST_PERTURBER_NUMBERS else None
    if pidx is None:
        t_ref, hp, hv = ap._state_vector_at_epoch(row, DATA_DIR)
        t0 = t_ref.tdb - ephem.jd_ref
        s = ephem.get_particle("Sun", t0)
        sim = rebound.Simulation()
        assist.Extras(sim, ephem)
        sim.t = t0
        sim.add(x=hp[0] + s.x, y=hp[1] + s.y, z=hp[2] + s.z, vx=hv[0] + s.vx, vy=hv[1] + s.vy, vz=hv[2] + s.vz)
    T = Time(jd_utc, format="jd", scale="utc").tdb.jd - ephem.jd_ref
    ast, sun, lt = np.empty((len(T), 3)), np.empty((len(T), 3)), np.zeros(len(T))
    tau = 0.0
    for i in np.argsort(T):
        for _ in range(ap.LT_MAX_ITER):
            if pidx is None:
                sim.integrate(T[i] - tau)
                p = sim.particles[0]
            else:
                p = ephem.get_particle(pidx, T[i] - tau)
            o = np.array([p.x, p.y, p.z])
            new = np.linalg.norm(o - gaia_bary[i]) / C_AU_PER_DAY
            done = abs(new - tau) < ap.LT_TOL_DAYS
            tau = new
            if done:
                break
        ast[i], lt[i] = o, tau
        q = ephem.get_particle("Sun", T[i] - tau)
        sun[i] = [q.x, q.y, q.z]
    return ast, sun, lt


def run_task(numbers_file, task, ntasks):
    from tessellate import asteroid_prediction as ap
    wanted = {int(x) for x in open(numbers_file).read().split()}
    objs = pd.read_parquet(f"{GAIA_DIR}/gaia_dr3_sso_objects.parquet")
    todo = sorted(set(objs.number_mp) & wanted)[task::ntasks]
    mpc = ap.load_mpcorb(DATA_DIR)
    by_packed = mpc.set_index("designation_packed")
    out, t0, bad = [], time.time(), 0
    for i, n in enumerate(todo):
        try:
            tr = pd.read_parquet(f"{GAIA_DIR}/gaia_dr3_sso/bucket_{n % NB:03d}.parquet", filters=[("number_mp", "==", n)])
            tr = tr[np.isfinite(tr.epoch_utc)]
            jd = tr.epoch_utc.values + (2455197.5 if tr.epoch_utc.max() < 1e6 else 0.0)   # DR3: JD - 2455197.5
            gaia = tr[["x_gaia", "y_gaia", "z_gaia"]].values
            row = by_packed.loc[pack_number(n)]
            ast, sun, lt = geometry(ap, row, n, jd, gaia)
            sv, ev = (sun - ast) @ ICRS_TO_ECL.T, (gaia - ast) @ ICRS_TO_ECL.T
            R, D = np.linalg.norm(sv, axis=1), np.linalg.norm(ev, axis=1)
            out.append(pd.DataFrame(dict(
                number_mp=n, transit_id=tr.transit_id.values, MJD_lc=jd - lt - 2400000.5,
                sx=sv[:, 0], sy=sv[:, 1], sz=sv[:, 2], ex=ev[:, 0], ey=ev[:, 1], ez=ev[:, 2], R=R, delta=D,
                SOE=np.degrees(np.arccos(np.clip(np.sum(sv * ev, 1) / (R * D), -1, 1))))))
        except Exception as e:
            bad += 1
            print(f"  ({n}): {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)} objects, {(time.time() - t0) / 60:.1f} min", flush=True)
    os.makedirs(GEOM_TASKS, exist_ok=True)
    if out:
        pd.concat(out, ignore_index=True).to_parquet(f"{GEOM_TASKS}/task_{task:04d}.parquet", index=False)
    print(f"task {task}: {len(todo) - bad} objects done, {bad} failed, {(time.time() - t0) / 60:.1f} min", flush=True)


def merge():
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{GEOM_TASKS}/task_*.parquet"))],
                  ignore_index=True)
    os.makedirs(f"{GAIA_DIR}/gaia_dr3_geometry", exist_ok=True)
    for b, part in d.groupby(d.number_mp % NB):
        part.to_parquet(f"{GAIA_DIR}/gaia_dr3_geometry/bucket_{b:03d}.parquet", index=False)
    print(f"merged {len(d):,} transits of {d.number_mp.nunique():,} objects into {d.number_mp.mod(NB).nunique()} buckets; "
          f"phase angle {d.SOE.min():.1f}-{d.SOE.max():.1f} deg, R {d.R.min():.2f}-{d.R.max():.2f} au", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge()
    else:
        run_task(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
