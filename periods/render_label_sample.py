"""Random sample of the raw-frame pooled report for checking by eye: N objects the deployed classifier
calls reliable (good/) and N with a period that it does not (bad/), each plotted with the pipeline's own
plot_asteroid (lightcurve, periodogram, fold) from the same store photometry and point mode as the run.
The sample and its scores go to sample.csv beside the folders, so deletions can be read back as labels.

    ASTEROID_SAVE_PLOTS=1 python3 -u render_label_sample.py [--n 100] [--out DIR]
    ASTEROID_SAVE_PLOTS=1 python3 -u render_label_sample.py --list objects.csv --label failed --out DIR
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import sector_report_pipeline as p
from tessellate import asteroid_store as st


def plot_one(designation, outpath, h_mag):
    """One object, loaded on its own: a single-designation filter lets the store skip every other row
    group, where one filter over the whole sample reads most of them."""
    phot = st.load_lightcurve(designation, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS)
    track = p.normalise(p.frames_to_points(p._store_frame(phot)))
    p.plot_asteroid(designation, track, outpath, h_mag=h_mag,
                    sbdb_entry=p.match_sbdb(designation, p._worker_sbdb_by_number, p._worker_sbdb_by_designation))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--out', default='/fred/oz335/rridden/asteroids_store/label_sample')
    ap.add_argument('--list', help='CSV with a designation column: plot exactly these, into <out>/<--label>/')
    ap.add_argument('--label', default='listed')
    a = ap.parse_args()
    rep = pd.read_csv(p.SUMMARY_PATH, low_memory=False, usecols=['designation', 'period_hr', 'reliability_score', 'is_reliable'])
    rep = rep[np.isfinite(rep.period_hr)]
    if a.list:
        sample = rep[rep.designation.isin(pd.read_csv(a.list).designation)].assign(cls=a.label).reset_index(drop=True)
    else:
        good = rep[rep.is_reliable.astype(bool)].sample(a.n, random_state=7).assign(cls='good')
        bad = rep[~rep.is_reliable.astype(bool)].sample(a.n, random_state=7).assign(cls='bad')
        sample = pd.concat([good, bad], ignore_index=True)
    sample['file'] = [f'{c}/{s:.3f}_P{ph:08.3f}h_{p.safe_name(d)}.png'
                      for c, s, ph, d in zip(sample.cls, sample.reliability_score, sample.period_hr, sample.designation)]
    for c in sample.cls.unique():
        os.makedirs(f'{a.out}/{c}', exist_ok=True)
    sample.to_csv(f'{a.out}/sample.csv', index=False)

    objects = st.load_objects(p.DATA_ROOT)
    objects = objects[objects.designation.isin(sample.designation)]
    h_lookup = {d: float(h) for d, h in zip(objects.designation, objects.magnitude_H) if np.isfinite(h)}
    sbdb = p.load_sbdb_physical_properties()
    n = 0
    with ProcessPoolExecutor(max_workers=p.n_available_workers(), initializer=p._pool_worker_init,
                             initargs=(h_lookup, *sbdb)) as pool:
        futures = {pool.submit(plot_one, d, f'{a.out}/{f}', h_lookup.get(d)): d
                   for d, f in zip(sample.designation, sample.file)}
        for fut in as_completed(futures):
            try:
                fut.result()
                n += 1
            except Exception as e:
                print(f'  FAILED {futures[fut]}: {e}', flush=True)
    print(f'{n} of {len(sample)} plotted -> {a.out}/' + ', '.join(sample.cls.unique()), flush=True)


if __name__ == '__main__':
    main()
