"""Does power_scan on co-moving cutouts recover asteroid rotation better than a periodogram of the PSF
photometry, especially for faint objects? Tested per visit (one cut crossing) on

  verified  objects whose independent sector periods agree and match their pooled period
            (agreement_labels.csv label_v2 == 1), 40 per per-frame S/N bin; truth = pooled period
  dropped   objects with photometry that never reach the period report (too faint for snr5 stacking)
            but have total S/N >= 20; no truth, so consistency between their two longest visits

For each object's two longest visits: a 31 x 31 px cube in the asteroid's frame (sub-pixel
registered on its stored positions, each pixel's median subtracted) goes to
power_scan.periodogram_detection over 1 h - 5 d (below 1 h the reduced cubes carry an even/odd-frame
pattern at 20 min), and the visit's own photometry gets a weighted Lomb-Scargle over the same range.
A period is recovered if it, or twice it, is within --tol of the truth (the report adopts 2x the
periodogram peak by default).

    python3 -u power_scan_test.py [--per-bin 40] [--dropped 100]
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.ndimage import shift as subpixel_shift

import power_scan as ps
from tessellate import asteroid_store as st

DATA = '/fred/oz335/TESSdata'
OUT = '/fred/oz335/rridden/asteroids_store_sector/power_scan_test.csv'
HALF = 15
CENTRE_PX = 3.0
P_MIN_D, P_MAX_D = 1 / 24, 5.0
BINS = [0, 0.6, 1, 2, 5, np.inf]


def comoving(cube, track):
    stamps, t = [], []
    for row in track.itertuples():
        xi, yi = int(round(row.x)), int(round(row.y))
        if xi - HALF - 1 < 0 or yi - HALF - 1 < 0 or xi + HALF + 2 > cube.shape[2] or yi + HALF + 2 > cube.shape[1]:
            continue
        s = np.asarray(cube[row.frame, yi - HALF - 1:yi + HALF + 2, xi - HALF - 1:xi + HALF + 2], dtype=float)
        if not np.isfinite(s).all():
            continue
        stamps.append(subpixel_shift(s, (yi - row.y, xi - row.x), order=1, mode='nearest')[1:-1, 1:-1])
        t.append(row.mjd)
    c = np.array(stamps)
    return np.array(t), (c - np.median(c, axis=0)) if len(c) else c


def photometry_period(track):
    t, f, e = track.mjd.values, track.flux_detrended.values.astype(float), track.e_flux.values.astype(float)
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    if ok.sum() < 30:
        return np.nan, np.nan
    freq = np.linspace(1 / P_MAX_D, 1 / P_MIN_D, 20000)
    p = LombScargle(t[ok], f[ok], e[ok]).power(freq)
    k = np.argmax(p)
    return 24 / freq[k], p[k] / np.median(p)


def power_scan_period(t, cube):
    if len(t) < 30:
        return np.nan, np.nan
    det = ps.periodogram_detection(time=t, data=cube, period_lim=[P_MIN_D, P_MAX_D], snr_lim=5, snr_search_lim=5, cpu=8)
    src = det.sources if det.sources is not None else pd.DataFrame()
    if not len(src):
        return np.nan, np.nan
    src = src[np.hypot(src.xcentroid - HALF, src.ycentroid - HALF) <= CENTRE_PX]
    if not len(src):
        return np.nan, np.nan
    best = src.sort_values('local_sig', ascending=False).iloc[0]
    return float(best.period) * 24, float(best.local_sig)


def recovered(p, truth, tol):
    return bool(np.isfinite(p) and np.isfinite(truth) and min(abs(p / truth - 1), abs(2 * p / truth - 1)) < tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-bin', type=int, default=40)
    ap.add_argument('--dropped', type=int, default=100)
    ap.add_argument('--tol', type=float, default=0.02)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    tracks = pd.concat([pd.read_parquet(f, columns=['designation', 'sector', 'cam', 'ccd', 'cut', 'part', 'n_points', 'avg_sig'])
                        for f in glob.glob(f'{DATA}/asteroid_store/tracks/sector*.parquet')])
    tracks = tracks[tracks.n_points > 0]
    per_obj = tracks.groupby('designation').agg(frame_sn=('avg_sig', 'max'), frames=('n_points', 'sum'),
                                                s2=('avg_sig', lambda s: 0))
    tracks['s2'] = np.clip(tracks.avg_sig.astype(float), 0, None) ** 2 * tracks.n_points
    per_obj['total_sn'] = np.sqrt(tracks.groupby('designation').s2.sum())
    labels = pd.read_csv('/fred/oz335/rridden/asteroids_store_sector/agreement_labels.csv', low_memory=False)
    ver = labels[labels.label_v2 == 1].join(per_obj, on='designation')
    sample = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        b = ver[(ver.frame_sn >= lo) & (ver.frame_sn < hi)]
        b = b.sample(min(a.per_bin, len(b)), random_state=1)
        sample += [('verified', d, p, f'{lo}-{hi}') for d, p in zip(b.designation, b.pooled_period_hr)]
    report = pd.read_csv('/fred/oz335/rridden/asteroids_store/all_sector_asteroid_report.csv', usecols=['designation'])
    dropped = per_obj[~per_obj.index.isin(report.designation) & (per_obj.total_sn >= 20)]
    dropped = dropped.sample(min(a.dropped, len(dropped)), random_state=1)
    sample += [('dropped', d, np.nan, 'dropped') for d in dropped.index]
    print(f'{len(sample)} objects', flush=True)

    rows, cubes = [], {}
    for kind, d, truth, sn_bin in sample:
        lc = st.load_lightcurve(d, DATA)
        visits = tracks[tracks.designation == d].sort_values('n_points', ascending=False).head(2)
        for v in visits.itertuples():
            tr = lc[(lc.sector == v.sector) & (lc.cam == v.cam) & (lc.ccd == v.ccd) & (lc.cut == v.cut) & (lc.part == v.part)]
            tr = tr.sort_values('mjd')
            folder = f'{DATA}/Sector{v.sector}/Cam{v.cam}/Ccd{v.ccd}' + (f'/Part{v.part}' if v.part else '') + f'/Cut{v.cut}of64'
            path = f'{folder}/sector{v.sector}_cam{v.cam}_ccd{v.ccd}_cut{v.cut}_of64_ReducedFlux.npy'
            t0 = time.time()
            try:
                cube = np.load(path, mmap_mode='r')
                t, c = comoving(cube, tr)
                ps_p, ps_sig = power_scan_period(t, c)
            except Exception as e:
                print(f'  {d} S{v.sector} C{v.cam}{v.ccd} cut {v.cut}: {type(e).__name__}: {e}', flush=True)
                ps_p, ps_sig = np.nan, np.nan
            ph_p, ph_pk = photometry_period(tr)
            rows.append(dict(kind=kind, designation=d, sn_bin=sn_bin, truth_hr=truth, sector=v.sector, cam=v.cam, ccd=v.ccd,
                             cut=v.cut, frames=len(tr), ps_period_hr=ps_p, ps_local_sig=ps_sig, phot_period_hr=ph_p,
                             phot_peak_over_median=ph_pk, ps_recovered=recovered(ps_p, truth, a.tol),
                             phot_recovered=recovered(ph_p, truth, a.tol), seconds=round(time.time() - t0, 1)))
        print(f'  {kind} {d}: {[(r["ps_period_hr"], r["phot_period_hr"]) for r in rows[-len(visits):]]} truth {truth}', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    v = out[out.kind == 'verified']
    print('\nverified objects, per visit (recovered = period or 2x period within tol of the truth):')
    for b, g in v.groupby('sn_bin', sort=False):
        obj = g.groupby('designation')[['ps_recovered', 'phot_recovered']].any()
        print(f'  frame S/N {b:>8s}: {len(g):3d} visits, power_scan {g.ps_recovered.mean():.0%}, photometry '
              f'{g.phot_recovered.mean():.0%} | per object (either visit) power_scan {obj.ps_recovered.mean():.0%}, '
              f'photometry {obj.phot_recovered.mean():.0%}; power_scan detections {g.ps_period_hr.notna().mean():.0%}')
    dr = out[out.kind == 'dropped']
    for col in ('ps_period_hr', 'phot_period_hr'):
        pairs = dr.groupby('designation')[col].apply(lambda s: s.dropna().to_list()).apply(lambda l: l if len(l) == 2 else None).dropna()
        same = [recovered(p[0], p[1], a.tol) or recovered(p[1], p[0], a.tol) for p in pairs]
        print(f'dropped objects, {col}: {len(pairs)} with a period in both visits, {sum(same)} consistent (same or 2x)')
    print(f'median seconds per visit {out.seconds.median():.1f} -> {OUT}')


if __name__ == '__main__':
    main()
