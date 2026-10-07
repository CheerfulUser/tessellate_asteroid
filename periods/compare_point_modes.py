"""Compare how frames become period-search points (ASTEROID_POINT_MODE: snr5 / raw / bin30) on the same
designation ranges (store tasks 0, 250, 500, 700 of 1,000). The classifier was trained on snr5 points,
so its scores are not comparable across modes; the yardstick is label-free: objects whose independent
sector periods all agree within --tol and match their pooled period ("verified"), split by the
object's per-frame S/N (from the asteroid store tracks).

    python3 compare_point_modes.py [--tol 0.02]
"""
import argparse
import glob

import numpy as np
import pandas as pd

TASKS = (0, 250, 500, 700)
BASE = '/fred/oz335/rridden/asteroids'
MODES = {'snr5': ('_store', '_store_sector'), 'raw': ('_store_raw', '_store_raw_sector'),
         'bin30': ('_store_bin30', '_store_bin30_sector')}


def parts(suffix):
    return pd.concat([pd.read_csv(f'{BASE}{suffix}/_summary_parts/part_{t:05d}.csv', low_memory=False) for t in TASKS],
                     ignore_index=True)


def verified(sector, pooled, tol):
    s = sector[np.isfinite(sector.period_hr)]
    rows = []
    for d, g in s.groupby('designation'):
        p = g.period_hr.to_numpy()
        if len(p) < 2:
            continue
        agree = all(abs(p[i] / p[j] - 1) < tol for i in range(len(p)) for j in range(i + 1, len(p)))
        rows.append((d, agree, float(np.median(p))))
    v = pd.DataFrame(rows, columns=['designation', 'agree', 'p_sector']).merge(
        pooled[['designation', 'period_hr']], on='designation', how='left')
    v['verified'] = v.agree & (np.abs(v.period_hr / v.p_sector - 1) < tol)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tol', type=float, default=0.02)
    a = ap.parse_args()
    tracks = pd.concat([pd.read_parquet(f, columns=['designation', 'n_points', 'avg_sig'])
                        for f in glob.glob('/fred/oz335/TESSdata/asteroid_store/tracks/sector*.parquet')])
    tracks = tracks[tracks.n_points > 0]
    frame_sn = tracks.groupby('designation').avg_sig.max().rename('frame_sn')
    bins = [0, 0.3, 0.6, 1, 2, 5, np.inf]
    sets = {}
    for mode, (ps, ss) in MODES.items():
        pooled, sector = parts(ps), parts(ss)
        v = verified(sector, pooled, a.tol).join(frame_sn, on='designation')
        sets[mode] = set(v.designation[v.verified])
        print(f'{mode:6s} pooled: {len(pooled):6,} objects, {int(pooled.period_hr.notna().sum()):6,} with a period, '
              f'{int(pooled.is_reliable.astype(bool).sum()):5,} classifier-reliable | per-sector: '
              f'{len(v):5,} objects with 2+ sector periods, {int(v.agree.sum()):5,} agree ({v.agree.mean():.1%}), '
              f'{int(v.verified.sum()):5,} verified')
        cut = pd.cut(v.frame_sn, bins)
        print('        verified by per-frame S/N: ' + ', '.join(
            f'{str(k)}: {int(x)}' for k, x in v[v.verified].groupby(cut[v.verified], observed=False).size().items()))
    for m in ('raw', 'bin30'):
        print(f'{m} vs snr5: verified in both {len(sets[m] & sets["snr5"]):,}, only {m} {len(sets[m] - sets["snr5"]):,}, '
              f'only snr5 {len(sets["snr5"] - sets[m]):,}')


if __name__ == '__main__':
    main()
