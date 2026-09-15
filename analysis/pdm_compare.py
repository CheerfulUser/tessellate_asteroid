"""Decide our period vs the LCDB period by phase-bin scatter (phase dispersion minimisation).

The idea: fold the lightcurve at a candidate period and bin in phase. At the CORRECT period,
points landing in the same phase bin are sampling the same rotational aspect, so the scatter
within bins is small; at a wrong period they sample random aspects and the within-bin scatter
rises to the full lightcurve scatter. The statistic is

    theta = s2_within / s2_total

which tends to 1 for a meaningless period and towards the photometric noise floor for the right
one. Lower wins. This is independent of the periodogram, so it is a genuinely separate line of
evidence from the Lomb-Scargle power the figures already show.

TWO LIMITS, both real and both reported rather than hidden:
  1. PDM CANNOT separate P from 2P. Folding a single-peaked curve at twice its period gives two
     identical half-cycles and exactly the same within-bin scatter. So for the HARM class this
     test is uninformative by construction, and its verdict there is suppressed.
  2. Bin occupancy differs between two very different periods, and a sparsely-occupied fold can
     look artificially tight. Bin count is therefore set from the point count, bins with fewer
     than MIN_PER_BIN points are dropped, and any object whose two folds differ too much in
     usable bins is flagged rather than scored.

Objects the user previously flagged as bad (period not believed in the 800-object round) are
excluded -- a scatter comparison between two periods is meaningless when the lightcurve itself
is junk.
"""
import os, re
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC = 'lcdb_lcs/stacked'
MIN_PER_BIN = 4
MIN_BINS = 8


def sn(d):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(d)).strip('_')


def theta(t, f, P_hr, nb):
    """PDM statistic: pooled within-bin variance over total variance."""
    P = P_hr / 24.0
    if not np.isfinite(P) or P <= 0:
        return np.nan, 0
    ph = (t % P) / P
    b = np.clip((ph * nb).astype(int), 0, nb - 1)
    tot = np.var(f, ddof=1)
    if tot <= 0:
        return np.nan, 0
    num = den = 0.0
    used = 0
    for k in range(nb):
        m = b == k
        n = int(m.sum())
        if n < MIN_PER_BIN:
            continue
        num += (n - 1) * np.var(f[m], ddof=1)
        den += (n - 1)
        used += 1
    if used < MIN_BINS or den <= 0:
        return np.nan, used
    return (num / den) / tot, used


def one(row):
    des = row['designation']
    p = f'{SRC}/{sn(des)}_lc.csv'
    if not os.path.exists(p):
        return None
    try:
        d = pd.read_csv(p).dropna(subset=['mjd', 'rel_flux'])
        if len(d) < 60:
            return None
        t = d['mjd'].values
        f = d['rel_flux'].values
        # clip the extreme tails so one bad point cannot dominate a variance ratio
        lo, hi = np.percentile(f, [0.5, 99.5])
        k = (f >= lo) & (f <= hi)
        t, f = t[k], f[k]
        nb = int(np.clip(len(t) // 12, 10, 40))
        th_o, nu_o = theta(t, f, float(row['period_hr']), nb)
        th_l, nu_l = theta(t, f, float(row['published_rot_per_hr']), nb)
        if not np.isfinite(th_o) or not np.isfinite(th_l):
            return None
        # NULL DISTRIBUTION. "Our period folds tighter" is circular -- our period was chosen to
        # fit THIS data, LCDB's came from different data, so ours is expected to win. The
        # non-circular question is whether the LCDB period is detectable in our data at all:
        # draw random periods spanning the same range and see where theta_lcdb falls. A
        # theta_lcdb sitting inside the random distribution means the literature period leaves
        # no trace in our lightcurve, which is a statement about LCDB, not about our fitting.
        rng = np.random.RandomState(abs(hash(str(des))) % (2**31))
        lo_p, hi_p = 1.0, min(240.0, (t.max() - t.min()) * 24 / 2)
        nulls = []
        for _ in range(60):
            Pr = float(np.exp(rng.uniform(np.log(lo_p), np.log(max(hi_p, lo_p * 2)))))
            th_r, nu_r = theta(t, f, Pr, nb)
            if np.isfinite(th_r):
                nulls.append(th_r)
        if len(nulls) < 20:
            return None
        nulls = np.array(nulls)
        null_med, null_lo = float(np.median(nulls)), float(np.percentile(nulls, 5))
        # z-like position of each candidate within the null
        ns = nulls.std() or 1e-9
        z_o = (null_med - th_o) / ns
        z_l = (null_med - th_l) / ns
        return dict(designation=des, cls=row['cls'], n_points=len(t), n_bins=nb,
                    aov_F=row['aov_F'], reliability_score=row['reliability_score'],
                    period_hr=row['period_hr'], lcdb_hr=row['published_rot_per_hr'],
                    ratio=float(row['period_hr']) / float(row['published_rot_per_hr']),
                    theta_ours=th_o, theta_lcdb=th_l,
                    null_med=null_med, null_p5=null_lo, z_ours=z_o, z_lcdb=z_l,
                    lcdb_detected=bool(th_l < null_lo),
                    ours_detected=bool(th_o < null_lo),
                    bins_ours=nu_o, bins_lcdb=nu_l,
                    # relative improvement; positive means OUR period has the tighter fold
                    gain=(th_l - th_o) / max(th_l, 1e-12))
    except Exception:
        return None


if __name__ == '__main__':
    d = pd.read_csv('comparison_data/lcdb_disagree_render.csv', low_memory=False)
    bad = set(pd.read_csv('comparison_data/flagged_bad.csv')['designation'])
    d = d[~d.designation.isin(bad)]
    print(f'{len(d):,} disagreement objects after excluding {len(bad):,} previously flagged bad',
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(one, r) for _, r in d.iterrows()]
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            if r:
                rows.append(r)
            if (i + 1) % 500 == 0:
                print(f'  {i+1:,}/{len(d):,}', flush=True)
    t = pd.DataFrame(rows)
    # a fold with far fewer usable bins is not comparable
    t['comparable'] = (np.minimum(t.bins_ours, t.bins_lcdb)
                       / np.maximum(t.bins_ours, t.bins_lcdb)) > 0.6
    t['winner'] = np.where(t.gain > 0.02, 'ours',
                           np.where(t.gain < -0.02, 'lcdb', 'tie'))
    t.loc[t.cls == 'HARM', 'winner'] = 'undecidable (P vs 2P)'
    t.to_csv('comparison_data/pdm_compare.csv', index=False)
    print(f'\nscored {len(t):,} objects; {int(t.comparable.sum()):,} with comparable bin coverage\n')

    c = t[t.comparable & (t.cls != 'HARM')]
    print('PDM verdict (excluding HARM, where the test cannot decide):')
    print(c.winner.value_counts().to_string())
    print(f'\n  ours tighter : {(c.winner=="ours").mean():.1%}')
    print(f'  LCDB tighter : {(c.winner=="lcdb").mean():.1%}')
    print('\nby class:')
    print(c.groupby('cls').winner.value_counts().unstack(fill_value=0).to_string())
    print('\nby our data quality (aov_F):')
    for lo, hi in [(0, 3), (3, 10), (10, 30), (30, 100), (100, 1e9)]:
        s = c[(c.aov_F >= lo) & (c.aov_F < hi)]
        if len(s) > 20:
            print(f'  aov_F {lo:>4}-{hi if hi<1e9 else "inf":<4} n={len(s):4d}  '
                  f'ours {(s.winner=="ours").mean():5.1%}   lcdb {(s.winner=="lcdb").mean():5.1%}')
    print('\nIS THE LCDB PERIOD DETECTABLE IN OUR DATA AT ALL?')
    print('  (theta below the 5th percentile of random periods = detected; this is the')
    print('   non-circular test, since it asks nothing about our own period)')
    print(f'  LCDB period detected : {c.lcdb_detected.mean():6.1%}')
    print(f'  our  period detected : {c.ours_detected.mean():6.1%}')
    both = c[c.lcdb_detected & c.ours_detected]
    print(f'  BOTH detected        : {len(both):,}  -> real ambiguity, needs an eye')
    neither = c[~c.lcdb_detected & ~c.ours_detected]
    print(f'  NEITHER detected     : {len(neither):,}  -> lightcurve carries no usable period')
    only_ours = c[c.ours_detected & ~c.lcdb_detected]
    print(f'  ONLY ours detected   : {len(only_ours):,}  -> strongest LCDB-error candidates')
    only_lcdb = c[~c.ours_detected & c.lcdb_detected]
    print(f'  ONLY LCDB detected   : {len(only_lcdb):,}  -> strongest OUR-error candidates')
    print('\nwrote comparison_data/pdm_compare.csv')
