"""Replace clearly-wrong LCDB periods with the TESS value in period_updated.

"Clearly wrong" is defined by the non-circular PDM null test, not by disagreement alone:

    the LCDB period folds our lightcurve NO BETTER THAN A RANDOM PERIOD
    AND our period folds it far better
    AND our data is strong enough for that to mean something

The first condition is the important one. It does not ask whether our period beats theirs --
ours was fitted to this data, so it would win by construction. It asks whether the published
period leaves ANY trace in 5,000-odd points of TESS photometry. (3550) Link is the worked case:
LCDB 12.371 h gives theta 1.004 against a random-period null of 1.002, and its half and double
give 1.005 and 0.996. All three sit inside the null. Ours gives 0.291.

Only rows already flagged LCDB_native_TESS_disagrees are eligible -- anything matching a 1x or
2x harmonic has already been handled by taking their cycle count with our precision. The native
Period column stays untouched throughout.
"""
import os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdm_compare import theta, sn  # noqa: E402

SRC = 'lcdb_lcs/stacked'
FLOOR_N, FLOOR_A, FLOOR_C = 100, 1.5, 1
N_NULL = 120


def one(row):
    p = f"{SRC}/{sn(row['designation'])}_lc.csv"
    if not os.path.exists(p):
        return None
    try:
        d = pd.read_csv(p).dropna(subset=['mjd', 'rel_flux'])
        if len(d) < 60:
            return None
        t, f = d.mjd.values, d.rel_flux.values
        lo, hi = np.percentile(f, [0.5, 99.5])
        k = (f >= lo) & (f <= hi)
        t, f = t[k], f[k]
        nb = int(np.clip(len(t) // 12, 10, 40))
        th_o, _ = theta(t, f, float(row['tess_period_hr']), nb)
        th_l, _ = theta(t, f, float(row['Period']), nb)
        if not (np.isfinite(th_o) and np.isfinite(th_l)):
            return None
        # also test the LCDB harmonics -- if any of them folds the data, the literature value is
        # a cycle-count issue rather than simply wrong, and must not be overwritten
        th_lh = []
        for mlt in (0.5, 2.0):
            th_x, _ = theta(t, f, float(row['Period']) * mlt, nb)
            if np.isfinite(th_x):
                th_lh.append(th_x)
        rng = np.random.RandomState(abs(hash(str(row['designation']))) % (2 ** 31))
        hi_p = min(240.0, (t.max() - t.min()) * 24 / 2)
        nulls = []
        for _ in range(N_NULL):
            Pr = float(np.exp(rng.uniform(np.log(1.0), np.log(max(hi_p, 2.0)))))
            th_r, _ = theta(t, f, Pr, nb)
            if np.isfinite(th_r):
                nulls.append(th_r)
        if len(nulls) < 40:
            return None
        p5 = float(np.percentile(nulls, 5))
        return dict(Number=row['Number'], designation=row['designation'],
                    theta_ours=th_o, theta_lcdb=th_l,
                    theta_lcdb_harm=min(th_lh) if th_lh else np.nan,
                    null_p5=p5,
                    ours_det=bool(th_o < p5),
                    lcdb_det=bool(th_l < p5),
                    lcdb_harm_det=bool(min(th_lh) < p5) if th_lh else False)
    except Exception:
        return None


if __name__ == '__main__':
    m = pd.read_csv('comparison_data/lcdb_updated.csv', low_memory=False)
    cand = m[(m.period_source == 'LCDB_native_TESS_disagrees') & m.designation.notna()].copy()
    print(f'{len(cand):,} candidate rows (TESS disagrees, not a harmonic)', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(one, r) for _, r in cand.iterrows()]
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            if r:
                rows.append(r)
            if (i + 1) % 400 == 0:
                print(f'  {i+1:,}/{len(cand):,}', flush=True)
    t = pd.DataFrame(rows)
    print(f'\ntested {len(t):,}')

    m = m.merge(t, on=['Number', 'designation'], how='left')
    floors = ((m.n_points >= FLOOR_N) & (m.aov_F >= FLOOR_A)
              & (m.n_phase_coverings >= FLOOR_C))
    wrong = (m.period_source == 'LCDB_native_TESS_disagrees') & \
            (m.ours_det == True) & (m.lcdb_det == False) & \
            (m.lcdb_harm_det == False) & floors        # noqa: E712
    print(f'\nclearly wrong LCDB periods: {int(wrong.sum()):,}')
    print(f'  (ours detected, LCDB and BOTH its harmonics undetected, our data passes floors)')
    m.loc[wrong, 'period_updated'] = m.loc[wrong, 'tess_period_hr']
    m.loc[wrong, 'period_source'] = 'TESS_replaces_wrong_LCDB'

    print('\nfinal period_updated provenance:')
    vc = m.period_source.value_counts()
    for k, v in vc.items():
        print(f'  {k:32s} {v:7,d} ({v/len(m):5.1%})')
    w = m[wrong]
    if len(w):
        print(f'\nreplaced entries, by LCDB quality code U:')
        print(w.U_clean.value_counts().sort_index().to_string())
        print(f'  median |shift| from the published value: '
              f'{((w.period_updated/w.Period-1).abs()).median():.1%}')
        print(f'  median TESS aov_F {w.aov_F.median():.0f}, n_points {w.n_points.median():.0f}')
    m.to_csv('comparison_data/lcdb_updated.csv', index=False)
    print('\nupdated comparison_data/lcdb_updated.csv (native Period column untouched)')
