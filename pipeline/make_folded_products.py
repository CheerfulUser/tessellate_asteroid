"""Build the compact phase-folded lightcurve product that the public pages will plot.

One binned curve per asteroid: phase, mean relative flux, standard error of the mean, and the
point count per bin. The SEM is what makes a binned curve honest -- a bare binned line hides how
well each point is determined, and for these objects the per-bin population varies by an order
of magnitude across the fold.

Size is the reason for binning at all. GitHub Pages enforces a hard 1 GB limit on the PUBLISHED
site (the 1 GB/5 GB repository figures are only recommendations and are not enforced -- the
user's own repos run to 1.5 GB). Full stacked lightcurves average 222 KB, so 16,000 of them is
3.6 GB and cannot be published; a binned curve is ~2 KB, so the whole set is ~30 MB and the site
stays well clear. Full-resolution lightcurves remain in the repo as downloads, which the Pages
limit does not apply to.

Bin count adapts to the point count: too many bins on a sparse curve gives empty bins and
meaningless error bars, too few on a dense one throws away real structure.
"""
import json, os, re, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC = 'lcdb_lcs/stacked'
OUT = 'folded_products'
MIN_PER_BIN = 3


def sn(d):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(d)).strip('_')


def fold_bin(t, f, P_hr, nb=None):
    P = P_hr / 24.0
    if not np.isfinite(P) or P <= 0 or len(t) < 20:
        return None
    ph = (t % P) / P
    if nb is None:
        nb = int(np.clip(len(t) // 15, 15, 60))
    idx = np.clip((ph * nb).astype(int), 0, nb - 1)
    cen, mean, sem, cnt = [], [], [], []
    for k in range(nb):
        m = idx == k
        n = int(m.sum())
        if n < MIN_PER_BIN:
            continue
        v = f[m]
        cen.append((k + 0.5) / nb)
        mean.append(float(v.mean()))
        sem.append(float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
        cnt.append(n)
    if len(cen) < 8:
        return None
    return dict(phase=cen, flux=mean, err=sem, n=cnt, n_bins=nb)


def one(row):
    des = row['designation']
    p = f'{SRC}/{sn(des)}_lc.csv'
    if not os.path.exists(p):
        return None
    try:
        d = pd.read_csv(p).dropna(subset=['mjd', 'rel_flux'])
        t, f = d['mjd'].values, d['rel_flux'].values
        lo, hi = np.percentile(f, [0.5, 99.5])
        k = (f >= lo) & (f <= hi)
        t, f = t[k], f[k]
        fb = fold_bin(t, f, float(row['period_hr']))
        if fb is None:
            return None
        amp = max(fb['flux']) - min(fb['flux'])
        out = dict(designation=des, period_hr=round(float(row['period_hr']), 6),
                   n_points=int(len(t)), n_visits=int(d['visit'].nunique()),
                   n_sectors=int(d['sector'].nunique()),
                   amplitude=round(float(amp), 5),
                   amplitude_mag=round(float(-2.5 * np.log10(min(fb['flux']) / max(fb['flux']))), 4),
                   phase=[round(x, 5) for x in fb['phase']],
                   flux=[round(x, 5) for x in fb['flux']],
                   err=[round(x, 6) for x in fb['err']],
                   n=fb['n'])
        os.makedirs(OUT, exist_ok=True)
        with open(f'{OUT}/{sn(des)}.json', 'w') as fh:
            json.dump(out, fh, separators=(',', ':'))
        return (des, os.path.getsize(f'{OUT}/{sn(des)}.json'))
    except Exception:
        return None


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'comparison_data/lcdb_overlap_render.csv'
    m = pd.read_csv(src, low_memory=False)
    m = m[m.period_hr.notna()]
    if len(sys.argv) > 2:
        m = m.head(int(sys.argv[2]))
    print(f'{len(m):,} objects', flush=True)
    sizes = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, fu in enumerate(as_completed([pool.submit(one, r) for _, r in m.iterrows()])):
            r = fu.result()
            if r:
                sizes.append(r[1])
            if (i + 1) % 1000 == 0:
                print(f'  {i+1:,}/{len(m):,}', flush=True)
    s = np.array(sizes)
    print(f'\nwrote {len(s):,} folded products to {OUT}/')
    print(f'  mean {s.mean()/1024:.1f} KB, median {np.median(s)/1024:.1f} KB, '
          f'max {s.max()/1024:.1f} KB')
    print(f'  total {s.sum()/1e6:.1f} MB   -> extrapolated to 16,000 objects: '
          f'{s.mean()*16000/1e6:.0f} MB')
