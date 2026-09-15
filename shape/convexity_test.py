"""Does stronger convexity regularisation reduce the flat polar caps?

Link and Wuyeesun each put 18-20% of their total surface area into a single facet sitting almost
exactly on the spin axis (latitudes -84.8 and -88.8 deg). Those facets face TESS directly -- mu
= +0.52 and +0.44 -- so they are the most consistently viewed part of the body, yet the
lightcurve barely constrains them: a facet on the rotation axis holds a near-constant projected
area through a rotation, and relative photometry with a free scale per session absorbs any
constant contribution entirely.

convexinv's control file carries a convexity regularisation weight, currently 0.1. This scans it
and records, for each value, how concentrated the facet areas become and whether the fit
degrades. What we want is the smallest weight that breaks up the cap without making the model
fit the data worse -- if no such value exists, the caps are intrinsic to what the data can say
and the honest response is to flag them rather than regularise them away.

Usage: python convexity_test.py link wuyeesun
"""
import json, os, re, subprocess, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_shape_tess import (TARGETS, prepare, build_geometry, write_lcs, areas_normals,  # noqa
                             CONVEXINV, HARM, NROWS, MIN_PTS)

WEIGHTS = [0.1, 0.3, 1.0, 3.0, 10.0]


def control(path, lam, bet, per, conv_w):
    with open(path, 'w') as f:
        f.write(f'{lam}\t\t1\tinital lambda\n{bet}\t\t1\tinitial beta\n{per}\t\t1\tinital period\n')
        f.write('0\t\t\tzero time\n0\t\t\tinitial rotation angle\n')
        f.write(f'{conv_w}\t\t\tconvexity regularization\n')
        f.write(f'{HARM} {HARM}\t\t\tdegree and order\n{NROWS}\t\t\tnumber of rows\n')
        for l in ["0.5\t\t0\ta", "0.1\t\t0\td", "-0.5\t\t0\tk", "0.1\t\t0\tc"]:
            f.write(l + '\n')
        f.write('50\t\t\titeration stop condition\n')


def run(tag):
    C = TARGETS[tag]
    res = json.load(open(f'{HERE}/{tag}_results.json'))
    bi = int(np.argmin([abs(s['lambda_deg'] - res['representative_lambda_deg'])
                        + abs(s['beta_deg'] - res['representative_beta_deg'])
                        for s in res['all_solutions']]))
    pl = open(f'{HERE}/{tag}_params_{bi}.txt').read().split()
    lam, bet, per = float(pl[0]), float(pl[1]), float(pl[2])

    df = prepare(pd.read_csv(C['lc']))
    sv, ev = build_geometry(df)
    lcs = f'{HERE}/_cv_{tag}_lcs.txt'
    write_lcs(df, sv, ev, lcs)
    obs = df['rel_flux'].values

    print(f'\n{C["name"]}  (starting from the adopted pole {lam:.1f}, {bet:+.1f})')
    print(f'{"conv_w":>7} {"largest":>8} {"top5":>7} {"polar<30deg":>12} {"fit rms":>9} {"a/b":>6} {"b/c":>6}')
    rows = []
    for w in WEIGHTS:
        cp = f'{HERE}/_cv_{tag}_c.txt'
        sp = f'{HERE}/_cv_{tag}_s.txt'
        pp = f'{HERE}/_cv_{tag}_p.txt'
        fp = f'{HERE}/_cv_{tag}_f.txt'
        control(cp, lam, bet, per, w)
        with open(lcs) as fin:
            r = subprocess.run([CONVEXINV, '-o', sp, '-p', pp, cp, fp], stdin=fin,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1200)
        if r.returncode != 0 or not os.path.exists(sp):
            print(f'{w:7.1f}   FAILED  {r.stderr.decode(errors="replace").strip()[:50]}')
            continue
        n, a = areas_normals(sp)
        n = np.array(n); a = np.array(a); af = a / a.sum()
        top = np.sort(af)[::-1]
        polar = np.abs(n[:, 2]) > np.cos(np.radians(60))
        mod = np.loadtxt(fp).astype(float)
        # per-session rescale before comparing, as the viewer does
        if len(mod) == len(df):
            for v in pd.unique(df['visit'].values):
                m = df['visit'].values == v
                mm = mod[m].mean()
                if np.isfinite(mm) and mm != 0:
                    mod[m] *= obs[m].mean() / mm
            rms = float(np.sqrt(np.mean((mod - obs) ** 2)))
        else:
            rms = float('nan')
        # axis ratios straight from the Gaussian image extent
        ext = np.sort(np.ptp(n * a[:, None], axis=0))[::-1]
        rows.append(dict(w=w, largest=top[0], top5=top[:5].sum(), polar=af[polar].sum(), rms=rms))
        print(f'{w:7.1f} {top[0]*100:7.1f}% {top[:5].sum()*100:6.1f}% {af[polar].sum()*100:11.1f}% '
              f'{rms:9.4f} {ext[0]/ext[1]:6.2f} {ext[1]/ext[2]:6.2f}')
    for f in os.listdir(HERE):
        if f.startswith(f'_cv_{tag}_'):
            os.remove(os.path.join(HERE, f))
    return rows


if __name__ == '__main__':
    for tag in (sys.argv[1:] or ['link']):
        run(tag)
