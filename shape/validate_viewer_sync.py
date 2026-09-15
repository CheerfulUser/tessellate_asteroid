"""Verify the viewer's spin sign and camera snap against the DATA.

Renders the mesh at each rotational phase using the real TESS observer/Sun directions, and
correlates the synthetic brightness with the observed folded lightcurve. A mirrored spin is
invisible by inspection but shows up immediately here -- it did once: a sign 'fix' reasoned
from the transform gave r=-0.69, the original gave r=+0.99.

Usage: python validate_viewer_sync.py [target]   (exits non-zero if the sync is wrong)
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from real_shape_tess import TARGETS, prepare, build_geometry, areas_normals


def read_shape(path):
    n, a = areas_normals(path)
    return np.array(n), np.array(a)

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)   # results/shape files sit beside this script


def main(argv_tag=None):
    tag = argv_tag or 'eurydike'
    C = TARGETS[tag]; res = json.load(open(f'{tag}_results.json'))
    bi = int(np.argmin([abs(s['lambda_deg'] - res['representative_lambda_deg'])
                        + abs(s['beta_deg'] - res['representative_beta_deg'])
                        for s in res['all_solutions']]))
    pl = open(f'{tag}_params_{bi}.txt').read().split()
    lam, bet, per_hr, t0, phi0 = (float(pl[0]), float(pl[1]), float(pl[2]), float(pl[3]), float(pl[4]))
    nor, ar = read_shape(f'{tag}_shape_{bi}.txt'); nor = np.array(nor); ar = np.array(ar)

    df = prepare(pd.read_csv(C['lc'])); sv, ev = build_geometry(df)
    en = ev / np.linalg.norm(ev, axis=1, keepdims=True)
    sn = sv / np.linalg.norm(sv, axis=1, keepdims=True)
    l, b = np.radians(lam), np.radians(bet)
    Rz = np.array([[np.cos(l), np.sin(l), 0], [-np.sin(l), np.cos(l), 0], [0, 0, 1]])
    a = np.pi / 2 - b
    Ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    Rpf = Ry @ Rz
    epf = (Rpf @ en.T).T.mean(0); epf /= np.linalg.norm(epf)
    spf = (Rpf @ sn.T).T.mean(0); spf /= np.linalg.norm(spf)

    ph = np.linspace(0, 1, 200, endpoint=False)
    def curve(sign):
        out = []
        for p in ph:
            t = sign * 2 * np.pi * p; c, s_ = np.cos(t), np.sin(t)
            n = (np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ nor.T).T
            mu = n @ epf; mu0 = n @ spf; v = (mu > 0) & (mu0 > 0)
            out.append(np.sum(ar[v] * mu[v] * mu0[v] * (1 / (mu[v] + mu0[v]) + 0.1)))
        out = np.array(out); return out / out.mean()

    jd = df['mjd'].values + 2400000.5
    op = ((phi0 + 2 * np.pi * (jd - t0) / (per_hr / 24.0)) / (2 * np.pi)) % 1.0
    fl = df['rel_flux'].values
    NB = 72; bb = np.clip((op * NB).astype(int), 0, NB - 1)
    cn = np.bincount(bb, minlength=NB); sm = np.bincount(bb, weights=fl, minlength=NB)
    ok = cn >= 3; obp = ((np.arange(NB) + .5) / NB)[ok]; obf = sm[ok] / cn[ok]; obf /= obf.mean()

    r_pos = np.corrcoef(np.interp(obp, ph, curve(+1)), obf)[0, 1]
    r_neg = np.corrcoef(np.interp(obp, ph, curve(-1)), obf)[0, 1]
    print(f'{C["name"]}  synthetic vs observed folded curve:')
    print(f'  viewer convention  Rz(+2pi p):  r = {r_pos:+.4f}   <- what the viewer uses')
    print(f'  reversed           Rz(-2pi p):  r = {r_neg:+.4f}')
    good = r_pos > 0.5 and r_pos > r_neg
    print('  -> ' + ('SYNC OK' if good else 'SYNC WRONG'))
    return good


if __name__ == '__main__':
    raise SystemExit(0 if main(sys.argv[1] if len(sys.argv) > 1 else None) else 1)
