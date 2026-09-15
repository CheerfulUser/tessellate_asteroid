"""Convex shape inversion for a TESS asteroid lightcurve, via the DAMIT convexinv suite.

Generalised from real_shape_may.py / real_shape_deflotte.py. Usage:
    python real_shape_tess.py eurydike

Carries the two adaptations the multi-sector TESS lightcurves need:
  * one RELATIVE session per visit, because distance changes over a long baseline otherwise put
    a large brightness trend into the data that gets absorbed as a spurious shape;
  * cut-overlap deduplication, since adjacent cuts observe the same asteroid at the same epoch.

Adds a RESIDUAL-VS-PHASE analysis, which is the point of running this on (75) Eurydike.
convexinv fits a CONVEX hull by construction: a concavity -- a crater, a flat facet, a
contact-binary neck -- is not representable, and the model must miss it. So if the bump inside
Eurydike's flat minimum is a concavity, the residuals will not be flat; they will spike at that
rotational phase specifically. A convex model fitting the bump cleanly would instead point at
albedo variation or a convex shape feature. The residuals discriminate.
"""
import json, os, subprocess, sys
import numpy as np, pandas as pd
from astropy.coordinates import get_body_barycentric
from astropy.time import Time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'polyhedrec')); sys.path.insert(0, HERE)
from polyhedrec_fast import reconstruct_fast  # noqa: E402

CONVEXINV = os.path.join(HERE, 'DAMIT-convex/convexinv/convexinv')
Y34 = '/Users/rridden/Documents/work/code/tess/asteroid/y3_4'

TARGETS = {
    'eurydike': dict(name='(75) Eurydike', lc=f'{Y34}/lcdb_lcs/stacked/75_Eurydike_lc.csv',
                     period_hr=5.357929509432807, tag='eurydike'),
    'deflotte': dict(name='(1295) Deflotte', lc=f'{Y34}/lcdb_lcs/stacked/1295_Deflotte_lc.csv',
                     period_hr=26.4384, tag='deflotte'),
}
# Resolution of the convex model. NROWS sets the triangulation of the Gaussian image
# (facets ~ 8*NROWS^2 + 2), HARM the degree/order of the spherical-harmonic expansion of the
# support function. 4/2 gave only 33 facets; 6/6 gives a few hundred. Raising these on
# single-apparition data risks fitting noise, so the convexity regularisation matters more
# here -- check that the fit rms does not degrade and the shape stays convex-plausible.
HARM = int(os.environ.get('CI_HARM', 6))
NROWS = int(os.environ.get('CI_NROWS', 6))
MIN_PTS = 50
START_POLES = [(0, 0), (90, 0), (180, 45), (270, -45), (45, 60), (135, -30), (225, 20), (315, -60)]


def build_geometry(df):
    t = Time(df['mjd'].values, format='mjd', scale='utc')
    ra, dec = np.radians(df['ra'].values), np.radians(df['dec'].values)
    u = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    ep = get_body_barycentric('earth', t).xyz.to('au').value.T
    sp = get_body_barycentric('sun', t).xyz.to('au').value.T
    ap = ep + df['delta_au'].values[:, None] * u
    return sp - ap, ep - ap


def prepare(df):
    df = df.dropna(subset=['mjd', 'rel_flux', 'ra', 'dec', 'delta_au']).copy()
    n0 = len(df)
    df['_k'] = df['mjd'].round(6)
    df['_vn'] = df.groupby('visit')['mjd'].transform('size')
    df = (df.sort_values(['_k', '_vn'], ascending=[True, False]).drop_duplicates('_k'))
    df = df[df.groupby('visit')['mjd'].transform('size') >= MIN_PTS]
    df = df.sort_values('mjd').reset_index(drop=True)
    print(f'  {n0} -> {len(df)} rows after dedup/min-visit cut, {df.visit.nunique()} sessions')
    return df


def write_lcs(df, sv, ev, path):
    jd = df['mjd'].values + 2400000.5; fl = df['rel_flux'].values; vis = df['visit'].values
    order = list(pd.unique(vis))
    with open(path, 'w') as f:
        f.write(f'{len(order)}\n')
        for v in order:
            m = np.where(vis == v)[0]
            f.write(f'{len(m)} 1\n')
            for i in m:
                f.write(f'{jd[i]:.6f} {fl[i]:.6f} '
                        f'{sv[i,0]:.6f} {sv[i,1]:.6f} {sv[i,2]:.6f} '
                        f'{ev[i,0]:.6f} {ev[i,1]:.6f} {ev[i,2]:.6f}\n')
    return len(order)


def write_control(path, lam, bet, per):
    with open(path, 'w') as f:
        f.write(f'{lam}\t\t1\tinital lambda [deg] (0/1 - fixed/free)\n')
        f.write(f'{bet}\t\t1\tinitial beta [deg] (0/1 - fixed/free)\n')
        f.write(f'{per}\t\t1\tinital period [hours] (0/1 - fixed/free)\n')
        f.write('0\t\t\tzero time [JD]\n0\t\t\tinitial rotation angle [deg]\n')
        f.write('0.1\t\t\tconvexity regularization\n')
        f.write(f'{HARM} {HARM}\t\t\tdegree and order of spherical harmonics expansion\n')
        f.write(f'{NROWS}\t\t\tnumber of rows\n')
        for line in ["0.5\t\t0\tphase funct. param. 'a' (0/1 - fixed/free)",
                     "0.1\t\t0\tphase funct. param. 'd' (0/1 - fixed/free)",
                     "-0.5\t\t0\tphase funct. param. 'k' (0/1 - fixed/free)",
                     "0.1\t\t0\tLambert coefficient 'c' (0/1 - fixed/free)"]:
            f.write(line + '\n')
        f.write('50\t\t\titeration stop condition\n')


def run(cp, lcs, sp, pp, fp):
    with open(lcs) as fin:
        r = subprocess.run([CONVEXINV, '-o', sp, '-p', pp, cp, fp],
                           stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0 or not os.path.exists(pp):
        return None
    L = open(pp).read().splitlines()
    if not L:
        return None
    lam, bet, per = map(float, L[0].split())
    return lam, bet, per


def areas_normals(path):
    L = [l.strip() for l in open(path) if l.strip()]
    n = int(L[0]); a, nor, i = [], [], 1
    for _ in range(n):
        a.append(float(L[i])); nor.append(np.array(list(map(float, L[i + 1].split())))); i += 2
    return nor, a


def main(key):
    C = TARGETS[key]; tag = C['tag']
    df = prepare(pd.read_csv(C['lc']))
    P = C['period_hr'] / 24.0
    span = df.mjd.max() - df.mjd.min()
    print(f"{C['name']}: {len(df):,} obs, {span:.1f} d, {span/P:.0f} rotations, "
          f"phase angle {df.phase_angle_deg.min():.2f}-{df.phase_angle_deg.max():.2f} deg")

    sv, ev = build_geometry(df)
    lcs = os.path.join(HERE, f'{tag}_inversion_input_lcs.txt')
    nses = write_lcs(df, sv, ev, lcs)

    sols = []
    for i, (l0, b0) in enumerate(START_POLES):
        cp = os.path.join(HERE, f'{tag}_control_{i}.txt')
        spp = os.path.join(HERE, f'{tag}_shape_{i}.txt')
        pp = os.path.join(HERE, f'{tag}_params_{i}.txt')
        fp = os.path.join(HERE, f'{tag}_fit_{i}.txt')
        write_control(cp, l0, b0, C['period_hr'])
        res = run(cp, lcs, spp, pp, fp)
        if res is None:
            print(f'  start ({l0:+4.0f},{b0:+3.0f}): FAILED'); continue
        lam, bet, per = res
        print(f'  start ({l0:+4.0f},{b0:+3.0f}) -> lambda={lam:7.2f} beta={bet:+6.2f} P={per:.6f} h')
        sols.append(dict(start=(l0, b0), lam=lam, bet=bet, per=per, shape=spp, fit=fp))
    if not sols:
        print('nothing converged'); return

    lam = np.array([s['lam'] for s in sols]); bet = np.array([s['bet'] for s in sols])
    per = np.array([s['per'] for s in sols])
    lcs_std = np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * np.radians(lam)))))))
    print(f'\nspread: lambda circ-std {lcs_std:.1f} deg, beta std {bet.std():.1f} deg, '
          f'period std {per.std()*3600:.2f} s')
    conv = lcs_std < 20 and bet.std() < 20
    print('  -> ' + ('starts CONVERGE' if conv else
                     'starts SCATTER: pole undetermined (expected for one apparition)'))

    best = min(sols, key=lambda s: abs(s['lam'] - np.median(lam)) + abs(s['bet'] - np.median(bet)))

    # --- residual vs rotational phase: can a CONVEX model reproduce the feature?
    model = np.loadtxt(best['fit'])
    obs = df['rel_flux'].values
    resid = np.full(len(obs), np.nan)
    if len(model) == len(obs):
        resid = obs - model
        ph = (df['mjd'].values % P) / P
        nb = 60
        b = np.clip((ph * nb).astype(int), 0, nb - 1)
        cnt = np.bincount(b, minlength=nb)
        rm = np.bincount(b, weights=resid, minlength=nb) / np.maximum(cnt, 1)
        ok = cnt >= 3
        rms_all = np.sqrt(np.nanmean(resid ** 2))
        print(f'\nconvex-model residuals: rms {rms_all:.5f}, '
              f'binned excursion {np.nanmin(rm[ok]):+.5f} to {np.nanmax(rm[ok]):+.5f} '
              f'({np.nanmax(np.abs(rm[ok]))/rms_all:.1f}x rms)')
        np.savetxt(os.path.join(HERE, f'{tag}_residual_phase.txt'),
                   np.column_stack([(np.arange(nb) + .5)[ok] / nb, rm[ok], cnt[ok]]),
                   header='phase binned_residual n')
    else:
        print(f'\nfit file has {len(model)} rows vs {len(obs)} obs -- residuals skipped')

    nor, ar = areas_normals(best['shape'])
    # Minkowski reconstruction is sensitive to the starting scale D and simply fails for some
    # values -- at 289 facets the natural choice (equal-area sphere radius) does not converge
    # but 1.25x does. Retry across a spread rather than dying, since the inversion itself has
    # already succeeded by this point and only the mesh build is at risk.
    D0 = (sum(ar) / (4 * np.pi)) ** 0.5
    Pm = None
    for fac in (1.0, 1.25, 0.8, 1.6, 0.6, 2.5, 0.4):
        try:
            Pm = reconstruct_fast(nor, ar, D=D0 * fac, options={'rtol': 1e-4, 'atol': 1e-7})
            if fac != 1.0:
                print(f'  (mesh reconstruction needed D x {fac})')
            break
        except Exception:
            continue
    if Pm is None:
        print('  mesh reconstruction FAILED at every D -- inversion results still written')
        Pm = None
    v = np.array(Pm.vertices); v = v - v.mean(axis=0)
    fc = [list(f.vertices) for f in Pm.faces]
    ext = np.sort(v.max(axis=0) - v.min(axis=0))[::-1]
    print(f'representative: lambda={best["lam"]:.2f} beta={best["bet"]:+.2f} '
          f'P={best["per"]:.6f} h, {len(fc)} facets, a/b={ext[0]/ext[1]:.3f} b/c={ext[1]/ext[2]:.3f}')

    json.dump({'recovered': {'verts': v.tolist(), 'facets': fc}},
              open(os.path.join(HERE, f'{tag}_mesh_data.json'), 'w'))
    json.dump(dict(target=C['name'], n_observations=int(len(df)), n_sessions=int(nses),
                   adopted_period_hr=C['period_hr'], baseline_days=float(span),
                   phase_angle_range_deg=float(df.phase_angle_deg.max() - df.phase_angle_deg.min()),
                   all_solutions=[dict(start_lambda=s['start'][0], start_beta=s['start'][1],
                                       lambda_deg=s['lam'], beta_deg=s['bet'],
                                       period_hr=s['per']) for s in sols],
                   representative_lambda_deg=best['lam'], representative_beta_deg=best['bet'],
                   representative_period_hr=best['per'],
                   lambda_circ_std_deg=float(lcs_std), beta_std_deg=float(bet.std()),
                   starts_converged=bool(conv), bbox_a_over_b=float(ext[0] / ext[1]),
                   bbox_b_over_c=float(ext[1] / ext[2]), recovered_facets=len(fc)),
              open(os.path.join(HERE, f'{tag}_results.json'), 'w'), indent=2)
    print(f'wrote {tag}_results.json, {tag}_mesh_data.json')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'eurydike')
