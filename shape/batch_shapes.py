"""Batch shape inversion: one asteroid per invocation, designed for a SLURM array.

Every failure mode found while modelling three objects by hand is handled here rather than
discovered again 16,000 times:

  convexinv POINTS_MAX   A TESS visit runs continuously for days, so one session can exceed the
                         compile-time 2000-point limit and convexinv rejects the WHOLE object.
                         (3550) Link hit this at 2,061 points. constants.h is patched to 3000 by
                         setup_binaries.sh; this script also splits any session that still
                         exceeds the limit rather than losing the object.
  silent failures        convexinv writes its reason to stderr. The first version discarded it,
                         so eight poles reported a bare "FAILED" with no cause. stderr is
                         captured and recorded per object.
  mesh reconstruction    polyhedrec failed on Deflotte at every starting scale after 33 min of
                         CPU. DAMIT's own minkowski solved it in 0.79 s. minkowski is primary;
                         polyhedrec is the fallback, under a timeout so it cannot stall a task.
  per-session scaling    convexinv fits relative photometry with a free scale per session, so
                         its model output is not on a common scale (Link: 0.886-1.233 across
                         seven sessions). Rescale each session to its own data before folding.
  polygonal faces        Faces are 4-9 sided, not triangles. Encode per-face vertex counts.

Output per object is a single JSON with the solution, the mesh, the folded curve and a set of
validation flags, so the site build consumes it directly with no post-processing.

Usage:
    python batch_shapes.py --index 0 --total 200 --targets targets.csv --out shapes/
    python batch_shapes.py --designation "(75) Eurydike" --out shapes/
"""
import argparse, json, os, re, subprocess, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONVEXINV = os.path.join(HERE, 'DAMIT-convex/convexinv/convexinv')
MINKOWSKI = os.path.join(HERE, 'DAMIT-convex/minkowski')
POINTS_MAX = int(os.environ.get('CI_POINTS_MAX', 3000))
HARM = int(os.environ.get('CI_HARM', 6))
NROWS = int(os.environ.get('CI_NROWS', 6))
MIN_PTS = 50
NB = 72
# Pole FIXED, not fitted. Single-apparition data cannot determine it -- all eight starting
# orientations fit to within 1-10% in rms -- and fitting it anyway lets the solver trade pole
# against shape, dumping unconstrained area into a facet on the spin axis (Link: 18-34% of its
# surface in one face, falling to 2.3% once fixed). One start is therefore sufficient.
FIX_POLE = os.environ.get('CI_FIX_POLE', '1') == '1'
_ALL_POLES = [(0, 0), (90, 0), (180, 45), (270, -45), (45, 60), (135, -30), (225, 20), (315, -60)]
START_POLES = ([(0.0, 0.0)] if FIX_POLE else _ALL_POLES)
T_INV = int(os.environ.get('T_INV', 900))
T_MINK = int(os.environ.get('T_MINK', 300))


def sn(d):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(d)).strip('_')


def prepare(df):
    """Deduplicate cut overlap, drop under-populated visits, sort."""
    df = df.dropna(subset=['mjd', 'rel_flux', 'ra', 'dec', 'delta_au']).copy()
    df['_k'] = df['mjd'].round(6)
    df['_vn'] = df.groupby('visit')['mjd'].transform('size')
    df = df.sort_values(['_k', '_vn'], ascending=[True, False]).drop_duplicates('_k')
    df = df[df.groupby('visit')['mjd'].transform('size') >= MIN_PTS]
    return df.sort_values('mjd').reset_index(drop=True)


def build_geometry(df):
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    t = Time(df['mjd'].values, format='mjd', scale='utc')
    ra, dec = np.radians(df['ra'].values), np.radians(df['dec'].values)
    u = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    ep = get_body_barycentric('earth', t).xyz.to('au').value.T
    sp = get_body_barycentric('sun', t).xyz.to('au').value.T
    ap = ep + df['delta_au'].values[:, None] * u
    return sp - ap, ep - ap


def session_blocks(df):
    """Visit ids, splitting any visit longer than POINTS_MAX into contiguous chunks.

    Losing a whole object because one visit is 61 points over a compile-time limit is not an
    acceptable failure mode at catalogue scale. Splitting costs only an extra free scale
    parameter per chunk, which is cheap next to dropping the object."""
    out = []
    for v in pd.unique(df['visit'].values):
        idx = np.where(df['visit'].values == v)[0]
        if len(idx) <= POINTS_MAX:
            out.append(idx)
        else:
            n = int(np.ceil(len(idx) / POINTS_MAX))
            out.extend(np.array_split(idx, n))
    return out


def write_lcs(df, sv, ev, blocks, path):
    jd = df['mjd'].values + 2400000.5
    fl = df['rel_flux'].values
    with open(path, 'w') as f:
        f.write(f'{len(blocks)}\n')
        for idx in blocks:
            f.write(f'{len(idx)} 1\n')
            for i in idx:
                f.write(f'{jd[i]:.6f} {fl[i]:.6f} '
                        f'{sv[i,0]:.6f} {sv[i,1]:.6f} {sv[i,2]:.6f} '
                        f'{ev[i,0]:.6f} {ev[i,1]:.6f} {ev[i,2]:.6f}\n')


def write_control(path, lam, bet, per):
    with open(path, 'w') as f:
        _fl = '0' if FIX_POLE else '1'
        f.write(f'{lam}\t\t{_fl}\tinital lambda\n{bet}\t\t{_fl}\tinitial beta\n'
                f'{per}\t\t1\tinital period\n')
        f.write('0\t\t\tzero time\n0\t\t\tinitial rotation angle\n0.1\t\t\tconvexity regularization\n')
        f.write(f'{HARM} {HARM}\t\t\tdegree and order\n{NROWS}\t\t\tnumber of rows\n')
        for l in ["0.5\t\t0\ta", "0.1\t\t0\td", "-0.5\t\t0\tk", "0.1\t\t0\tc"]:
            f.write(l + '\n')
        f.write('50\t\t\titeration stop condition\n')


def run_pole(cp, lcs, sp, pp, fp):
    """Returns (lambda, beta, period) or (None, reason) -- stderr is KEPT, not discarded."""
    try:
        with open(lcs) as fin:
            r = subprocess.run([CONVEXINV, '-o', sp, '-p', pp, cp, fp], stdin=fin,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=T_INV)
    except subprocess.TimeoutExpired:
        return None, f'timeout after {T_INV}s'
    if r.returncode != 0:
        return None, (r.stderr.decode(errors='replace').strip().splitlines() or ['nonzero exit'])[-1]
    if not os.path.exists(pp):
        return None, 'no params file written'
    L = open(pp).read().splitlines()
    if not L:
        return None, 'empty params file'
    return tuple(map(float, L[0].split()))[:3], None


def minkowski(shape_path):
    if not os.path.exists(MINKOWSKI):
        return None
    try:
        with open(shape_path) as fin:
            r = subprocess.run([MINKOWSKI], stdin=fin, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, timeout=T_MINK)
        L = [l.split() for l in r.stdout.decode().splitlines() if l.strip()]
        nv, nf = int(L[0][0]), int(L[0][1])
        V = np.array([[float(x) for x in L[1 + i]] for i in range(nv)])
        i, F = 1 + nv, []
        while len(F) < nf:
            F.append([int(x) - 1 for x in L[i + 1]]); i += 2
        return V - V.mean(axis=0), F
    except Exception:
        return None


def check_convex(V, F):
    """Every vertex on or inside every face plane. A mesh failing this must not be published."""
    bad = 0
    for f in F:
        p = V[f]
        n = np.zeros(3)
        for k in range(len(f)):
            n += np.cross(p[k], p[(k + 1) % len(f)])
        nl = np.linalg.norm(n)
        if nl == 0:
            bad += 1; continue
        n /= nl
        if (V @ n - np.dot(n, p.mean(0)) > 1e-6).any():
            bad += 1
    return bad


def process(des, period_hr, lc_path, workdir, outdir):
    t0 = time.time()
    tag = sn(des)
    rec = dict(designation=des, key=tag, adopted_period_hr=float(period_hr), ok=False)
    try:
        df = prepare(pd.read_csv(lc_path))
        if len(df) < 200:
            rec['error'] = f'only {len(df)} usable points'; return rec
        sv, ev = build_geometry(df)
        blocks = session_blocks(df)
        rec.update(n_obs=int(len(df)), n_visits=int(df['visit'].nunique()),
                   n_sessions=len(blocks), n_sectors=int(df['sector'].nunique()),
                   baseline_days=float(df.mjd.max() - df.mjd.min()),
                   phase_angle_range_deg=float(df.phase_angle_deg.max() - df.phase_angle_deg.min()))
        lcs = f'{workdir}/{tag}_lcs.txt'
        write_lcs(df, sv, ev, blocks, lcs)

        sols, fails = [], []
        for i, (l0, b0) in enumerate(START_POLES):
            cp, spp = f'{workdir}/{tag}_c{i}.txt', f'{workdir}/{tag}_s{i}.txt'
            pp, fp = f'{workdir}/{tag}_p{i}.txt', f'{workdir}/{tag}_f{i}.txt'
            write_control(cp, l0, b0, period_hr)
            res, why = run_pole(cp, lcs, spp, pp, fp)
            if res is None:
                fails.append(f'({l0},{b0}): {why}'); continue
            sols.append(dict(start=[l0, b0], lam=res[0], bet=res[1], per=res[2],
                             shape=spp, fit=fp, params=pp))
        rec['n_poles_converged'] = len(sols)
        rec['pole_failures'] = fails[:3]
        if not sols:
            rec['error'] = 'no starting pole converged'; return rec

        lam = np.array([s['lam'] for s in sols]); bet = np.array([s['bet'] for s in sols])
        per = np.array([s['per'] for s in sols])
        lam_std = float(np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * np.radians(lam))))))))
        best = min(sols, key=lambda s: abs(s['lam'] - np.median(lam)) + abs(s['bet'] - np.median(bet)))
        rec.update(lambda_circ_std_deg=lam_std, beta_std_deg=float(bet.std()),
                   period_std_s=float(per.std() * 3600),
                   representative_lambda_deg=best['lam'], representative_beta_deg=best['bet'],
                   model_period_hr=best['per'], pole_constrained=bool(lam_std < 20))

        mk = minkowski(best['shape'])
        if mk is None:
            rec['error'] = 'mesh reconstruction failed'; return rec
        V, F = mk
        nbad = check_convex(V, F)
        rec.update(n_verts=len(V), n_facets=len(F), nonconvex_faces=int(nbad))
        if nbad:
            rec['error'] = f'{nbad} non-convex faces'; return rec
        ext = np.sort(V.max(0) - V.min(0))[::-1]
        rec.update(bbox_a_over_b=float(ext[0] / ext[1]), bbox_b_over_c=float(ext[1] / ext[2]))

        # folded curve + per-session-rescaled model on one grid
        pl = open(best['params']).read().split()
        t0j, phi0 = float(pl[3]), float(pl[4])
        jd = df['mjd'].values + 2400000.5
        ph = ((phi0 + 2 * np.pi * (jd - t0j) / (best['per'] / 24.0)) / (2 * np.pi)) % 1.0
        fl = df['rel_flux'].values
        b = np.clip((ph * NB).astype(int), 0, NB - 1)
        cnt = np.bincount(b, minlength=NB)
        mean = np.bincount(b, weights=fl, minlength=NB) / np.maximum(cnt, 1)
        var = np.bincount(b, weights=(fl - mean[b]) ** 2, minlength=NB)
        with np.errstate(divide='ignore', invalid='ignore'):
            sem = np.sqrt(var / np.maximum(cnt - 1, 1)) / np.sqrt(np.maximum(cnt, 1))
        ok = cnt >= 3
        mod = np.loadtxt(best['fit']).astype(float)
        mrow = None
        if len(mod) == len(df):
            for idx in blocks:                       # rescale EACH session to its own data
                mm = mod[idx].mean()
                if np.isfinite(mm) and mm != 0:
                    mod[idx] *= fl[idx].mean() / mm
            ms = np.bincount(b, weights=mod, minlength=NB) / np.maximum(cnt, 1)
            mrow = [round(float(x), 6) for x in ms[ok]]
        dat = mean[ok]
        rec['amplitude_mag'] = float(-2.5 * np.log10(dat.min() / dat.max())) if dat.min() > 0 else None
        if mrow is not None:
            rec['model_rms'] = float(np.sqrt(np.mean((np.array(mrow) - dat) ** 2)))

        out = dict(solution={k: v for k, v in rec.items() if k not in ('ok',)},
                   mesh={'v': [round(float(c), 4) for p in V for c in p],
                         'f': [int(i) for t in F for i in t], 'fn': [len(t) for t in F]},
                   lightcurve=dict(phase=[round(float(x), 5) for x in ((np.arange(NB) + .5) / NB)[ok]],
                                   flux=[round(float(x), 5) for x in dat],
                                   err=[round(float(x), 6) for x in np.nan_to_num(sem[ok])],
                                   model_flux=mrow))
        os.makedirs(outdir, exist_ok=True)
        json.dump(out, open(f'{outdir}/{tag}.json', 'w'), separators=(',', ':'))
        rec['ok'] = True
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
    finally:
        rec['seconds'] = round(time.time() - t0, 1)
        for f in os.listdir(workdir):
            if f.startswith(tag + '_'):
                try: os.remove(os.path.join(workdir, f))
                except OSError: pass
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets'); ap.add_argument('--designation')
    ap.add_argument('--index', type=int, default=0); ap.add_argument('--total', type=int, default=1)
    ap.add_argument('--lcdir', default='lcdb_lcs/stacked')
    ap.add_argument('--out', default='shapes'); ap.add_argument('--work', default='_work')
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    t = pd.read_csv(a.targets, low_memory=False)
    if a.designation:
        t = t[t.designation == a.designation]
    else:
        t = t.iloc[a.index::a.total]
    print(f'task {a.index}/{a.total}: {len(t)} objects', flush=True)
    recs = []
    for _, r in t.iterrows():
        rec = process(r['designation'], r['period_hr'],
                      f"{a.lcdir}/{sn(r['designation'])}_lc.csv", a.work, a.out)
        recs.append(rec)
        print(f"  {rec['key']:26s} {'ok' if rec['ok'] else 'FAIL'} {rec['seconds']:6.1f}s "
              f"{rec.get('error','')}", flush=True)
    pd.DataFrame(recs).to_csv(f'{a.out}/_status_{a.index:04d}.csv', index=False)
    n = sum(r['ok'] for r in recs)
    print(f'\n{n}/{len(recs)} succeeded', flush=True)
