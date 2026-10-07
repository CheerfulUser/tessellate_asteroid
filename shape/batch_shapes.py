"""Batch shape inversion: one asteroid per invocation, designed for a SLURM array.

Every failure mode found while modelling three objects by hand is handled here rather than
discovered again 16,000 times:

  convexinv POINTS_MAX   A TESS visit runs continuously for days, so one session can exceed the
                         compile-time 2000-point limit and convexinv rejects the WHOLE object.
                         (3550) Link hit this at 2,061 points. The fork CheerfulUser/DAMIT-convex
                         allocates its arrays at run time (no limit); this script still splits
                         any session longer than POINTS_MAX, which costs only a free scale each.
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
  ecliptic frame         convexinv's docs require ECLIPTIC astrocentric Sun/Earth vectors. The
                         first version passed ICRS equatorial ones while giving the fixed pole in
                         ecliptic lon/lat, so the "equator-on" pole was up to the 23.4 deg
                         obliquity away from perpendicular. Vectors and pole now share one frame.
  relative flag          lcs code 1 means CALIBRATED to convexinv (Inrel=0: no per-lightcurve
                         renormalisation, size scale free). These are per-visit relative fluxes,
                         so the code is 0.

Output per object is a single JSON with the solution, the mesh, the folded curve and a set of
validation flags, so the site build consumes it directly with no post-processing.

Usage:
    python batch_shapes.py --index 0 --total 200 --targets targets.csv --out shapes/
    python batch_shapes.py --designation "(75) Eurydike" --out shapes/
"""
import argparse, json, os, re, subprocess, sys, time, zlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# binaries from the fork CheerfulUser/DAMIT-convex (`make` in its top directory builds both);
# DAMIT_CONVEX is the clone, defaulting to shape/DAMIT-convex (the symlink layout used on ozstar)
DAMIT_CONVEX = os.environ.get('DAMIT_CONVEX', os.path.join(HERE, 'DAMIT-convex'))
# The fork's convexinv calls BLAS; one convexinv per worker, so keep each to a single thread (OpenBLAS
# otherwise starts one per core and oversubscribes the node). Inherited by every convexinv subprocess.
for _v in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
CONVEXINV = os.path.join(DAMIT_CONVEX, 'convexinv/convexinv')
MINKOWSKI = os.path.join(DAMIT_CONVEX, 'minkowski')
POINTS_MAX = int(os.environ.get('CI_POINTS_MAX', 3000))
HARM = int(os.environ.get('CI_HARM', 6))
NROWS = int(os.environ.get('CI_NROWS', 6))
MIN_PTS = 50
NB = 72
# 0.1 let the solver dump unconstrained area into a spin-axis facet; 1.0 collapses that from
# 18-34% to ~2-3% of the surface with no cost in fit quality (rms flat over 0.03-3.0).
CONVEXITY_W = float(os.environ.get('CI_CONVEX', 1.0))
# Pole FIXED, not fitted. Single-apparition data cannot determine it -- all eight starting
# orientations fit to within 1-10% in rms -- and fitting it anyway lets the solver trade pole
# against shape, dumping unconstrained area into a facet on the spin axis (Link: 18-34% of its
# surface in one face, falling to 2.3% once fixed). One start is therefore sufficient.
FIX_POLE = os.environ.get('CI_FIX_POLE', '1') == '1'
POLE_SCAN = int(os.environ.get('CI_POLE_SCAN', 12))
_ALL_POLES = [(0, 0), (90, 0), (180, 45), (270, -45), (45, 60), (135, -30), (225, 20), (315, -60)]
_OBL = np.radians(23.4392911)  # J2000 mean obliquity: ICRS -> ecliptic J2000
_ICRS_TO_ECL = np.array([[1, 0, 0], [0, np.cos(_OBL), np.sin(_OBL)], [0, -np.sin(_OBL), np.cos(_OBL)]])


def perpendicular_poles(ev, n=POLE_SCAN):
    """Candidate axes perpendicular to the mean observer direction, i.e. aspect 90 deg.

    The perpendicular constraint is what keeps the geometry equator-on: a blind pole left
    (75) Eurydike 5.9 deg from pole-on, where no shape can produce its amplitude. But
    "perpendicular" is a CIRCLE, and the choice within it changes the folded fit by 1.3-2.7x
    across the four test objects, so it is scanned rather than assumed.

    ev must be in the ecliptic frame build_geometry returns; lon/lat are read straight off
    the same vectors so the pole cannot drift into a different frame from the geometry."""
    en = (ev / np.linalg.norm(ev, axis=1, keepdims=True)).mean(axis=0)
    en /= np.linalg.norm(en)
    a = np.array([0.0, 0.0, 1.0]) - np.dot([0.0, 0.0, 1.0], en) * en
    if np.linalg.norm(a) < 1e-6:
        a = np.array([1.0, 0.0, 0.0]) - en[0] * en
    a /= np.linalg.norm(a)
    b = np.cross(en, a)
    out = []
    for ang in np.linspace(0, np.pi, n, endpoint=False):
        v = np.cos(ang) * a + np.sin(ang) * b
        out.append((float(np.degrees(np.arctan2(v[1], v[0])) % 360.0),
                    float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))))
    return out


def folded_rms(df, model, period_hr, blocks, nb=NB):
    """Model-minus-data rms on the folded, binned curve, after per-session rescaling.

    Per-point rms is noise-dominated and barely moves between good and bad orientations
    (Wuyeesun: 0.0751-0.0770 across the whole circle, while the binned rms spans 2.3x)."""
    obs = df['rel_flux'].values
    m = np.asarray(model, dtype=float).copy()
    for idx in blocks:
        mm = m[idx].mean()
        if np.isfinite(mm) and mm != 0:
            m[idx] *= obs[idx].mean() / mm
    per = period_hr / 24.0
    ph = (df['mjd'].values % per) / per
    bi = np.clip((ph * nb).astype(int), 0, nb - 1)
    cnt = np.bincount(bi, minlength=nb)
    dm = np.bincount(bi, weights=obs, minlength=nb) / np.maximum(cnt, 1)
    mm2 = np.bincount(bi, weights=m, minlength=nb) / np.maximum(cnt, 1)
    ok = cnt >= 3
    return float(np.sqrt(np.mean((mm2[ok] - dm[ok]) ** 2))) if ok.sum() else float('inf')
T_INV = int(os.environ.get('T_INV', 900))
# minkowski on the 39,431-object store run: median 4.9 s, 99th percentile 25 s; 281 objects hit a 30 s
# timeout. minkowski_seconds is recorded per object.
T_MINK = int(os.environ.get('T_MINK', 30))
T_POLY = int(os.environ.get('T_POLY', 300))
# The mesh comes from minkowski_py (a convex-optimisation Minkowski solver: ~1 s, reproduces the facet
# areas 9-30x more closely than DAMIT's minkowski, and handles the (near-)zero facet areas that made
# minkowski fail outright on 1,429 objects of the store run). DAMIT's minkowski is the fallback on the same
# shape, then the next-best orientations (mesh_orientation_rank > 0 records that the published shape is not
# the best fit), then polyhedrec.
MESH_TRIES = int(os.environ.get('MESH_TRIES', 3))
# validation sample: for this fraction of objects (fixed by a hash of the designation), DAMIT's minkowski also
# meshes the adopted shape and the two meshes are compared (mesh_compare_* columns)
MESH_COMPARE_FRAC = float(os.environ.get('MESH_COMPARE_FRAC', 0.02))


def sn(d):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(d)).strip('_')


def prepare(df):
    """Deduplicate cut overlap, drop under-populated visits, sort.

    clean_lightcurves/ stores raw `flux`, not `rel_flux` -- the per-visit median normalisation
    is step 4 of the production path and is what puts separate visits and sectors onto one
    brightness scale. Without it the inversion sees each visit's zeropoint as real variation.
    """
    if 'rel_flux' not in df.columns:
        m = df.groupby('visit')['flux'].transform('median')
        ok = m.notna() & (m > 0)
        df = df[ok].copy()
        df['rel_flux'] = df['flux'] / m[ok]
    df = df.dropna(subset=['mjd', 'rel_flux', 'ra', 'dec', 'delta_au']).copy()
    df['_k'] = df['mjd'].round(6)
    df['_vn'] = df.groupby('visit')['mjd'].transform('size')
    df = df.sort_values(['_k', '_vn'], ascending=[True, False]).drop_duplicates('_k')
    df = df[df.groupby('visit')['mjd'].transform('size') >= MIN_PTS]
    return df.sort_values('mjd').reset_index(drop=True)


def build_geometry(df):
    """Asteroid-centric Sun and Earth vectors (au) in the ECLIPTIC J2000 frame, as convexinv
    requires. Built in ICRS, then rotated by the J2000 mean obliquity."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    t = Time(df['mjd'].values, format='mjd', scale='utc')
    ra, dec = np.radians(df['ra'].values), np.radians(df['dec'].values)
    u = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    ep = get_body_barycentric('earth', t).xyz.to('au').value.T
    sp = get_body_barycentric('sun', t).xyz.to('au').value.T
    ap = ep + df['delta_au'].values[:, None] * u
    return (sp - ap) @ _ICRS_TO_ECL.T, (ep - ap) @ _ICRS_TO_ECL.T


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


C_AU_PER_DAY = 173.1446326846693


def lt_jd(df, ev):
    """Light-time-corrected JD: convexinv requires it, and the observer distance changes by
    tenths of an au over a multi-week baseline. Anything phasing data against the model's
    rotation (t0, phi0) must use these epochs too."""
    return df['mjd'].values + 2400000.5 - np.linalg.norm(ev, axis=1) / C_AU_PER_DAY


def write_lcs(df, sv, ev, blocks, path):
    """Code 0 = relative lightcurve (1 would tell convexinv the fluxes are calibrated)."""
    jd = lt_jd(df, ev)
    fl = df['rel_flux'].values
    with open(path, 'w') as f:
        f.write(f'{len(blocks)}\n')
        for idx in blocks:
            f.write(f'{len(idx)} 0\n')
            for i in idx:
                f.write(f'{jd[i]:.6f} {fl[i]:.6f} '
                        f'{sv[i,0]:.6f} {sv[i,1]:.6f} {sv[i,2]:.6f} '
                        f'{ev[i,0]:.6f} {ev[i,1]:.6f} {ev[i,2]:.6f}\n')


def write_control(path, lam, bet, per):
    with open(path, 'w') as f:
        _fl = '0' if FIX_POLE else '1'
        f.write(f'{lam}\t\t{_fl}\tinital lambda\n{bet}\t\t{_fl}\tinitial beta\n'
                f'{per}\t\t1\tinital period\n')
        f.write('0\t\t\tzero time\n0\t\t\tinitial rotation angle\n'
                f'{CONVEXITY_W}\t\t\tconvexity regularization\n')
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
        # A convexinv facet with zero area comes back as a one-vertex "face". It carries no area,
        # and check_convex counted it as non-convex: (7694) Krasetin, (17974) 1999 JL52 and
        # (157253) 2004 RD151 were rejected on 2, 1 and 4 of these with no real violation
        # (largest 6e-11 of the mesh size).
        F = [f for f in F if len(set(f)) >= 3]
        return V - V.mean(axis=0), F
    except Exception:
        return None


_POLY_CODE = """
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + '/polyhedrec')
from polyhedrec_fast import reconstruct_fast
L = open(sys.argv[2]).read().split()
n = int(L[0]); x = np.array(L[1:1 + 4 * n], dtype=float).reshape(n, 4)
p = reconstruct_fast(x[:, 1:] / np.linalg.norm(x[:, 1:], axis=1)[:, None], x[:, 0])
V = np.array(p.vertices, dtype=float)
F = []
for f in p.faces:
    ids = list(f.vertices)
    q = V[ids]
    nrm = sum(np.cross(q[k], q[(k + 1) % len(ids)]) for k in range(len(ids)))
    F.append(ids if np.dot(nrm, f.unormal) >= 0 else ids[::-1])
print(json.dumps({'v': V.tolist(), 'f': F}))
"""


def polyhedrec_mesh(shape_path):
    """Fallback reconstructor, in a subprocess so T_POLY can stop it (it ran 33 min on Deflotte and failed)."""
    try:
        r = subprocess.run([sys.executable, '-c', _POLY_CODE, HERE, shape_path], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=T_POLY)
        m = json.loads(r.stdout.decode())
        V = np.array(m['v'])
        F = [f for f in m['f'] if len(set(f)) >= 3]
        return V - V.mean(axis=0), F
    except Exception:
        return None


def minkowski_py_mesh(shape_path):
    try:
        import minkowski_py
        V, F, info = minkowski_py.from_areas_file(shape_path)
        return (V, F) if info['converged'] and info['max_area_error'] < 1e-3 else None
    except Exception:
        return None


def caliper_widths(V):
    """Max and min width of the mesh perpendicular to the spin axis (z), over all directions."""
    ang = np.radians(np.arange(0.0, 180.0, 0.5))
    w = np.ptp(V[:, 0][:, None] * np.cos(ang) + V[:, 1][:, None] * np.sin(ang), axis=0)
    return w.max(), w.min()


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

        starts = perpendicular_poles(ev) if FIX_POLE else _ALL_POLES
        rec['n_orientations'] = len(starts)
        sols, fails = [], []
        for i, (l0, b0) in enumerate(starts):
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
        if FIX_POLE:
            for sol in sols:
                try:
                    sol['brms'] = folded_rms(df, np.loadtxt(sol['fit']), sol['per'], blocks)
                except Exception:
                    sol['brms'] = float('inf')
            best = min(sols, key=lambda s: s['brms'])
            _f = sorted(s['brms'] for s in sols if np.isfinite(s['brms']))
            rec['orientation_rms_min'] = _f[0] if _f else None
            rec['orientation_rms_max'] = _f[-1] if _f else None
        else:
            best = min(sols, key=lambda s: abs(s['lam'] - np.median(lam))
                       + abs(s['bet'] - np.median(bet)))
        rec.update(lambda_circ_std_deg=lam_std, beta_std_deg=float(bet.std()),
                   period_std_s=float(per.std() * 3600),
                   representative_lambda_deg=best['lam'], representative_beta_deg=best['bet'],
                   model_period_hr=best['per'],
                   # meaningless when the pole is fixed: the scatter is zero by construction
                   pole_constrained=(None if FIX_POLE else bool(lam_std < 20)),
                   pole_fixed=FIX_POLE)

        # mesh: minkowski on the best orientation, then the next-best ones, then polyhedrec on the best
        order = sorted(sols, key=lambda q: q.get('brms', 0.0))[:MESH_TRIES] if FIX_POLE else [best]
        attempts = ([(0, order[0], 'minkowski_py'), (0, order[0], 'minkowski')]
                    + [(rank, q, m_) for rank, q in enumerate(order) if rank for m_ in ('minkowski_py', 'minkowski')]
                    + [(0, order[0], 'polyhedrec')])
        mk, why = None, 'mesh reconstruction failed'
        for rank, q, method in attempts:
            _tm = time.time()
            got = {'minkowski': minkowski, 'minkowski_py': minkowski_py_mesh, 'polyhedrec': polyhedrec_mesh}[method](q['shape'])
            rec[f'{method}_seconds'] = round(rec.get(f'{method}_seconds', 0.0) + time.time() - _tm, 2)
            if got is None:
                continue
            nbad = check_convex(*got)
            if nbad:
                why = f'{nbad} non-convex faces'
                continue
            mk, best = got, q
            rec.update(mesh_method=method, mesh_orientation_rank=rank)
            break
        if mk is None:
            rec['error'] = why; return rec
        V, F = mk
        if zlib.crc32(tag.encode()) % 10000 < MESH_COMPARE_FRAC * 10000:
            _tm = time.time()
            mm = minkowski(best['shape']) if rec['mesh_method'] != 'minkowski' else None
            rec['mesh_compare_minkowski_seconds'] = round(time.time() - _tm, 2)
            if mm is not None:
                from scipy.spatial import ConvexHull, cKDTree
                # both meshes centred; scaled to unit volume, since each reconstructor picks its own size
                a_ = V / ConvexHull(V).volume ** (1 / 3)
                b_ = mm[0] / ConvexHull(mm[0]).volume ** (1 / 3)
                d = max(cKDTree(b_).query(a_)[0].max(), cKDTree(a_).query(b_)[0].max())
                rec['mesh_compare_max_vertex_dist_rel'] = float(d / np.ptp(a_, axis=0).max())
            else:
                rec['mesh_compare_max_vertex_dist_rel'] = None
        rec.update(n_verts=len(V), n_facets=len(F), nonconvex_faces=0,
                   representative_lambda_deg=best['lam'], representative_beta_deg=best['bet'],
                   model_period_hr=best['per'])
        # z is the spin axis, so only the EQUATORIAL ratio relates to lightcurve amplitude.
        # Sorting all three extents and calling the top two "a/b" reported the polar ratio for
        # elongated-along-z bodies -- it gave 1.54 for Link whose equatorial ratio is 1.08.
        # Caliper widths, not the x/y bounding box: a body lying diagonally in x-y has near-equal
        # box extents whatever its elongation ((194793) 2001 YP90: box 1.15, caliper 1.66).
        wmax, wmin = caliper_widths(V)
        rec.update(equatorial_ratio=float(wmax / wmin), polar_ratio=float(np.ptp(V[:, 2]) / wmin))

        # folded curve + per-session-rescaled model on one grid
        pl = open(best['params']).read().split()
        # convexinv writes phi0 in DEGREES (convexinv.c: Phi_0 * RAD2DEG); 0 in every run so far
        t0j, phi0 = float(pl[3]), np.radians(float(pl[4]))
        # rotation zero-point, so the model can be phased to any epoch (e.g. an occultation):
        # phi(t) = phi0 + 2 pi (t_lt - t0) / P, t_lt light-time corrected, as convexinv defines it
        rec.update(rotation_t0_jd=t0j, rotation_phi0_rad=phi0)
        jd = lt_jd(df, ev)
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
    # clean_lightcurves/ uses *_clean_lc.csv; lcdb_lcs/stacked/ uses *_lc.csv
    ap.add_argument('--lcsuffix', default='_lc.csv')
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
                      f"{a.lcdir}/{sn(r['designation'])}{a.lcsuffix}", a.work, a.out)
        recs.append(rec)
        print(f"  {rec['key']:26s} {'ok' if rec['ok'] else 'FAIL'} {rec['seconds']:6.1f}s "
              f"{rec.get('error','')}", flush=True)
    pd.DataFrame(recs).to_csv(f'{a.out}/_status_{a.index:04d}.csv', index=False)
    n = sum(r['ok'] for r in recs)
    print(f'\n{n}/{len(recs)} succeeded', flush=True)
