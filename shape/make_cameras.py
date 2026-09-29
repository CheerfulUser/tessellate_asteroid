"""Recover the real per-object viewing geometry for every published shape model.

build_pages.py hardcoded the observer to (1,0,0) in the pole frame for every object, on the
assumption that a pole fixed perpendicular to the mean observer makes all objects identical.
That fixes the ASPECT but not the AZIMUTH: rotation about the spin axis is object-dependent and
acts as a constant phase offset, so the viewer showed the wrong face at a given lightcurve phase.

This reproduces build_shape_viewer.py's derivation exactly -- same frame convention, same signed
aspect -- and emits one row per object for patching back into the pages. No refitting.

Run on ozstar from /fred/oz335/rridden/asteroids.
"""
import csv, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'shape')
from batch_shapes import prepare, build_geometry, sn   # noqa: E402

OUT = 'shapes/out'
LCDIR = 'clean_lightcurves'
LCSUF = '_clean_lc.csv'
IDX = int(sys.argv[1]) if len(sys.argv) > 2 else 0
TOT = int(sys.argv[2]) if len(sys.argv) > 2 else 1
DEST = f'shapes/cameras/cameras_{IDX:04d}.csv'


def mat2quat(R):
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    return (q / np.linalg.norm(q)).tolist()


def camera(df, lam_deg, bet_deg):
    sv, ev = build_geometry(df)
    en = ev / np.linalg.norm(ev, axis=1, keepdims=True)
    snv = sv / np.linalg.norm(sv, axis=1, keepdims=True)
    l, b = np.radians(lam_deg), np.radians(bet_deg)
    Rz = np.array([[np.cos(l), np.sin(l), 0], [-np.sin(l), np.cos(l), 0], [0, 0, 1]])
    a = np.pi / 2 - b
    Ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    Rpf = Ry @ Rz
    epf = (Rpf @ en.T).T.mean(axis=0); epf /= np.linalg.norm(epf)
    spf = (Rpf @ snv.T).T.mean(axis=0); spf /= np.linalg.norm(spf)
    # SIGNED aspect: which hemisphere the observer sits in is real information
    aspect = float(np.degrees(np.arccos(np.clip(epf[2], -1, 1))))
    zc = epf
    up = np.array([0.0, 0.0, 1.0]) - np.dot(zc, [0, 0, 1.0]) * zc
    if np.linalg.norm(up) < 1e-6:                      # pole-on degenerate case
        up = np.array([0.0, 1.0, 0.0]) - np.dot(zc, [0, 1.0, 0]) * zc
    yc = up / np.linalg.norm(up)
    xc = np.cross(yc, zc)
    Rcam = np.vstack([xc, yc, zc])
    return mat2quat(Rcam), (Rcam @ spf).tolist(), aspect


files = sorted(glob.glob(f'{OUT}/*.json'))
files = [f for f in files if not os.path.basename(f).startswith('_')]
files = files[IDX::TOT]
os.makedirs('shapes/cameras', exist_ok=True)
print(f'{len(files):,} shape models', flush=True)

rows, bad = [], 0
for i, f in enumerate(files):
    key = os.path.basename(f)[:-5]
    try:
        d = json.load(open(f))
        s = d['solution']
        lam, bet = s.get('representative_lambda_deg'), s.get('representative_beta_deg')
        if lam is None or bet is None:
            bad += 1; continue
        df = prepare(pd.read_csv(f'{LCDIR}/{sn(s["designation"])}{LCSUF}'))
        if not len(df):
            bad += 1; continue
        q, sc, asp = camera(df, lam, bet)
        rows.append(dict(key=key,
                         q0=q[0], q1=q[1], q2=q[2], q3=q[3],
                         sx=sc[0], sy=sc[1], sz=sc[2], aspect=asp))
    except Exception as e:
        bad += 1
        if bad <= 5:
            print(f'  {key}: {type(e).__name__}: {e}', flush=True)
    if (i + 1) % 1000 == 0:
        print(f'  {i+1:,}/{len(files):,}  ok={len(rows):,} bad={bad}', flush=True)

with open(DEST, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
a = np.array([r['aspect'] for r in rows])
print(f'\nwrote {DEST}: {len(rows):,} cameras, {bad} failed')
print(f'aspect: median {np.median(a):.1f}  p5 {np.percentile(a,5):.1f}  p95 {np.percentile(a,95):.1f}')
print(f'within 5 deg of equator-on: {np.mean(np.abs(a-90) < 5):.1%}')
