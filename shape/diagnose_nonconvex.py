"""Why do (7694) Krasetin, (17974) 1999 JL52 and (157253) 2004 RD151 fail check_convex after the
frame fix, when their published fits passed? Reruns process() with check_convex wrapped to
record the size of every violation, relative to the mesh, instead of just counting faces.

Run on ozstar from /fred/oz335/rridden/asteroids.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import batch_shapes as bs  # noqa: E402

KEYS = ['7694_Krasetin', '17974_1999_JL52', '157253_2004_RD151']
OUT = 'shape_frame_fix_validation/diag_out'
WORK = 'shape_frame_fix_validation/diag_work'
record = {}


def check_convex_recording(V, F):
    viol, degenerate = [], []
    for f in F:
        p = V[f]
        n = np.zeros(3)
        for k in range(len(f)):
            n += np.cross(p[k], p[(k + 1) % len(f)])
        nl = np.linalg.norm(n)
        if nl == 0:
            degenerate.append(dict(n_idx=len(f), n_unique_idx=len(set(f)),
                                   n_unique_xyz=len(np.unique(np.round(p, 12), axis=0))))
            continue
        n /= nl
        viol.append(float((V @ n - np.dot(n, p.mean(0))).max()))
    viol = np.array(viol)
    size = float(np.ptp(V, axis=0).max())
    record.update(n_degenerate=len(degenerate), degenerate_faces=degenerate[:6],
                  n_real_violations=int((viol > 1e-6).sum()), max_violation=float(viol.max()),
                  mesh_size=size, max_violation_rel=float(viol.max() / size),
                  n_verts=len(V), n_faces=len(F),
                  published_check_count=len(degenerate) + int((viol > 1e-6).sum()))
    return record['published_check_count']


bs.check_convex = check_convex_recording
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
for key in KEYS:
    s = json.load(open(f'shapes/out/{key}.json'))['solution']
    record.clear()
    rec = bs.process(s['designation'], s['adopted_period_hr'],
                     f"clean_lightcurves/{bs.sn(s['designation'])}_clean_lc.csv", WORK, OUT)
    print(f"\n{key}: ok={rec.get('ok')} error={rec.get('error')}")
    print({k: v for k, v in record.items()})
