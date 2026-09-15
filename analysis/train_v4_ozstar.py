"""Train the v4 reliability classifier natively on ozstar (sklearn 1.8.0).

A pickle trained locally (1.9.0) cannot be loaded here -- it raises
"<class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module" -- so the model is
rebuilt from the version-independent table exported by export_v4_training.py.

The dev/test split comes from the `split` column of that table, NOT from a fresh
train_test_split, so the held-out 200 are exactly the objects the local metrics were measured
on. Reported held-out numbers should therefore match the local run closely; a large discrepancy
means something differs beyond the sklearn version and should be investigated before deploying.

Local reference: held-out precision 90.9%, recall 92.6%, accuracy 95.5%, AUC 0.9803.
"""
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

TABLE = '/fred/oz335/rridden/asteroids/v4_training_table.csv'
OUT = '/fred/oz335/rridden/asteroids/reliability_classifier.joblib'
THRESH = 0.5640          # chosen on the dev set locally; NOT re-tuned here
W_LCDB = 0.5

t = pd.read_csv(TABLE, low_memory=False)
F = [c for c in t.columns if c not in ('designation', 'split', 'y', 'w')]
xf = lambda d: d[F].astype(float).replace([np.inf, -np.inf], np.nan).values
dev, test, lab = t[t.split == 'dev'], t[t.split == 'test'], t[t.split == 'lcdb']
print(f'{len(t):,} rows: {len(dev)} dev, {len(test)} test, {len(lab):,} lcdb')
print(f'{len(F)} features')

X = np.vstack([xf(dev), xf(lab)])
y = np.concatenate([dev.y.values, lab.y.values])
w = np.concatenate([dev.w.values, lab.w.values * W_LCDB])
mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                     l2_regularization=1.0, min_samples_leaf=20,
                                     random_state=0).fit(X, y, sample_weight=w)

st = mdl.predict_proba(xf(test))[:, 1]
yt = test.y.values
p = st >= THRESH
tp = int((p & (yt == 1)).sum()); fp = int((p & (yt == 0)).sum())
fn = int((~p & (yt == 1)).sum()); tn = int((~p & (yt == 0)).sum())
prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
auc = roc_auc_score(yt, st)
print(f'\nheld-out test ({len(yt)} objects):')
print(f'  precision {prec:.1%}   recall {rec:.1%}   accuracy {(tp+tn)/len(yt):.1%}   AUC {auc:.4f}')
print(f'  false accepts {fp}, false rejects {fn}')
print('  local reference: precision 90.9%, recall 92.6%, accuracy 95.5%, AUC 0.9803')
if abs(auc - 0.9803) > 0.03:
    print('  WARNING: AUC differs from the local run by more than 0.03 -- investigate before use')

joblib.dump({'model': mdl, 'features': F, 'calibrator': None, 'threshold': THRESH,
             'floors': {'n_points': 100, 'aov_F': 1.5}, 'lcdb_weight': W_LCDB,
             'trained_on': '800 hand-vetted + vetted/gated LCDB (lcdb_updated.csv)',
             'note': 'v4; trained natively on ozstar from v4_training_table.csv'}, OUT)
print(f'\nsaved {OUT}')
