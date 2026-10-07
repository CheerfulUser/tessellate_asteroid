"""Fit the reliability classifier natively on ozstar (sklearn 1.8.0 there vs 1.9.0 locally --
joblib pickles do not reliably cross that boundary; the input is a version-independent CSV).

Training set: LCDB U=3 ("secure") entries ONLY. U=2 is 78% of the labels but its reliability
collapses at long periods -- only 5.3% of LCDB entries above 72hr are secure, and our agreement
with U=2 there falls to 26% while U=3 holds at 81-88%. Training on U=2 taught the model to
distrust exactly the regime where TESS's continuous coverage beats ground-based photometry.

Target: EXACT agreement with the literature period (|P/P_lcdb - 1| < 0.02). An earlier lenient
target that also credited the 0.5x alias was masking the under-doubling bug.
"""
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

TABLE = '/fred/oz335/rridden/asteroids/classifier_training_table.csv'
# the pipeline loads the deployed classifier from the asteroid store (sector_report_pipeline.RELIABILITY_MODEL_PATH)
OUT   = '/fred/oz335/TESSdata/asteroid_store/reliability_classifier.joblib'

t = pd.read_csv(TABLE, low_memory=False)
F = [c for c in t.columns if c not in ('correct', 'exact', 'U_clean', 'designation',
                                       'period_hr', 'lcdb_period')]
X = lambda d: d[F].astype(float).replace([np.inf, -np.inf], np.nan).values
y = t['exact'].astype(int).values
mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                            l2_regularization=1.0, min_samples_leaf=20,
                                            random_state=0)
print(f'training on {len(t)} LCDB U=3 objects, exact-match rate {y.mean():.1%}')
print(f'{len(F)} features: {F}')

cv = StratifiedKFold(5, shuffle=True, random_state=0)
oof = cross_val_predict(mk(), X(t), y, cv=cv, method='predict_proba')[:, 1]
auc = roc_auc_score(y, oof)
print(f'5-fold CV AUC = {auc:.4f}   (0.8741 measured locally -- flag any large discrepancy)')
if auc < 0.82:
    print('WARNING: AUC well below the local value; check for sklearn version drift')

iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(oof, y)
print('calibration (out-of-fold): predicted -> actual')
q = iso.predict(oof)
for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)]:
    m = (q >= lo) & (q < hi)
    if m.sum() >= 25:
        print(f'   {lo:.2f}-{hi:<5.2f} n={m.sum():4d}  actual {y[m].mean():.2f}')

final = mk().fit(X(t), y)
joblib.dump({'model': final, 'features': F, 'calibrator': iso}, OUT)
print(f'saved {OUT}')
