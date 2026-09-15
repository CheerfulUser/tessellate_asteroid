"""Retrain on the 800 hand labels plus the vetted/gated LCDB table.

Target throughout is the same question: IS OUR PIPELINE'S PERIOD CORRECT? The label for each
LCDB row follows from what the vetting established about that row, not from raw agreement with
the published value -- which is the mistake the earlier models made, since 42% of raw LCDB
negatives turned out to be LCDB errors rather than ours.

  POSITIVE  our period independently corroborated
    TESS_precision_LCDB_harmonic  2,958  matches a 1x/2x multiple of LCDB, good data
    LCDB_native_rescale_below_cut 3,759  matches a multiple, but OUR data is poor -- the period
                                         agreement is still evidence we are right, so positive,
                                         down-weighted for the noise
    TESS_replaces_wrong_LCDB_VETTED 281  user confirmed LCDB wrong and our period right
    TESS_replaces_wrong_LCDB        116  passed the cut calibrated at 98.1% purity
  NEGATIVE  our period not corroborated
    LCDB_native_VETTED_keep          89  user confirmed our period should NOT be adopted
    LCDB_native_below_cut           470  failed the cut
    LCDB_native_TESS_disagrees    2,264  unresolved disagreement

Excluded: LCDB_native rows TESS never observed (no information about our pipeline).

Weighting reflects how each label was established -- eye-vetted rows carry full weight, rows
resting on a statistical cut less, and the noisy-but-agreeing rescale rows least. The 800 hand
labels are the only ones independent of LCDB entirely and are the evaluation set: all reported
precision/recall is out-of-fold on them.
"""
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve

F = ['aov_F', 'amp_snr', 'max_phase_gap', 'n_cycles_observed', 'n_points', 'loo_worst_chi2',
     'power_ratio', 'fap', 'n_visits', 'n_sectors', 'baseline_hr', 'n_outliers_rejected',
     'frac_points_stacked', 'phase_angle_range_deg', 'median_n_stack', 'pk_over_med',
     'pk_prom', 'pk_snr']
FLOOR_N, FLOOR_A = 100, 1.5
xf = lambda d: d[F].astype(float).replace([np.inf, -np.inf], np.nan).values

POS = {'TESS_precision_LCDB_harmonic': 0.6, 'LCDB_native_rescale_below_cut': 0.2,
       'TESS_replaces_wrong_LCDB_VETTED': 1.0, 'TESS_replaces_wrong_LCDB': 0.6}
NEG = {'LCDB_native_VETTED_keep': 1.0, 'LCDB_native_below_cut': 0.6,
       'LCDB_native_TESS_disagrees': 0.4}

lc = pd.read_csv('comparison_data/lcdb_updated.csv', low_memory=False)
cat = pd.read_csv('population_figs/all_sector_report_v4.csv', low_memory=False)
m = pd.read_csv('comparison_data/label800_labelled.csv')
m_all = m.copy()

lab = lc[lc.period_source.isin(POS | NEG) & lc.designation.notna()].copy()
lab['y'] = lab.period_source.isin(POS).astype(int)
lab['w'] = lab.period_source.map({**POS, **NEG})
lab = lab.merge(cat[['designation'] + [c for c in F if c not in lab.columns]],
                on='designation', how='left')
lab = lab[lab.designation.notna()].drop_duplicates('designation')
# never train on an object that is in the hand-labelled evaluation set
lab = lab[~lab.designation.isin(set(m_all.designation))]   # exclude BOTH dev and test
print(f'LCDB-derived training rows: {len(lab):,}  ({lab.y.mean():.1%} positive)')
print(lab.groupby('period_source').size().to_string())

# TRUE HOLDOUT. Selecting the LCDB weight AND the decision threshold on the same out-of-fold
# predictions used to report precision is selection on the evaluation data, and biases the
# reported number upward. 25% of the hand labels are set aside before anything is fitted, are
# never seen during CV or threshold selection, and carry the only unbiased estimate.
from sklearn.model_selection import train_test_split
m_dev, m_test = train_test_split(m, test_size=0.25, stratify=m['believed'],
                                 random_state=42)
print(f'holdout: {len(m_dev)} dev / {len(m_test)} test '
      f'({m_test["believed"].mean():.1%} positive in test)')
Xt, yt = xf(m_test), m_test['believed'].astype(int).values
m = m_dev
Xh, yh = xf(m), m['believed'].astype(int).values
Xl, yl, wl = xf(lab), lab.y.values, lab.w.values
print(f'\nhand labels (evaluation set): {len(m)}  ({yh.mean():.1%} positive)')

mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                            l2_regularization=1.0, min_samples_leaf=20,
                                            random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
cur_p = ((m.is_reliable) & (m.believed)).sum() / max(m.is_reliable.sum(), 1)
cur_r = ((m.is_reliable) & (m.believed)).sum() / max(m.believed.sum(), 1)
print(f'current pipeline: precision {cur_p:.1%} at recall {cur_r:.1%}\n')
print(f'{"w_lcdb":>7} {"AUC":>7} {"prec@r86":>9}')

best = None
for w in (0.0, 0.25, 0.5, 1.0, 2.0):
    oof = np.zeros(len(m))
    for tr, te in cv.split(Xh, yh):
        if w > 0:
            XX = np.vstack([Xh[tr], Xl]); yy = np.concatenate([yh[tr], yl])
            ww = np.concatenate([np.ones(len(tr)), wl * w])
            mdl = mk().fit(XX, yy, sample_weight=ww)
        else:
            mdl = mk().fit(Xh[tr], yh[tr])
        oof[te] = mdl.predict_proba(Xh[te])[:, 1]
    auc = roc_auc_score(yh, oof)
    pr, rc, th = precision_recall_curve(yh, oof)
    i = int(np.argmin(np.abs(rc - cur_r)))
    print(f'{w:7.2f} {auc:7.4f} {pr[i]:8.1%}')
    if best is None or pr[i] > best[0]:
        best = (pr[i], w, float(th[min(i, len(th) - 1)]), auc)

p, w, TH, auc = best
print(f'\nchosen w_lcdb={w}: precision {p:.1%} at recall {cur_r:.1%} '
      f'(current {cur_p:.1%}), AUC {auc:.4f}')
if w > 0:
    XX = np.vstack([Xh, Xl]); yy = np.concatenate([yh, yl])
    ww = np.concatenate([np.ones(len(yh)), wl * w])
    final = mk().fit(XX, yy, sample_weight=ww)
else:
    final = mk().fit(Xh, yh)
joblib.dump({'model': final, 'features': F, 'threshold': TH,
             'floors': {'n_points': FLOOR_N, 'aov_F': FLOOR_A}, 'lcdb_weight': w,
             'trained_on': '800 hand-vetted + vetted/gated LCDB (lcdb_updated.csv)',
             'note': 'labels from the vetted replacement/rescale regime, not raw LCDB agreement'},
            'comparison_data/reliability_classifier_v4.joblib')
print('wrote comparison_data/reliability_classifier_v4.joblib')

# unbiased estimate on data never used for fitting or threshold selection
st = final.predict_proba(Xt)[:, 1]
pred = st >= TH
tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum())
fn = int((~pred & (yt == 1)).sum()); tn = int((~pred & (yt == 0)).sum())
print(f'\nHELD-OUT TEST ({len(yt)} objects, never seen in fitting or threshold selection):')
print(f'  precision {tp/max(tp+fp,1):.1%}   recall {tp/max(tp+fn,1):.1%}   '
      f'accuracy {(tp+tn)/len(yt):.1%}   AUC {roc_auc_score(yt, st):.4f}')
print(f'  false accepts {fp}, false rejects {fn}')
cur_tp = int(((m_test.is_reliable) & (m_test.believed)).sum())
cur_fp = int(((m_test.is_reliable) & (~m_test.believed)).sum())
cur_fn = int(((~m_test.is_reliable) & (m_test.believed)).sum())
print(f'  current pipeline on the same holdout: precision {cur_tp/max(cur_tp+cur_fp,1):.1%}   '
      f'recall {cur_tp/max(cur_tp+cur_fn,1):.1%}')

sc = cat[cat.period_hr.notna()].copy()
s = final.predict_proba(xf(sc))[:, 1]
acc = (s >= TH) & (sc.n_points >= FLOOR_N).values & (sc.aov_F >= FLOOR_A).values
print(f'\napplied to the catalogue: {int(acc.sum()):,} of {len(sc):,} pass '
      f'({acc.mean():.1%});  current is_reliable = {int(sc.is_reliable.sum()):,}')
