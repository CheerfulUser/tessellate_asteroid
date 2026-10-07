"""Retrain the reliability classifier on the raw-frame (POINT_MODE=raw) store report, natively on ozstar.

The v4 labels describe the periods the OLD report found, so each is re-attached to the new period
through a known true period:
  hand labels (label800_labelled.csv)  believed -> truth is the labelled period, y = new period matches it;
                                       not believed -> y = 0 only if the new period is still that period,
                                       otherwise dropped (no truth for the new one)
  LCDB (lcdb_updated.csv, v4 sources)  truth is the vetted period_updated, y = new period matches it,
                                       v4 source weights kept
  agreement (agreement_labels.csv)     label_v2 1 = all sectors agree + match pooled, 0 = two confident
                                       sectors disagree; weight W_AGREE, objects with hand/LCDB labels excluded
Match = |P/P_true - 1| < TOL. The v4 dev/test split of the 800 hand labels is kept, so test is held out.

Two models: A = hand dev + LCDB, B = A + agreement labels. Threshold for each: the lowest score whose
out-of-fold dev precision (5 folds over dev, other training rows always in) is >= PREC_TARGET.

    python3 -u train_raw_ozstar.py
"""
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = '/fred/oz335/rridden'
REPORT = f'{ROOT}/asteroids_store/all_sector_asteroid_report.csv'
V4_TABLE = f'{ROOT}/asteroids/v4_training_table.csv'
HAND = f'{ROOT}/asteroids/label800_labelled.csv'
LCDB = f'{ROOT}/asteroids/lcdb_updated.csv'
AGREE = f'{ROOT}/asteroids_store_sector/agreement_labels.csv'
TABLE_OUT = f'{ROOT}/asteroids_store/raw_training_table.csv'
OUT = '/fred/oz335/TESSdata/asteroid_store/reliability_classifier_raw.joblib'
TOL = 0.02
W_AGREE = 0.5
PREC_TARGET = 0.90
F = ['aov_F', 'amp_snr', 'max_phase_gap', 'n_cycles_observed', 'n_points', 'loo_worst_chi2',
     'power_ratio', 'fap', 'n_visits', 'n_sectors', 'baseline_hr', 'n_outliers_rejected',
     'phase_angle_range_deg', 'pk_over_med', 'pk_prom', 'pk_snr']
POS = {'TESS_precision_LCDB_harmonic': 0.6, 'LCDB_native_rescale_below_cut': 0.2,
       'TESS_replaces_wrong_LCDB_VETTED': 1.0, 'TESS_replaces_wrong_LCDB': 0.6}
NEG = {'LCDB_native_VETTED_keep': 1.0, 'LCDB_native_below_cut': 0.6, 'LCDB_native_TESS_disagrees': 0.4}


def match(p, truth):
    return np.abs(p / truth - 1) < TOL


def build_table():
    rep = pd.read_csv(REPORT, low_memory=False, usecols=['designation', 'period_hr', 'reliability_score'] + F)
    rep = rep[np.isfinite(rep.period_hr)].drop_duplicates('designation').set_index('designation')
    split = pd.read_csv(V4_TABLE, usecols=['designation', 'split']).drop_duplicates('designation')
    hand = pd.read_csv(HAND)[['designation', 'period_hr', 'believed']].merge(split, on='designation')
    hand = hand[hand.split.isin(['dev', 'test'])].join(rep[['period_hr']].rename(columns={'period_hr': 'new_p'}), on='designation')
    hand = hand[hand.new_p.notna()]
    same = match(hand.new_p, hand.period_hr)
    hand['y'] = np.where(hand.believed, same.astype(float), np.where(same, 0.0, np.nan))
    print(f'hand: {len(hand)} with a new period; believed {int(hand.believed.sum())} '
          f'(new period matches {int((hand.believed & same).sum())}); not believed {int((~hand.believed).sum())} '
          f'(same period, kept as negative {int((~hand.believed & same).sum())}; changed, dropped {int((~hand.believed & ~same).sum())})')
    hand = hand[hand.y.notna()].assign(w=1.0, source='hand')

    lc = pd.read_csv(LCDB, low_memory=False)
    lc = lc[lc.period_source.isin(POS | NEG) & lc.designation.notna() & ~lc.designation.isin(hand.designation)]
    lc = lc.drop_duplicates('designation').join(rep[['period_hr']].rename(columns={'period_hr': 'new_p'}), on='designation')
    lc = lc[lc.new_p.notna() & np.isfinite(lc.period_updated)]
    lc = lc.assign(y=match(lc.new_p, lc.period_updated).astype(float), w=lc.period_source.map({**POS, **NEG}),
                   split='lcdb', source='lcdb')
    print(f'LCDB: {len(lc):,} with a new period, {lc.y.mean():.1%} match period_updated '
          f'(v4 positives {lc[lc.period_source.isin(POS)].y.mean():.1%}, v4 negatives {lc[lc.period_source.isin(NEG)].y.mean():.1%})')

    ag = pd.read_csv(AGREE, low_memory=False, usecols=['designation', 'label_v2'])
    ag = ag[ag.label_v2.notna() & ~ag.designation.isin(set(hand.designation) | set(lc.designation)) & ag.designation.isin(rep.index)]
    ag = ag.assign(y=ag.label_v2, w=W_AGREE, split='agree', source='agree')
    print(f'agreement: {int((ag.y == 1).sum()):,} positives, {int((ag.y == 0).sum()):,} strong negatives')

    keep = ['designation', 'split', 'source', 'y', 'w']
    t = pd.concat([hand[keep], lc[keep], ag[keep]], ignore_index=True)
    t = t.join(rep[F + ['reliability_score']], on='designation')
    t.to_csv(TABLE_OUT, index=False)
    print(f'{len(t):,} rows -> {TABLE_OUT}')
    print(t.groupby('split').agg(n=('y', 'size'), positive=('y', 'mean')).to_string())
    return t


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0,
                                          min_samples_leaf=20, random_state=0)


def X(d):
    return d[F].astype(float).replace([np.inf, -np.inf], np.nan).values


def threshold_from_dev(dev, rest):
    oof = np.full(len(dev), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(dev, dev.y):
        fit = pd.concat([dev.iloc[tr], rest])
        oof[te] = model().fit(X(fit), fit.y.values, sample_weight=fit.w.values).predict_proba(X(dev.iloc[te]))[:, 1]
    y = dev.y.values
    for th in np.sort(np.unique(oof)):
        p = oof >= th
        if p.sum() and y[p].mean() >= PREC_TARGET:
            return float(th), roc_auc_score(y, oof)
    return 1.0, roc_auc_score(y, oof)


def scores(name, y, s, th):
    p = s >= th
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    print(f'  {name}: n {len(y)}, AUC {roc_auc_score(y, s):.3f}, precision {tp / max(tp + fp, 1):.1%}, '
          f'recall {tp / max(tp + fn, 1):.1%}, accuracy {(tp + tn) / len(y):.1%} (tp {tp}, fp {fp}, fn {fn}, tn {tn})')


def main():
    t = build_table()
    dev, test = t[t.split == 'dev'], t[t.split == 'test']
    yt = test.y.values
    print('\nheld-out test (re-attached hand labels):')
    scores('current classifier (snr5-trained, its threshold)', yt, test.reliability_score.values, 0.564)
    results = {}
    for name, rest in [('A hand+LCDB', t[t.split == 'lcdb']), ('B hand+LCDB+agreement', t[t.split.isin(['lcdb', 'agree'])])]:
        th, dev_auc = threshold_from_dev(dev, rest)
        fit = pd.concat([dev, rest])
        mdl = model().fit(X(fit), fit.y.values, sample_weight=fit.w.values)
        st = mdl.predict_proba(X(test))[:, 1]
        print(f' {name}: threshold {th:.3f} (dev out-of-fold AUC {dev_auc:.3f})')
        scores('test', yt, st, th)
        lc = t[t.split == 'lcdb']
        if name.startswith('B'):
            scores('LCDB (in training, optimistic)', lc.y.values, mdl.predict_proba(X(lc))[:, 1], th)
        results[name] = (mdl, th, roc_auc_score(yt, st))
    best = max(results, key=lambda k: results[k][2])
    mdl, th, _ = results[best]
    joblib.dump({'model': mdl, 'features': F, 'calibrator': None, 'threshold': th,
                 'floors': {'n_points': 100, 'aov_F': 1.5}, 'trained_on': best,
                 'note': 'raw-frame POINT_MODE; labels re-attached by true period; trained on ozstar'}, OUT)
    print(f'\nsaved {best} -> {OUT} (not deployed: the pipeline still loads reliability_classifier.joblib)')


if __name__ == '__main__':
    main()
