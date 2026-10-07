"""Reliability labels from cross-sector period agreement, without hand vetting.

Objects seen in two or more sectors have an independent period from each sector (the per-sector store
run, ASTEROID_STORE_SPLIT=sector). Each pair of sector periods either agrees (ratio within --tol of 1),
is a factor-2 alias of each other (the pipeline adopts 2x the periodogram peak by default, so one
sector can land on the other alias), or disagrees. Per object:
  label 1   every pair agrees
  label 0   some pair disagrees beyond the alias relation
  no label  the only disagreement is an alias, or fewer than two sector periods
The label is about the pooled period only where that period matches the agreed sector period
(pooled_matches); the training label is weak_label = label if pooled_matches else 0 for disagreements.

Validated against the 800 hand-vetted (dev/test) labels of the v4 training table only, using objects
whose pooled period now is the period they were labelled at (old report, within --tol). The LCDB
labels are not ground truth -- LCDB itself has wrong periods (172 already corrected in the TESS-updated
LCDB catalogue) -- so they are compared, not scored against: every LCDB-labelled object whose weak
label contradicts its LCDB label is written to lcdb_conflicts.csv for review.

    python3 agreement_labels.py [--tol 0.02]
"""
import argparse

import numpy as np
import pandas as pd

SECTOR_REPORT = '/fred/oz335/rridden/asteroids_store_sector/all_sector_asteroid_report.csv'
POOLED_REPORT = '/fred/oz335/rridden/asteroids_store/all_sector_asteroid_report.csv'
OLD_REPORT = '/fred/oz335/rridden/asteroids/all_sector_asteroid_report.csv'
TRAINING = '/fred/oz335/rridden/asteroids/v4_training_table.csv'
OUT = '/fred/oz335/rridden/asteroids_store_sector/agreement_labels.csv'
CONFLICTS = '/fred/oz335/rridden/asteroids_store_sector/lcdb_conflicts.csv'


def relation(r, tol):
    """'agree', 'alias' (x2 or x0.5) or 'disagree' for a period ratio r."""
    if abs(r - 1) < tol:
        return 'agree'
    if abs(r - 2) < 2 * tol or abs(r - 0.5) < 0.5 * tol:
        return 'alias'
    return 'disagree'


def object_labels(sector, tol):
    rows = []
    for d, g in sector[np.isfinite(sector.period_hr)].groupby('designation'):
        p = g.period_hr.to_numpy()
        if len(p) < 2:
            continue
        rel = [relation(p[i] / p[j], tol) for i in range(len(p)) for j in range(i + 1, len(p))]
        if all(x == 'agree' for x in rel):
            label = 1
        elif any(x == 'disagree' for x in rel):
            label = 0
        else:
            label = np.nan
        rows.append(dict(designation=d, n_sector_periods=len(p), n_pairs=len(rel),
                         n_agree=rel.count('agree'), n_alias=rel.count('alias'), n_disagree=rel.count('disagree'),
                         label=label, agreed_period_hr=float(np.median(p)) if label == 1 else np.nan,
                         sectors=','.join(map(str, sorted(g.split_sector.astype(int))))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tol', type=float, default=0.02)
    a = ap.parse_args()
    sector = pd.read_csv(SECTOR_REPORT, low_memory=False)
    pooled = pd.read_csv(POOLED_REPORT, low_memory=False)[['designation', 'period_hr', 'is_reliable', 'reliability_score']]
    lab = object_labels(sector, a.tol).merge(pooled.rename(columns={'period_hr': 'pooled_period_hr'}),
                                             on='designation', how='left')
    lab['pooled_matches'] = np.abs(lab.pooled_period_hr / lab.agreed_period_hr - 1) < a.tol
    lab['weak_label'] = np.where(lab.label == 1, lab.pooled_matches.astype(float), lab.label)
    lab.to_csv(OUT, index=False)

    n = len(lab)
    print(f'{n:,} objects with 2+ sector periods (tol {a.tol:.0%}): label 1 {int((lab.label == 1).sum()):,}, '
          f'label 0 {int((lab.label == 0).sum()):,}, alias-only {int(lab.label.isna().sum()):,}')
    print(f'label 1 whose pooled period matches: {int(lab.pooled_matches.sum()):,}; weak_label 1 '
          f'{int((lab.weak_label == 1).sum()):,}, 0 {int((lab.weak_label == 0).sum()):,}')
    print('current classifier (is_reliable) vs weak label:')
    print(pd.crosstab(lab.weak_label, lab.is_reliable.astype('boolean'), dropna=False).to_string())

    # validation against the labels the classifier was trained on, at the periods they were labelled at
    train = pd.read_csv(TRAINING, low_memory=False)[['designation', 'split', 'y']]
    old = pd.read_csv(OLD_REPORT, low_memory=False)[['designation', 'period_hr']].rename(columns={'period_hr': 'old_period_hr'})
    v = train.merge(old, on='designation').merge(lab, on='designation')
    v = v[np.abs(v.pooled_period_hr / v.old_period_hr - 1) < a.tol]
    hand = v[(v.split != 'lcdb') & v.weak_label.notna()]
    if len(hand):
        tp = int(((hand.weak_label == 1) & (hand.y == 1)).sum()); fp = int(((hand.weak_label == 1) & (hand.y == 0)).sum())
        fn = int(((hand.weak_label == 0) & (hand.y == 1)).sum()); tn = int(((hand.weak_label == 0) & (hand.y == 0)).sum())
        print(f'hand-vetted (dev+test): {len(hand)} objects; weak label vs hand label: precision {tp / max(tp + fp, 1):.1%}, '
              f'recall {tp / max(tp + fn, 1):.1%}, agreement {(tp + tn) / len(hand):.1%} (tp {tp}, fp {fp}, fn {fn}, tn {tn})')
    else:
        print('hand-vetted (dev+test): no overlap')
    lcdb = v[(v.split == 'lcdb') & v.weak_label.notna()]
    conflicts = lcdb[lcdb.weak_label != lcdb.y]
    conflicts[['designation', 'y', 'weak_label', 'old_period_hr', 'pooled_period_hr', 'agreed_period_hr', 'sectors',
               'n_agree', 'n_alias', 'n_disagree', 'reliability_score']].to_csv(CONFLICTS, index=False)
    print(f'LCDB-labelled: {len(lcdb)} with a weak label; {len(conflicts)} conflicts '
          f'({int(((conflicts.y == 0) & (conflicts.weak_label == 1)).sum())} LCDB-unreliable but sectors agree, '
          f'{int(((conflicts.y == 1) & (conflicts.weak_label == 0)).sum())} LCDB-reliable but sectors disagree) -> {CONFLICTS}')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
