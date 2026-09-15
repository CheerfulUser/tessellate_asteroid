"""Apply the quality gate to the 6,717 harmonic rescalings.

These were never gated on our data quality. The rule only required our period to land near a
1x/2x multiple of the LCDB value -- but on a poor lightcurve that can happen by coincidence, and
the rescaled entry then inherits OUR NOISE instead of our precision, which is the opposite of
what the rescale is for. All six sampled examples were rejected on sight, three of them with
ZERO well-covered cycles and reliability scores of 0.04-0.12.

Same floor as the replacement regime, which scored 98.1% purity against the hand-vetted sample:
    aov_F >= 15  AND  n_phase_coverings >= 4
Rows failing it revert to the native LCDB period. Nothing is lost -- `tess_period_hr` and
`harmonic_n` are retained, so a row can be re-promoted if its reduction improves.

Note the asymmetry with the replacement case. A failed REPLACEMENT means we cannot show the
literature is wrong, so the published value stands on its own merit. A failed RESCALE means only
that we cannot improve the published value's precision; the LCDB period itself is unaffected, so
reverting is strictly a loss of precision rather than a change of answer.
"""
import numpy as np, pandas as pd

FLOOR_A, FLOOR_C = 15.0, 4

m = pd.read_csv('comparison_data/lcdb_updated.csv', low_memory=False)
h = m.period_source == 'TESS_precision_LCDB_harmonic'
print(f'{int(h.sum()):,} harmonic rescalings to gate')
print(f'  of which pure precision (n=1): {int((h & (m.harmonic_n == 1)).sum()):,}')
print(f'  cycle-count changes (n=0.5/2): {int((h & (m.harmonic_n != 1)).sum()):,}\n')

print('current quality of the rescaled rows:')
for lo, hi in [(0, 3), (3, 6), (6, 15), (15, 40), (40, np.inf)]:
    g = m[h & (m.aov_F >= lo) & (m.aov_F < hi)]
    print(f'  aov_F {lo:>3}-{str(hi) if hi != np.inf else "inf":<4} {len(g):6,d}')
print(f'  with 0 well-covered cycles: {int((h & (m.n_phase_coverings < 1)).sum()):,}')
print(f'  is_reliable = False       : {int((h & (~m.is_reliable.fillna(False).astype(bool))).sum()):,}')

passes = h & (m.aov_F >= FLOOR_A) & (m.n_phase_coverings >= FLOOR_C)
fails = h & ~passes
print(f'\ngate: aov_F >= {FLOOR_A} AND covered cycles >= {FLOOR_C}')
print(f'  pass {int(passes.sum()):,} ({passes.sum()/h.sum():.1%})   fail {int(fails.sum()):,} '
      f'({fails.sum()/h.sum():.1%})')

# failures revert to the published value; the TESS columns stay for later re-promotion
m.loc[fails, 'period_updated'] = m.loc[fails, 'Period']
m.loc[fails, 'period_source'] = 'LCDB_native_rescale_below_cut'

f = m[fails]
print(f'\nreverted rows: median aov_F {f.aov_F.median():.1f}, '
      f'covered cycles {f.n_phase_coverings.median():.0f}, score {f.reliability_score.median():.2f}')
p = m[passes]
print(f'retained rows: median aov_F {p.aov_F.median():.1f}, '
      f'covered cycles {p.n_phase_coverings.median():.0f}, score {p.reliability_score.median():.2f}')
print(f'  of the retained, cycle-count changes: {int((p.harmonic_n != 1).sum()):,}')

print('\nfinal table:')
for k, v in m.period_source.value_counts().items():
    print(f'  {k:36s} {v:7,d} ({v/len(m):5.1%})')
tess = m.period_source.str.startswith('TESS_')
print(f'\n  entries carrying a TESS-derived period: {int(tess.sum()):,} of {len(m):,} '
      f'({tess.mean():.1%})')
m.to_csv('comparison_data/lcdb_updated.csv', index=False)
print('\nwrote comparison_data/lcdb_updated.csv')
