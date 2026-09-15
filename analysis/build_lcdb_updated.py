"""Extend the LCDB summary table with a TESS-precision period column.

For every LCDB entry:
    period_updated = n * P_tess    where TESS observed it, and n is the simple harmonic
                                   {1/2, 1, 2} that brings our period closest to the LCDB value
    period_updated = Period        (the native LCDB value) where TESS did not observe it

This keeps LCDB's harmonic/cycle-count judgement -- which is what sparse ground-based photometry
is actually good at establishing, via amplitude and shape -- and replaces only the NUMERICAL
PRECISION, which is what a 437-day continuous baseline is good at. (75) Koshiishi is the worked
case: LCDB 12.0068 h, ours 5.98988 h. Their value is 0.23% off exactly twice ours, which over
876 cycles accumulates to two full cycles of drift and decoheres the fold completely. The
updated entry becomes 2 x 5.98988 = 11.97976 h: their factor of 2, our precision (~25 s).

3P/P3 are excluded: three-peaked lightcurves need three-fold shape symmetry, which has no
physical basis in the population.

The native Period column is preserved untouched; nothing is overwritten.
"""
import re
import numpy as np, pandas as pd

LCDB = 'comparison_data/lcdb_lc_summary.csv'
OURS = 'population_figs/all_sector_report_v4.csv'
OUT = 'comparison_data/lcdb_updated.csv'
RATIOS = [0.5, 1.0, 2.0]


def tol_of(a):
    """How close counts as the same harmonic, scaled by OUR data quality -- the measured width
    of the 2x relation runs from 2% at aov_F>50 to 19% below aov_F 3."""
    if not np.isfinite(a):
        return 0.15
    if a >= 50:
        return 0.06
    if a >= 10:
        return 0.15
    if a >= 3:
        return 0.25
    return 0.35


cols = open(LCDB).readlines()[16].strip().split(',')
lc = pd.read_csv(LCDB, skiprows=22, header=None, names=cols, dtype=str, on_bad_lines='skip')
lc['Period'] = pd.to_numeric(lc['Period'], errors='coerce')
lc['Number'] = pd.to_numeric(lc['Number'], errors='coerce')
lc['U_clean'] = pd.to_numeric(lc['U'].astype(str).str.extract(r'(\d)')[0], errors='coerce')
lc = lc[lc['Period'].notna() & (lc['Period'] > 0)].copy()
print(f'LCDB entries with a period: {len(lc):,}')

ours = pd.read_csv(OURS, low_memory=False)
ours = ours[ours.period_hr.notna() & (ours.period_hr > 0)].copy()
num = lambda d: (lambda m: int(m.group(1)) if m else np.nan)(re.match(r'^\((\d+)\)', str(d).strip()))
ours['Number'] = ours['designation'].apply(num)
o = (ours[ours.Number.notna()]
     .sort_values('reliability_score', ascending=False)
     .drop_duplicates('Number')[['Number', 'designation', 'period_hr', 'aov_F', 'n_points',
                                 'reliability_score', 'is_reliable', 'n_phase_coverings']])
print(f'TESSELLATE objects with an MPC number and a period: {len(o):,}')

m = lc.merge(o, on='Number', how='left', suffixes=('', '_tess'))
has = m.period_hr.notna()
print(f'  LCDB entries also observed by TESS: {int(has.sum()):,}')

P, L, A = m.period_hr.values, m.Period.values, m.aov_F.values
best_n = np.full(len(m), np.nan); best_off = np.full(len(m), np.inf)
for n in RATIOS:
    off = np.abs(n * P / L - 1)
    take = has.values & (off < best_off)
    best_n[take] = n
    best_off[take] = off[take]
tol = np.array([tol_of(a) for a in A])
matched = has.values & (best_off < tol)

m['tess_period_hr'] = np.where(has, P, np.nan)
m['harmonic_n'] = np.where(matched, best_n, np.nan)
m['harmonic_offset'] = np.where(has, best_off, np.nan)
m['period_updated'] = np.where(matched, best_n * P, m.Period.values)
m['period_source'] = np.where(matched, 'TESS_precision_LCDB_harmonic',
                              np.where(has, 'LCDB_native_TESS_disagrees', 'LCDB_native'))

print('\nperiod_updated provenance:')
vc = m.period_source.value_counts()
for k, v in vc.items():
    print(f'  {k:32s} {v:7,d} ({v/len(m):5.1%})')

u = m[m.period_source == 'TESS_precision_LCDB_harmonic']
print(f'\n{len(u):,} entries updated to TESS precision:')
print(u.harmonic_n.value_counts().sort_index().to_string())
frac = (u.period_updated / u.Period - 1).abs()
print(f'  median shift from the native LCDB value: {frac.median():.3%}')
print(f'  entries where the harmonic is 1 (pure precision gain): {int((u.harmonic_n==1).sum()):,}')
print(f'  entries where the cycle count changes  : {int((u.harmonic_n!=1).sum()):,}')

keep = ['Number', 'Name', 'Period', 'U', 'U_clean', 'Amp', 'Diameter', 'Albedo',
        'period_updated', 'period_source', 'tess_period_hr', 'harmonic_n', 'harmonic_offset',
        'designation', 'aov_F', 'n_points', 'reliability_score', 'is_reliable',
        'n_phase_coverings']
keep = [k for k in keep if k in m.columns]
m[keep].to_csv(OUT, index=False)
print(f'\nwrote {OUT}  ({len(m):,} rows, {len(keep)} columns)')
