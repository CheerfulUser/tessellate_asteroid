"""Export the v4 training data as a version-independent CSV for native training on ozstar.

joblib pickles do not cross the sklearn boundary (local 1.9.0 vs ozstar 1.8.0): loading v4 there
fails outright with "<class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module".
So the model is retrained on the cluster from a plain CSV instead of being shipped as a pickle.

The dev/test SPLIT is frozen into the exported table rather than recomputed on ozstar, so the
held-out 200 are identical in both places regardless of any difference in
sklearn.model_selection behaviour between versions. Without that, the "held out" set could
silently differ from the one the local numbers were measured on.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

F = ['aov_F', 'amp_snr', 'max_phase_gap', 'n_cycles_observed', 'n_points', 'loo_worst_chi2',
     'power_ratio', 'fap', 'n_visits', 'n_sectors', 'baseline_hr', 'n_outliers_rejected',
     'frac_points_stacked', 'phase_angle_range_deg', 'median_n_stack', 'pk_over_med',
     'pk_prom', 'pk_snr']
POS = {'TESS_precision_LCDB_harmonic': 0.6, 'LCDB_native_rescale_below_cut': 0.2,
       'TESS_replaces_wrong_LCDB_VETTED': 1.0, 'TESS_replaces_wrong_LCDB': 0.6}
NEG = {'LCDB_native_VETTED_keep': 1.0, 'LCDB_native_below_cut': 0.6,
       'LCDB_native_TESS_disagrees': 0.4}

m = pd.read_csv('comparison_data/label800_labelled.csv')
cat = pd.read_csv('population_figs/all_sector_report_v4.csv', low_memory=False)
lc = pd.read_csv('comparison_data/lcdb_updated.csv', low_memory=False)

dev, test = train_test_split(m, test_size=0.25, stratify=m['believed'], random_state=42)
dev = dev.assign(split='dev', y=dev['believed'].astype(int), w=1.0)
test = test.assign(split='test', y=test['believed'].astype(int), w=1.0)

lab = lc[lc.period_source.isin(POS | NEG) & lc.designation.notna()].copy()
lab['y'] = lab.period_source.isin(POS).astype(int)
lab['w'] = lab.period_source.map({**POS, **NEG})
lab = lab.merge(cat[['designation'] + [c for c in F if c not in lab.columns]],
                on='designation', how='left').drop_duplicates('designation')
lab = lab[~lab.designation.isin(set(m.designation))]
lab['split'] = 'lcdb'

keep = ['designation', 'split', 'y', 'w'] + F
out = pd.concat([dev[keep], test[keep], lab[keep]], ignore_index=True)
out.to_csv('comparison_data/v4_training_table.csv', index=False)
print(f'{len(out):,} rows -> comparison_data/v4_training_table.csv')
print(out.groupby('split').agg(n=('y', 'size'), positive=('y', 'mean')).to_string())
print(f'\nfeatures: {len(F)}')
print('local reference metrics to reproduce on ozstar:')
print('  held-out test: precision 90.9%, recall 92.6%, accuracy 95.5%, AUC 0.9803')
