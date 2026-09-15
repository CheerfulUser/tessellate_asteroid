"""Half-vs-half period test across the whole catalogue.

A rotation period is wrong by a factor of two more often than any other way. If the adopted
period is 2x the truth, the folded curve contains the SAME rotation sampled twice and its two
half-cycles differ only by noise. If the period is right, the two maxima belong to different
faces and differ systematically. Comparing the halves against the per-point errors makes that a
chi-square test.

The test is only meaningful when it had the power to see a difference, so it also records the
half-difference detectable at 3 sigma as a fraction of the amplitude. A null result with poor
sensitivity says nothing -- that is the difference between (683) Lanzia, where the null is
informative, and (7358) Oze, where it is not.

This CANNOT prove a period correct. Two genuinely similar maxima also give indistinguishable
halves, which is why the earlier reverts required LCDB to agree independently: of 12 candidates
with a published period, only 6 matched the halved value.

Writes data/period_test.csv for the whole catalogue.
"""
import json, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
Y34 = os.environ.get('Y34_DIR', '/Users/rridden/Documents/work/code/tess/asteroid/y3_4')
FOLDED = f'{Y34}/folded'
CAT = f'{Y34}/population_figs/all_sector_report_v5_doubled.csv'
DEST = f'{ROOT}/data/period_test.csv'


def sn(d):
    return re.sub(r'[^A-Za-z0-9]+', '_', str(d)).strip('_')


def main():
    c = pd.read_csv(CAT, low_memory=False)
    r = c[(c.is_reliable == True) & c.period_hr.notna()].copy()        # noqa: E712
    print(f'  {len(r):,} reliable objects')

    rows = []
    for i, t in enumerate(r.itertuples()):
        key = sn(t.designation)
        fp = f'{FOLDED}/{key}.json'
        if not os.path.exists(fp):
            continue
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        ph = np.asarray(d.get('phase') or [], float)
        fl = np.asarray(d.get('flux') or [], float)
        er = np.asarray(d.get('err') or [], float)
        if len(ph) < 24 or len(er) != len(fl):
            continue
        n = len(ph) // 2
        a, b = fl[:n], fl[n:2 * n]
        ea, eb = er[:n], er[n:2 * n]
        ok = (ea > 0) & (eb > 0) & np.isfinite(a) & np.isfinite(b)
        dof = int(ok.sum())
        if dof < 8:
            continue
        sig = np.hypot(ea[ok], eb[ok])
        chi2 = float(np.sum(((a[ok] - b[ok]) / sig) ** 2))
        p = float(stats.chi2.sf(chi2, dof))
        amp = float(np.nanmax(fl) - np.nanmin(fl))
        detect = 3 * float(np.median(sig)) / np.sqrt(dof)
        rows.append(dict(
            designation=t.designation, key=key, period_hr=float(t.period_hr),
            lcdb=None if pd.isna(t.published_rot_per_hr) else float(t.published_rot_per_hr),
            chi2=chi2, dof=dof, p=p, red=chi2 / dof, amp=amp,
            sens=detect / amp if amp > 0 else np.nan))
        if (i + 1) % 4000 == 0:
            print(f'    {i+1:,}/{len(r):,}', flush=True)

    d = pd.DataFrame(rows)
    # a null only counts as evidence when a real difference would have been visible
    d['informative'] = d.sens < 0.10
    d['verdict'] = np.where(~d.informative, 'undetermined',
                   np.where(d.p > 0.05, 'halves = noise (period may be 2x too long)',
                   np.where(d.p < 0.001, 'halves differ (period supported)', 'ambiguous')))
    d.to_csv(DEST, index=False)
    print(f'\n  wrote {os.path.relpath(DEST, ROOT)}: {len(d):,} tested')
    print(d.verdict.value_counts().to_string().replace('\n', '\n    '))

    sus = d[(d.verdict.str.startswith('halves = noise'))]
    print(f'\n  suspect (informative null): {len(sus):,} = {len(sus)/len(d):.1%} of the catalogue')
    withl = sus[sus.lcdb.notna()]
    if len(withl):
        half = (withl.period_hr / 2 - withl.lcdb).abs() / withl.lcdb < 0.03
        same = (withl.period_hr - withl.lcdb).abs() / withl.lcdb < 0.03
        print(f'    of the {len(withl):,} with an LCDB period:')
        print(f'      LCDB matches HALF ours (revert candidates): {int(half.sum()):,}')
        print(f'      LCDB matches ours (test is a false alarm):  {int(same.sum()):,}')
        print(f'      neither:                                    {int((~half & ~same).sum()):,}')
        print(f'    -> false-alarm rate where LCDB can arbitrate: '
              f'{same.sum()/(same.sum()+half.sum()):.0%}')


if __name__ == '__main__':
    main()
