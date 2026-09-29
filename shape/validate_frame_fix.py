"""Before/after check for the batch_shapes geometry fixes (ecliptic frame, relative lcs flag,
light-time-corrected epochs, caliper elongation) on objects already in the published catalogue.

For each object: rerun process() with the fixed code into a separate directory, then compare
against the published solution in shapes/out/:
  - aspect actually used: the published fits paired an ecliptic lon/lat pole with EQUATORIAL
    vectors, so their true aspect is measured in the equatorial frame; the fixed fits should sit
    at 90 deg in the ecliptic frame by construction
  - folded-curve model rms (the published model_rms vs the rerun's)
  - elongation: published bounding-box ratio, the same mesh by caliper, and the rerun's caliper

Run on ozstar from /fred/oz335/rridden/asteroids (needs clean_lightcurves/ and shapes/out/).
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import batch_shapes as bs  # noqa: E402

OUT_OLD = 'shapes/out'
OUT_NEW = os.environ.get('VALIDATE_OUT', 'shape_frame_fix_validation/out')
WORK = os.environ.get('VALIDATE_WORK', 'shape_frame_fix_validation/work')
KEYS = os.environ.get('VALIDATE_KEYS', '3550_Link,75_Eurydike,3570_Wuyeesun').split(',')
N_RANDOM = int(os.environ.get('VALIDATE_N_RANDOM', 7))


def aspect_deg(ev, lam, bet):
    en = (ev / np.linalg.norm(ev, axis=1, keepdims=True)).mean(0)
    en /= np.linalg.norm(en)
    l, b = np.radians(lam), np.radians(bet)
    pole = np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])
    return float(np.degrees(np.arccos(np.clip(pole @ en, -1, 1))))


def main():
    os.makedirs(OUT_NEW, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    rng = np.random.default_rng(3)
    pool = sorted(f[:-5] for f in os.listdir(OUT_OLD) if f.endswith('.json') and not f.startswith('_'))
    keys = KEYS + list(rng.choice([k for k in pool if k not in KEYS], N_RANDOM, replace=False))
    rows = []
    for key in keys:
        old = json.load(open(f'{OUT_OLD}/{key}.json'))
        s = old['solution']
        lc = f"clean_lightcurves/{bs.sn(s['designation'])}_clean_lc.csv"
        new = bs.process(s['designation'], s['adopted_period_hr'], lc, WORK, OUT_NEW)
        df = bs.prepare(pd.read_csv(lc))
        _, ev_ecl = bs.build_geometry(df)
        ev_eq = ev_ecl @ bs._ICRS_TO_ECL        # back to ICRS: the frame the old fit really used
        Vold = np.array(old['mesh']['v']).reshape(-1, 3)
        wmax, wmin = bs.caliper_widths(Vold)
        row = dict(key=key, period_hr=s['adopted_period_hr'],
                   old_aspect_true=aspect_deg(ev_eq, s['representative_lambda_deg'],
                                              s['representative_beta_deg']),
                   old_rms=s.get('model_rms'), old_ratio_box=s.get('equatorial_ratio'),
                   old_ratio_caliper=wmax / wmin, new_ok=new.get('ok'), new_error=new.get('error'))
        if new.get('ok'):
            row.update(new_aspect=aspect_deg(ev_ecl, new['representative_lambda_deg'],
                                             new['representative_beta_deg']),
                       new_rms=new.get('model_rms'), new_ratio_caliper=new.get('equatorial_ratio'),
                       new_period_hr=new.get('model_period_hr'))
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(f'{OUT_NEW}/../frame_fix_comparison.csv', index=False)
    print('\n' + t.round(3).to_string(index=False))
    ok = t[t.new_ok == True]  # noqa: E712
    if len(ok):
        print(f"\nold fits: aspect actually used {t.old_aspect_true.min():.1f}-{t.old_aspect_true.max():.1f} deg "
              f"(median |90-aspect| {np.median(np.abs(90 - t.old_aspect_true)):.1f})")
        print(f"new fits: aspect {ok.new_aspect.min():.2f}-{ok.new_aspect.max():.2f} deg")
        print(f"model rms new/old: median {np.median(ok.new_rms / ok.old_rms):.3f} "
              f"(range {np.min(ok.new_rms / ok.old_rms):.3f}-{np.max(ok.new_rms / ok.old_rms):.3f})")


if __name__ == '__main__':
    main()
