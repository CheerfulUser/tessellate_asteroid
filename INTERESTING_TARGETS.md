# Targets flagged for possible REAL changes in lightcurve behaviour

Not errors. Flagged during vetting of the automatic LCDB replacements because our period differs
from the literature by a large, non-harmonic factor while our detection is statistically strong
— which may indicate the object's rotational behaviour genuinely differs from when it was
published, rather than either value being wrong.

## (1418) Fayeta

| | |
|---|---|
| LCDB period | 63.641 h (U=2) |
| TESS period | 218.737 h |
| ratio | 3.44 (not a harmonic) |
| `aov_F` | 364.2 |
| points | 2,521 |
| reliability score | 0.686 |
| well-covered cycles | **1** |
| baseline | 464 h = **2.1 rotations** at our period |
| PDM | theta_ours 0.295, theta_lcdb 0.993, null 5th pct 0.565 |

Our period is strongly detected and the LCDB value folds no better than random. But the baseline
covers only 2.1 rotations, so a slow trend is hard to separate from a genuine long rotation.

## (12269) 1990 QR

| | |
|---|---|
| LCDB period | 76.000 h (U=1) |
| TESS period | 207.862 h |
| ratio | 2.74 (not a harmonic) |
| `aov_F` | 158.8 |
| points | 2,984 |
| reliability score | **0.111** (`is_reliable` = False) |
| well-covered cycles | 2 |
| PDM | theta_ours 0.544, theta_lcdb 0.814, null 5th pct 0.618 |

User's read: looks bad in BOTH the LCDB and TESSELLATE values. Our own pipeline already rejects
this period (score 0.111), so the automatic rule should never have overwritten the literature
here — see the bug note below.

## Why both may be real rather than artefacts

Both show a much LONGER TESS period than published, at ~2.7-3.4x, with few covered cycles.
Candidate physical explanations worth checking rather than assuming error:
- **Tumbling** (non-principal-axis rotation) produces more than one period, and which one a
  dataset recovers depends on its sampling and baseline. Raised earlier in this project as a
  possibility for objects with discrepant periods.
- A genuinely slow rotator whose published short period came from a single night that could not
  close the cycle count.

Neither is settled by the data in hand. Both need either a longer baseline or a targeted
re-reduction before a claim is made.

## Bug this exposed in `replace_wrong_lcdb.py`

The automatic replacement gated on `n_points`, `aov_F` and `n_phase_coverings` but **not on our
own `reliability_score`**. 62.5% of the 784 automatic replacements had `is_reliable` = False,
and 48.9% scored below 0.2 — the rule was overwriting published periods with periods our own
pipeline rejects. 1990 QR is the clean example (score 0.111).

The PDM null test cannot catch this on its own: it asks only whether the LCDB period folds our
data better than random, and on a junk lightcurve nothing folds well, so the literature value
fails by default. The test has no way to return "neither".

484 replacements scoring below 0.45 have been reverted to the native LCDB value and flagged
`LCDB_native_BOTH_UNRELIABLE`. Fayeta survived that cut at 0.686, so the score threshold alone
is not sufficient — the long-period population (60 of the remaining 300 have P > 48 h, median
**2** covered cycles against 17 for short-period replacements) is the residual risk.
