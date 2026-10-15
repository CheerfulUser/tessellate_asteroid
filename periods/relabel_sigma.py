"""Relabel the synthetic training set by each object's own period uncertainty instead of a flat tolerance.
sigma_P / P = P sqrt(6 / N) (sigma / A) / (pi T)   (Montgomery & O'Donoghue 1999, sinusoid least squares), with
A = half the fitted peak-to-peak (amp_snr = ptp / median error, so sigma / A = 2 / amp_snr), the noise scaled to the
residual scatter by sqrt(max(chi2_red_fourier, 1)), N points, T baseline (days), P the adopted period (days).
  correct_3sig   |P - P_inj| < 3 sigma_P
  harmonic_3sig  |P - 2 P_inj| or |P - P_inj / 2| < 3 sigma_P
  signal_3sig    correct or harmonic: a real signal at the right period up to a factor of two, within its errors
Writes blank_injection/synthetic_training_set_all_sigma.csv. Runs on ozstar."""
import numpy as np
import pandas as pd

B = "/fred/oz335/rridden/asteroids_store/blank_injection"
d = pd.read_csv(f"{B}/synthetic_training_set_all.csv", low_memory=False)
P = d.period_hr / 24
T = d.baseline_hr / 24
noise = np.sqrt(np.clip(d.chi2_red_fourier.astype(float), 1, None))
d["sigma_P_frac"] = P * np.sqrt(6 / d.n_points) * (2 / d.amp_snr) * noise / (np.pi * T)
rot = (d.kind == "rotator") & d.period_hr.notna() & d.error.isna()
for name, f in (("c1", 1.0), ("c2", 2.0), ("c05", 0.5)):
    d[name] = rot & ((d.period_hr / (d.period_inj_hr * f) - 1).abs() < 3 * d.sigma_P_frac)
d["correct_3sig"] = d.c1
d["harmonic_3sig"] = (d.c2 | d.c05) & ~d.c1
d["signal_3sig"] = d.correct_3sig | d.harmonic_3sig
d = d.drop(columns=["c1", "c2", "c05"])
d.to_csv(f"{B}/synthetic_training_set_all_sigma.csv", index=False)
r = d[rot]
two = r.recovered_2pct.astype(str).str.lower().eq("true")
print(f"{len(r):,} rotators: within 2 % {two.mean():.1%}; correct within 3 sigma {r.correct_3sig.mean():.1%}; "
      f"harmonic within 3 sigma {r.harmonic_3sig.mean():.1%}; signal label {r.signal_3sig.mean():.1%}")
print(f"  within 2 % but outside 3 sigma: {(two & ~r.correct_3sig).mean():.1%}; outside 2 % but within 3 sigma: {(~two & r.correct_3sig).mean():.1%}")
print("sigma_P/P quantiles 10/50/90:", r.sigma_P_frac.quantile([.1, .5, .9]).map(lambda x: f"{x:.1e}").tolist())
print("signal label by per-frame S/N:")
print(r.groupby(pd.cut(r.s_frame, [0.1, 0.5, 1, 2, 5, 20, 50]), observed=True)[["correct_3sig", "harmonic_3sig", "signal_3sig"]].mean().round(3).to_string())

# reachable_shape: what the classifier should answer when identification (harmonic, alias) and precision are left to
# the shape stage -- the pipeline period lies within the combined fit's period-scan range (+-8.829 %,
# combined_tess_atlas.py) of the injected period or of a harmonic candidate the shape stage tests (x2, 1/2, 3/2, 2/3,
# 3, 1/3), so the right solution is reachable from it.
SCAN = 0.08829
ratio = d.period_hr / d.period_inj_hr
d["reachable_shape"] = rot & np.any([(ratio / q - 1).abs() < SCAN for q in (1, 2, 0.5, 1.5, 2 / 3, 3, 1 / 3)], axis=0)
d.to_csv(f"{B}/synthetic_training_set_all_sigma.csv", index=False)
r = d[rot]
print(f"reachable by the shape stage: {r.reachable_shape.mean():.1%} of rotators")
print(r.groupby(pd.cut(r.s_frame, [0.1, 0.5, 1, 2, 5, 20, 50]), observed=True).reachable_shape.mean().round(3).to_string())
