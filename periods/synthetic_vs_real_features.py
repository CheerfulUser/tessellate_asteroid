"""Are real correct-period objects less self-consistent across visits than the synthetic positives? Compares feature
distributions, at matched amp_snr, between the user's good real objects (bin_range_test.csv + labels), the synthetic
positives (phase-coherent recovery) and the synthetic negatives; and, for the user-good objects the synthetic model
rejects, where each feature sits within the synthetic positives (percentile; ~50 = typical, ~100 = beyond them).
Runs on ozstar.
"""
import joblib
import numpy as np
import pandas as pd

S = "/fred/oz335/rridden/asteroids_store"
FEATS = ["loo_worst_chi2", "chi2_red_fourier", "fold_vn", "aov_F", "n_outliers_rejected", "max_phase_gap", "power_ratio",
         "pk_snr", "n_visits", "baseline_hr", "n_cycles_observed"]
BINS = [0, 2, 4, 8, 16, 1e9]


def main():
    bundle = joblib.load("/fred/oz335/TESSdata/asteroid_store/reliability_classifier_synthetic.joblib")
    syn = pd.read_csv(f"{S}/blank_injection/synthetic_training_set_100k.csv", low_memory=False)
    syn = syn[syn.get("error", pd.Series(np.nan, index=syn.index)).isna() & syn.period_hr.notna()]
    syn["group"] = np.where(syn.recovered.astype(str).str.lower() == "true", "synthetic correct", "synthetic wrong")
    lab = pd.concat([pd.read_csv(f"{S}/label_batches/labels_final.csv")[["designation", "label_final"]],
                     pd.read_csv(f"{S}/label_batches/labels_faint2.csv").assign(label_final=lambda x: np.where(x.good, "good", "bad"))[["designation", "label_final"]]]).drop_duplicates("designation")
    real = pd.read_csv(f"{S}/bin_range_test.csv", low_memory=False).drop_duplicates("designation").merge(lab, on="designation")
    real = real[(real.label_final == "good") & real.period_hr.notna()].copy()
    real["group"] = "user good (real)"
    Xr = real[bundle["features"]].astype(float).replace([np.inf, -np.inf], np.nan).values
    real["syn_score"] = bundle["model"].predict_proba(Xr)[:, 1]
    real["syn_pass"] = real.syn_score >= bundle["threshold"]
    both = pd.concat([syn[["group", "amp_snr"] + FEATS], real[["group", "amp_snr"] + FEATS]], ignore_index=True)
    both["amp_snr_bin"] = pd.cut(both.amp_snr, BINS)
    print(f"user-good real objects: {len(real)}, synthetic model passes {real.syn_pass.mean():.1%}")
    print("\ncounts per amp_snr bin:")
    print(both.groupby(["amp_snr_bin", "group"], observed=True).size().unstack().to_string())
    for f in FEATS:
        print(f"\nmedian {f} by amp_snr bin:")
        print(both.groupby(["amp_snr_bin", "group"], observed=True)[f].median().unstack().round(3).to_string())
    # user-good objects the synthetic model rejects: percentile of each feature within the synthetic positives of
    # the same amp_snr bin
    pos = syn[syn.group == "synthetic correct"].assign(amp_snr_bin=lambda x: pd.cut(x.amp_snr, BINS))
    rej = real[~real.syn_pass].assign(amp_snr_bin=lambda x: pd.cut(x.amp_snr, BINS))
    acc = real[real.syn_pass].assign(amp_snr_bin=lambda x: pd.cut(x.amp_snr, BINS))
    print(f"\nfeature percentile within synthetic positives (same amp_snr bin), median over user-good objects:")
    rows = []
    for f in FEATS:
        out = {}
        for name, grp in (("rejected by synthetic model", rej), ("passed by synthetic model", acc)):
            pc = []
            for b, g in grp.groupby("amp_snr_bin", observed=True):
                ref = pos[pos.amp_snr_bin == b][f].dropna().values
                if len(ref) > 50:
                    pc += list(100 * np.searchsorted(np.sort(ref), g[f].dropna().values) / len(ref))
            out[name] = np.median(pc) if pc else np.nan
        rows.append(dict(feature=f, **out))
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
