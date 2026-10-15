"""For each user-labelled real object, the fraction of its K nearest synthetic injections (standardised log pk_prom,
log amp_snr, fold_vn, log aov_F, log n_points) whose adopted period was correct (within 2 %): an empirical estimate of
how often an object that looks like this has the right period. Summarised by the user's label and the synthetic
classifier's verdict (F: good & passed, G: good & rejected, H: bad & passed, I: bad & rejected). Runs on ozstar."""
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

S = "/fred/oz335/rridden/asteroids_store"
K = 200
COLS = ["pk_prom", "amp_snr", "fold_vn", "aov_F", "n_points"]
LOG = {"pk_prom", "amp_snr", "aov_F", "n_points"}


def space(df):
    m = np.column_stack([np.log10(np.clip(df[c].astype(float), 1e-6, None)) if c in LOG else df[c].astype(float) for c in COLS])
    return m


def main():
    syn = pd.read_csv(f"{S}/blank_injection/synthetic_training_set_all.csv", low_memory=False)
    syn = syn[syn.error.isna() & syn.period_hr.notna()].dropna(subset=COLS)
    ys = syn.recovered_2pct.astype(str).str.lower().eq("true").values
    Ms = space(syn); mu, sd = Ms.mean(0), Ms.std(0)
    nn = NearestNeighbors(n_neighbors=K).fit((Ms - mu) / sd)
    b = joblib.load("/fred/oz335/TESSdata/asteroid_store/reliability_classifier_synthetic_recovered_2pct.joblib")
    lab = pd.concat([pd.read_csv(f"{S}/label_batches/labels_final.csv")[["designation", "label_final"]]]
                    + [pd.read_csv(f"{S}/label_batches/labels_{n}.csv").assign(label_final=lambda x: np.where(x.good, "good", "bad"))[["designation", "label_final"]]
                       for n in ("faint2", "faint3")]).drop_duplicates("designation")
    h = pd.read_csv(f"{S}/bin_range_test.csv", low_memory=False).drop_duplicates("designation").merge(lab, on="designation")
    h = h[h.period_hr.notna()].dropna(subset=COLS).reset_index(drop=True)
    h["syn_pass"] = b["model"].predict_proba(h[b["features"]].astype(float).replace([np.inf, -np.inf], np.nan).values)[:, 1] >= b["threshold"]
    _, idx = nn.kneighbors((space(h) - mu) / sd)
    h["neighbour_correct"] = ys[idx].mean(1)
    good = h.label_final == "good"
    h["group"] = np.select([good & h.syn_pass, good & ~h.syn_pass, ~good & h.syn_pass, ~good & ~h.syn_pass],
                           ["F user good, synthetic passes", "G user good, synthetic rejects", "H user bad, synthetic passes",
                            "I user bad, synthetic rejects"])
    print(f"fraction of the {K} nearest synthetic injections with the correct period (quantiles 10/25/50/75/90):")
    print(h.groupby("group").neighbour_correct.describe(percentiles=[.1, .25, .5, .75, .9])[["count", "10%", "25%", "50%", "75%", "90%"]].round(2).to_string())
    g = h[h.group.str.startswith("G")]
    print(f"\nG objects whose look-alike injections are correct >= 50 % of the time: {(g.neighbour_correct >= 0.5).mean():.1%}; "
          f"< 20 %: {(g.neighbour_correct < 0.2).mean():.1%}")
    h[["designation", "label_final", "group", "neighbour_correct"] + COLS].to_csv(f"{S}/synthetic_neighbours.csv", index=False)


if __name__ == "__main__":
    main()
