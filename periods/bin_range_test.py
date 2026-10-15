"""Test the binned-fold features bin_range_sig (range significance), fold_vn and fold_jag (smoothness; see
sector_report_pipeline.plot_asteroid) on every
user-labelled object: labels_final.csv (596, round 1 + relabels) and labels_faint2.csv (500 faint, S/N < 1).
Recomputes each object's summary row with the pipeline itself (plots off), writes asteroids_store/bin_range_test.csv,
then prints AUCs (bin_range_sig, aov_F, amp_snr; all labels and faint only) and classifier CV on the user labels
with the pipeline features alone and with each added (AUC, and recall at 90 % precision).
    ASTEROID_USE_STORE=1 python3 -u bin_range_test.py
"""
import multiprocessing as mp
import os
import numpy as np, pandas as pd
import sector_report_pipeline as p
from tessellate import asteroid_store as st

S = "/fred/oz335/rridden/asteroids_store"
L = f"{S}/label_batches"


def one(d):
    try:
        phot = st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS)
        track = p.normalise(p.frames_to_points(p._store_frame(phot)))
        return p.plot_asteroid(d, track, os.devnull, h_mag=p._worker_h_lookup.get(d),
                               sbdb_entry=p.match_sbdb(d, p._worker_sbdb_by_number, p._worker_sbdb_by_designation))
    except Exception as ex:
        return dict(designation=d, error=f"{type(ex).__name__}: {ex}")


def labels():
    a = pd.read_csv(f"{L}/labels_final.csv").drop_duplicates("designation")[["designation", "label_final"]].assign(source="round1")
    out = [a]
    for name in ("faint2", "faint3"):
        f = f"{L}/labels_{name}.csv"
        if os.path.exists(f):
            b = pd.read_csv(f)
            out.append(b.assign(label_final=np.where(b.good, "good", "bad"), source=name)[["designation", "label_final", "source"]])
    lab = pd.concat(out, ignore_index=True)
    return lab.drop_duplicates("designation")


def recall_at_precision(y, s, target=0.9):
    o = np.argsort(-s)
    tp = np.cumsum(y[o]); prec = tp / np.arange(1, len(y) + 1)
    ok = np.flatnonzero(prec >= target)
    return float(tp[ok.max()] / y.sum()) if len(ok) else 0.0


def analyse(d):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold
    import train_raw_ozstar as tr
    d = d[d.error.isna()] if "error" in d else d
    y = (d.label_final == "good").astype(int).values
    faint = d.source.isin(["faint2", "faint3"]).values
    bad = d.label_final != "good"
    print(f"{len(d)} objects ({int(faint.sum())} faint2 + faint3), {int(y.sum())} good")
    print("bin_range_sig quantiles (10/50/90 %): good", d.bin_range_sig[~bad].quantile([.1, .5, .9]).round(2).tolist(),
          "bad", d.bin_range_sig[bad].quantile([.1, .5, .9]).round(2).tolist())
    for c in ["fold_vn", "fold_jag"]:
        print(f"{c} quantiles (10/50/90 %): good", d[c][~bad].quantile([.1, .5, .9]).round(2).tolist(),
              "bad", d[c][bad].quantile([.1, .5, .9]).round(2).tolist(),
              "| faint good", d[c][~bad & faint].quantile([.1, .5, .9]).round(2).tolist(),
              "faint bad", d[c][bad & faint].quantile([.1, .5, .9]).round(2).tolist())
    for c, sign in [("bin_range_sig", 1), ("fold_vn", -1), ("fold_jag", -1), ("aov_F", 1), ("amp_snr", 1), ("reliability_score", 1)]:
        x = sign * d[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(d[c].median()).values
        print(f"  AUC {c:18s} all {roc_auc_score(y, x):.3f}   faint2 {roc_auc_score(y[faint], x[faint]):.3f}")
    F = [f for f in tr.F if f in d.columns]
    missing = sorted(set(tr.F) - set(F))
    if missing:
        print("pipeline features not in the recomputed rows (left out):", missing)
    for name, feats in [("pipeline features", F), ("pipeline + bin_range_sig", F + ["bin_range_sig"]),
                        ("pipeline + fold_vn", F + ["fold_vn"]), ("pipeline + fold_jag", F + ["fold_jag"]),
                        ("pipeline + fold_vn + fold_jag", F + ["fold_vn", "fold_jag"])]:
        X = d[feats].astype(float).replace([np.inf, -np.inf], np.nan).values
        oof, cnt = np.zeros(len(d)), np.zeros(len(d))
        for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(X, y):
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4, l2_regularization=1.0,
                                               min_samples_leaf=20, random_state=0).fit(X[a], y[a])
            oof[b] += m.predict_proba(X[b])[:, 1]; cnt[b] += 1
        s = oof / cnt
        print(f"  classifier CV, {name:30s}: AUC all {roc_auc_score(y, s):.3f}, faint2 {roc_auc_score(y[faint], s[faint]):.3f}; "
              f"recall at 90 % precision all {recall_at_precision(y, s):.1%}, faint2 {recall_at_precision(y[faint], s[faint]):.1%}")


def main():
    fn = f"{S}/bin_range_test.csv"
    if os.environ.get("BRS_REUSE") == "1":
        return analyse(pd.read_csv(fn).merge(labels(), on="designation"))
    lab = labels()
    objects = st.load_objects(p.DATA_ROOT)
    objects = objects[objects.designation.isin(lab.designation)]
    h_lookup = {dd: float(h) for dd, h in zip(objects.designation, objects.magnitude_H) if np.isfinite(h)}
    sbdb = p.load_sbdb_physical_properties()
    rows = []
    with mp.get_context("fork").Pool(p.n_available_workers(), initializer=p._pool_worker_init,
                                     initargs=(h_lookup, *sbdb), maxtasksperchild=10) as pool:
        for k, row in enumerate(pool.imap_unordered(one, lab.designation), 1):
            rows.append(row)
            if k % 100 == 0:
                print(f"  {k}/{len(lab)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(fn, index=False)
    analyse(out.merge(lab, on="designation"))


if __name__ == "__main__":
    main()
