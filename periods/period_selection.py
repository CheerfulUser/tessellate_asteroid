"""Period selection trained on synthetic truth: for each object, candidate periods from the periodogram, features per
candidate, a model scoring which candidate is the true period (blank-path injections, blank_injection.py), and the
highest-scoring candidate adopted. Runs on ozstar.

Candidates: a lightcurve with k maxima per rotation puts its strongest Lomb-Scargle peak at k / P, so from each of the
N_PEAKS strongest peaks f (separated by > 3 peak widths) the periods P = k / f for k = 1..4, plus the pipeline's
adopted period; candidates within DEDUP of each other merged, each refined on a local grid by a TEMPLATE_HARM-term
Fourier chi2 (per-visit offsets free). Label: within TOL of the injected period.

Features per candidate: k, peak rank and power relative to the strongest, log10 P, chi2 per point and its excess over
the best candidate, fitted amplitude, number of maxima, odd-harmonic power fraction, aov_F and fold_vn (15-bin fold),
per-visit reduced chi2 against the global model (median, worst), cycles observed, largest phase gap, whether it is the
pipeline's period; object level: points, visits, baseline, median relative error.

Stages:
  syn <run> <bucket>   candidates of every synthetic object of one bucket (blank_injection/<run>/phot) ->
                       period_selection/cand_<run>_<bucket>.parquet
  real <index> <total> candidates of real reference objects (LCDB U = 3 native periods; user-labelled objects) ->
                       period_selection/real_<index>.parquet
  train                grouped CV (by source) on the synthetic candidates: recovery of the chosen period vs the
                       pipeline's, the candidate ceiling, by S/N / maxima / period; the final model applied to the
                       real objects (LCDB truth; the user's accepted periods); model saved (not deployed)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

import sector_report_pipeline as p

S = "/fred/oz335/rridden/asteroids_store"
BI = f"{S}/blank_injection"
OUT = f"{S}/period_selection"
N_PEAKS = 5
K_MAX = 4
DEDUP = 0.003
TEMPLATE_HARM = 4
REFINE_STEPS = 15
TOL = 0.02
N_BINS = 15
P_MAX_HR = 2000.0
FEATS = ["k", "peak_rank", "peak_power_rel", "logP", "chi2_pp", "chi2_excess", "amp", "n_max", "odd_frac", "aov_F",
         "fold_vn", "visit_chi2_med", "visit_chi2_max", "n_cycles", "max_gap", "is_pipeline",
         "n_pts", "n_vis", "baseline_hr", "med_err"]


def design(t, f, vis_idx, n_vis, n=TEMPLATE_HARM):
    A = np.zeros((len(t), n_vis + 2 * n))
    A[np.arange(len(t)), vis_idx] = 1.0
    for j in range(1, n + 1):
        A[:, n_vis + 2 * j - 2] = np.cos(2 * np.pi * j * f * t)
        A[:, n_vis + 2 * j - 1] = np.sin(2 * np.pi * j * f * t)
    return A


def fit(t, y, w, f, vis_idx, n_vis):
    A = design(t, f, vis_idx, n_vis)
    c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    r = (y - A @ c) * w
    return float(np.sum(r ** 2)), c, r


def candidate_features(t, y, e, visit, p_pipe):
    """t in days, y relative flux, e errors, visit labels; returns a DataFrame of candidates with features."""
    vis_codes, vis_idx = np.unique(visit, return_inverse=True)
    n_vis = len(vis_codes)
    w = 1 / e
    base = np.ptp(t)
    width = 1 / base
    fr, pw = LombScargle(t, y, e).autopower(minimum_frequency=1 / p.MAX_PERIOD_DAYS, maximum_frequency=1 / p.MIN_PERIOD_DAYS,
                                            samples_per_peak=p.SAMPLES_PER_PEAK)
    loc = np.flatnonzero((pw > np.roll(pw, 1)) & (pw >= np.roll(pw, -1)))
    loc = loc[np.argsort(pw[loc])[::-1]]
    peaks = []
    for i in loc:
        if all(abs(fr[i] - q) > 3 * width for q, _ in peaks):
            peaks.append((fr[i], pw[i]))
        if len(peaks) == N_PEAKS:
            break
    cands = []
    for rank, (f0, pw0) in enumerate(peaks):
        for k in range(1, K_MAX + 1):
            cands.append(dict(f_peak=f0, k=k, peak_rank=rank, peak_power_rel=pw0 / peaks[0][1], P_day=k / f0))
    cands = [c for c in cands if c["P_day"] * 24 <= P_MAX_HR]
    kept = []
    for c in sorted(cands, key=lambda c: (c["peak_rank"], c["k"])):
        if all(abs(c["P_day"] / q["P_day"] - 1) > DEDUP for q in kept):
            kept.append(c)
    # the pipeline's own period always stays a candidate, unrefined (a merge or the refinement grid can move it
    # onto a neighbouring alias)
    if np.isfinite(p_pipe):
        kept.append(dict(f_peak=np.nan, k=0, peak_rank=-1, peak_power_rel=np.nan, P_day=p_pipe / 24))
    rows = []
    for c in kept:
        # refine: frequency of the rotation, +-1 peak width / k around the candidate
        f_rot = 1 / c["P_day"]
        if c["k"] == 0:
            fb = f_rot
        else:
            grid = f_rot + np.linspace(-1, 1, REFINE_STEPS) * width / c["k"]
            grid = grid[grid > 0]
            fb = grid[int(np.argmin([fit(t, y, w, g, vis_idx, n_vis)[0] for g in grid]))]
        chi2, coef, r = fit(t, y, w, fb, vis_idx, n_vis)
        P = 1 / fb
        ph = (t * fb) % 1
        g = np.linspace(0, 1, 400, endpoint=False)
        cs = coef[n_vis:]
        m = sum(cs[2 * j - 2] * np.cos(2 * np.pi * j * g) + cs[2 * j - 1] * np.sin(2 * np.pi * j * g) for j in range(1, TEMPLATE_HARM + 1))
        pw_h = np.array([cs[2 * j - 2] ** 2 + cs[2 * j - 1] ** 2 for j in range(1, TEMPLATE_HARM + 1)])
        b = np.minimum((ph * N_BINS).astype(int), N_BINS - 1)
        yc = y - coef[:n_vis][vis_idx]
        mb = np.array([yc[b == q].mean() if (b == q).sum() else np.nan for q in range(N_BINS)])
        ok = np.isfinite(mb)
        st_ = np.diff(np.append(mb[ok], mb[ok][0]))
        vn = float(np.sum(st_ ** 2) / (2 * np.sum((mb[ok] - mb[ok].mean()) ** 2))) if ok.sum() > 2 else np.nan
        gm = yc.mean()
        ssb = sum((b == q).sum() * (mb[q] - gm) ** 2 for q in range(N_BINS) if ok[q])
        ssw = sum(((yc[b == q] - mb[q]) ** 2).sum() for q in range(N_BINS) if ok[q])
        aov = float((ssb / max(ok.sum() - 1, 1)) / (ssw / max(len(y) - ok.sum(), 1))) if ssw > 0 else np.nan
        vchi = [np.mean(r[vis_idx == v] ** 2) for v in range(n_vis) if (vis_idx == v).sum() >= 10]
        srt = np.sort(ph)
        rows.append(dict(c, P_hr=P * 24, logP=np.log10(P * 24), chi2_pp=chi2 / len(y), amp=float(np.ptp(m)),
                         n_max=int(np.sum((m > np.roll(m, 1)) & (m >= np.roll(m, -1)))),
                         odd_frac=float(pw_h[0::2].sum() / pw_h.sum()) if pw_h.sum() > 0 else np.nan,
                         aov_F=aov, fold_vn=vn,
                         visit_chi2_med=float(np.median(vchi)) if vchi else np.nan,
                         visit_chi2_max=float(np.max(vchi)) if vchi else np.nan,
                         n_cycles=base / P, max_gap=float(np.max(np.diff(np.append(srt, srt[0] + 1)))),
                         is_pipeline=int(c["k"] == 0),
                         n_pts=len(t), n_vis=n_vis, baseline_hr=base * 24, med_err=float(np.median(e))))
    df = pd.DataFrame(rows)
    if len(df):
        df["chi2_excess"] = df.chi2_pp - df.chi2_pp.min()
    return df


def track_arrays(frame):
    tr = p.normalise(p.frames_to_points(frame)).sort_values("mjd").reset_index(drop=True)
    tr = tr[p.preclip_mask(tr)]
    return tr.mjd.values, tr.rel_flux.values, tr.rel_flux_err.values, tr.visit.values


def syn_bucket(run, bucket):
    os.makedirs(OUT, exist_ok=True)
    ts = pd.read_csv(f"{BI}/synthetic_training_set_all.csv", low_memory=False,
                     usecols=["syn", "source", "run", "kind", "s_frame", "template", "n_maxima", "period_inj_hr", "period_hr",
                              "recovered_2pct", "recovered", "error"])
    ts = ts[(ts.run == run) & ts.error.isna() & ts.period_hr.notna()].set_index("syn")
    files = glob.glob(f"{BI}/{run}/phot/bucket={bucket}/*.parquet")
    ph = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    out = []
    for d, g in (ph.groupby("designation") if len(ph) else []):
        if d not in ts.index:
            continue
        o = ts.loc[d]
        try:
            t, y, e, v = track_arrays(p._store_frame(g))
            if len(t) < p.MIN_POINTS_FOR_PERIODOGRAM:
                continue
            c = candidate_features(t, y, e, v, float(o.period_hr))
        except Exception as ex:
            print(f"  {d}: {type(ex).__name__}: {ex}", flush=True)
            continue
        if not len(c):
            continue
        c["correct"] = (o.kind == "rotator") & (np.abs(c.P_hr / o.period_inj_hr - 1) < TOL)
        out.append(c.assign(syn=d, source=o.source, kind=o.kind, s_frame=o.s_frame, n_maxima=o.n_maxima,
                            period_inj_hr=o.period_inj_hr, pipeline_period_hr=o.period_hr,
                            pipeline_ok=bool(o.kind == "rotator" and abs(o.period_hr / o.period_inj_hr - 1) < TOL)))
    if out:
        pd.concat(out, ignore_index=True).to_parquet(f"{OUT}/cand_{run}_{bucket:03d}.parquet", index=False)
    print(f"{run} bucket {bucket}: {len(out)} objects", flush=True)


def real_targets():
    rep = pd.read_csv(p.SUMMARY_PATH, low_memory=False, usecols=["designation", "period_hr", "is_reliable"]).drop_duplicates("designation")
    lc = pd.read_csv("/fred/oz335/rridden/asteroids/lcdb_updated.csv", low_memory=False)
    lc["designation"] = lc.designation.where(lc.designation.notna(), "(" + lc.Number.astype("Int64").astype(str) + ") " + lc.Name.astype(str))
    lc = lc[(lc.U_clean == 3) & (lc.period_source == "LCDB_native")][["designation", "Period"]].rename(columns={"Period": "truth_hr"})
    lab = pd.concat([pd.read_csv(f"{S}/label_batches/labels_final.csv")[["designation", "label_final", "labelled_period_hr"]],
                     pd.read_csv(f"{S}/label_batches/labels_faint2.csv").assign(label_final=lambda x: np.where(x.good, "good", "bad"),
                                                                                labelled_period_hr=lambda x: x.period_hr)[["designation", "label_final", "labelled_period_hr"]]]).drop_duplicates("designation")
    t = rep.merge(lc, on="designation", how="left").merge(lab, on="designation", how="left")
    return t[t.truth_hr.notna() | t.label_final.notna()].reset_index(drop=True)


def real_task(index, total):
    os.makedirs(OUT, exist_ok=True)
    from tessellate import asteroid_store as st
    t = real_targets().iloc[index::total]
    out = []
    for _, r in t.iterrows():
        try:
            fr = p._store_frame(st.load_lightcurve(r.designation, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
            tt, y, e, v = track_arrays(fr)
            if len(tt) < p.MIN_POINTS_FOR_PERIODOGRAM:
                continue
            c = candidate_features(tt, y, e, v, float(r.period_hr) if np.isfinite(r.period_hr) else np.nan)
        except Exception as ex:
            print(f"  {r.designation}: {type(ex).__name__}: {ex}", flush=True)
            continue
        out.append(c.assign(designation=r.designation, truth_hr=r.truth_hr, label_final=r.label_final,
                            labelled_period_hr=r.labelled_period_hr, pipeline_period_hr=r.period_hr, is_reliable=r.is_reliable))
    if out:
        pd.concat(out, ignore_index=True).to_parquet(f"{OUT}/real_{index:03d}.parquet", index=False)
    print(f"real task {index}: {len(out)} objects", flush=True)


def train():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    HGB = dict(max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0, min_samples_leaf=40, random_state=0)
    c = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/cand_*.parquet"))], ignore_index=True)
    X = c[FEATS].astype(float).replace([np.inf, -np.inf], np.nan).values
    y = c.correct.astype(bool).values
    oof = np.zeros(len(c))
    for a, b in GroupKFold(n_splits=5).split(c, y, groups=c.source):
        oof[b] = HistGradientBoostingClassifier(**HGB).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    c["score"] = oof
    best = c.loc[c.groupby("syn").score.idxmax()]
    obj = c.groupby("syn").agg(ceiling=("correct", "any"), kind=("kind", "first"), s_frame=("s_frame", "first"),
                               n_maxima=("n_maxima", "first"), period_inj_hr=("period_inj_hr", "first"),
                               pipeline_ok=("pipeline_ok", "first"), n_cand=("correct", "size"))
    obj["chosen_ok"] = best.set_index("syn").correct.reindex(obj.index).values
    obj["chosen_score"] = best.set_index("syn").score.reindex(obj.index).values
    rot = obj[obj.kind == "rotator"]
    print(f"{len(obj):,} objects ({len(rot):,} rotators), {len(c):,} candidates (median {obj.n_cand.median():.0f} per object)")
    print(f"rotators: pipeline period correct {rot.pipeline_ok.mean():.1%}; selected period correct {rot.chosen_ok.mean():.1%}; "
          f"candidate ceiling {rot.ceiling.mean():.1%}")
    for col, bins in (("s_frame", [0.1, 0.5, 1, 2, 5, 20, 50]), ("period_inj_hr", [1, 3, 10, 30, 100, 500])):
        g = rot.groupby(pd.cut(rot[col], bins), observed=True)
        print(g.agg(n=("chosen_ok", "size"), pipeline=("pipeline_ok", "mean"), selected=("chosen_ok", "mean"),
                    ceiling=("ceiling", "mean")).round(3).to_string())
    print(rot.groupby("n_maxima").agg(n=("chosen_ok", "size"), pipeline=("pipeline_ok", "mean"), selected=("chosen_ok", "mean"),
                                      ceiling=("ceiling", "mean")).round(3).to_string())
    # confidence of the selected candidate as a reliability score: precision / recall at thresholds
    for th in (0.5, 0.7, 0.9):
        sel = obj.chosen_score >= th
        tp = (sel & obj.chosen_ok.astype(bool)).sum()
        print(f"  selected-period confidence >= {th}: passes {sel.mean():.1%} of all objects; precision {tp / max(sel.sum(), 1):.1%}, "
              f"recall {tp / max(rot.ceiling.sum(), 1):.1%} (of recoverable rotators)")
    model = HistGradientBoostingClassifier(**HGB).fit(X, y)
    joblib.dump(dict(model=model, features=FEATS, note="period selection on synthetic truth; NOT deployed"),
                "/fred/oz335/TESSdata/asteroid_store/period_selection_synthetic.joblib")
    files = sorted(glob.glob(f"{OUT}/real_*.parquet"))
    if not files:
        return
    r = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    r["score"] = model.predict_proba(r[FEATS].astype(float).replace([np.inf, -np.inf], np.nan).values)[:, 1]
    rb = r.loc[r.groupby("designation").score.idxmax()].set_index("designation")
    rp = r.groupby("designation").first()
    lcdb = rp.truth_hr.notna()
    def ok(a, b): return np.abs(a / b - 1) < TOL
    print(f"\nreal LCDB U=3 objects ({int(lcdb.sum())}): pipeline period matches literature "
          f"{ok(rp.pipeline_period_hr[lcdb], rp.truth_hr[lcdb]).mean():.1%}, selected period {ok(rb.P_hr[lcdb], rp.truth_hr[lcdb]).mean():.1%}")
    good = rp.label_final == "good"
    print(f"user-good real objects ({int(good.sum())}): selected period = user-accepted period "
          f"{ok(rb.P_hr[good], rp.labelled_period_hr[good]).mean():.1%} (pipeline {ok(rp.pipeline_period_hr[good], rp.labelled_period_hr[good]).mean():.1%})")
    diff = lcdb & (ok(rb.P_hr, rp.truth_hr) != ok(rp.pipeline_period_hr, rp.truth_hr))
    print("LCDB objects where the two disagree on being right:")
    print(pd.DataFrame(dict(truth=rp.truth_hr, pipeline=rp.pipeline_period_hr, selected=rb.P_hr, score=rb.score))[diff].round(4).head(30).to_string())


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "syn":
        syn_bucket(sys.argv[2], int(sys.argv[3]))
    elif a == "real":
        real_task(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "train":
        train()
