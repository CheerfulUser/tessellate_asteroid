"""Artefact augmentation of the synthetic set (agreed 2026-10-15): copies of existing synthetic objects with artefacts
added to their stored detrended flux, so the classifier sees strong peaks that come from systematics rather than
rotation (1999 XG19, 2001 OT35, 2000 AU31; 2012 XY7's bright event). The injected period is unchanged, so the label
stays exact (reach of P_inj x {1, 2, 1/2}; blanks / constants: never). No new photometry: the parent's stored
photometry (blank_injection/<run>/phot) + artefacts -> the unchanged pipeline (blank_injection.pipe), detection
statistic and event-dependence drops, as cut_augment.py.

Artefacts (amplitudes in units of the visit's median per-frame error e, log-uniform over AMP_*):
  event   1-3 Gaussian bumps in time (FWHM log-uniform EVENT_FWHM_HR), sign + with probability EVENT_POS
  drift   per sector a sinusoid of period log-uniform DRIFT_D days, random phase (scattered-light-like trends)
  ramp    per visit a linear ramp, total change across the visit = amplitude (normalisation keeps it)
Each copy gets one type (P_ONE) or two; a share CUT_FRAC also gets a cut_augment scheme first (the real failures
combine an artefact with poor coverage).

Calibration: the amplitude ranges are wide on purpose; `calib` compares, per amplitude band, the artefact copies'
artefact-sensitive features (loo_worst_chi2, chi2_red_fourier, n_outliers_rejected, max_1day / max_3day_power_drop)
with the real catalogue's and the artefact-free synthetic set's, so the full run's ranges can be set to what the
real data show.

Stages (RUN art1 or ART_RUN, numbers ART_OFFSET + 200 i + b):
  plan                 -> <run>/objects.csv          make <bucket>   -> <run>/phot/bucket=<b>/art.parquet
  (RUN=<run> blank_injection.py pipe <bucket>)       post <bucket>   detection power + drops
  calib                feature quantiles: real / synthetic / artefact copies by amplitude band
  assemble <base> <out>  label and append to a training table
Runs on ozstar.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import sector_report_pipeline as p
import cut_augment as ca

B = ca.B
ED = ca.ED
S = "/fred/oz335/rridden/asteroids_store"
RUN = os.environ.get("ART_RUN", "art1")
OUT = f"{B}/{RUN}"
N_ART = int(os.environ.get("N_ART", 30000))
ART_OFFSET = int(os.environ.get("ART_OFFSET", 400000))
SEED = int(os.environ.get("ART_SEED", 20261016))
N_BUCKETS = ca.N_BUCKETS
TYPES = ("event", "drift", "ramp")
P_ONE = 0.6
CUT_FRAC = 0.3
EVENT_POS = 0.8
EVENT_FWHM_HR = (0.5, 12.0)
DRIFT_D = (3.0, 15.0)
# amplitude ranges (units of e), "lo,hi" env overrides once calibrated
_r = lambda k, d: tuple(float(x) for x in os.environ[k].split(",")) if os.environ.get(k) else d
AMP_EVENT = _r("AMP_EVENT", (1.0, 100.0))
AMP_DRIFT = _r("AMP_DRIFT", (0.03, 3.0))
AMP_RAMP = _r("AMP_RAMP", (0.1, 10.0))
SCAN = 0.08829
# realistic bands from the full calibration (2026-10-15): above these, leave-one-out worst chi2 runs far past the real
# catalogue's 99th percentile (drift > 0.64 e: q90 1.9e4; ramp > 0.46 e: q90 1.7e4-5e6; real q99 238)
DRIFT_CAP = float(os.environ.get("DRIFT_CAP", 0.64))
RAMP_CAP = float(os.environ.get("RAMP_CAP", 0.46))
CALIB_FEATS = ["loo_worst_chi2", "chi2_red_fourier", "n_outliers_rejected", "max_1day_power_drop", "max_3day_power_drop"]


def plan():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(f"{B}/synthetic_training_set_all_detect.csv", low_memory=False,
                    usecols=["syn", "run", "kind", "period_inj_hr", "n_visits", "P_ref_hr", "error", "period_hr"])
    d = d[d.error.isna() & d.period_hr.notna() & d.run.isin(ca.PARENT_RUNS)]
    objs = pd.concat([pd.read_csv(f"{B}/{r}/objects.csv") for r in ca.PARENT_RUNS]).set_index("syn")
    pick = d.sample(N_ART, replace=True, random_state=rng.integers(1 << 31))
    rows, per_bucket = [], {}
    for _, r in pick.iterrows():
        b = int(r.syn[3:]) % N_BUCKETS
        i = per_bucket.get(b, 0)
        per_bucket[b] = i + 1
        num = ART_OFFSET + N_BUCKETS * i + b
        types = [TYPES[int(rng.integers(3))]] if rng.uniform() < P_ONE else list(rng.choice(TYPES, 2, replace=False))
        scheme = ca.SCHEMES[int(rng.integers(3))] if rng.uniform() < CUT_FRAC else "none"
        if scheme == "visits" and r.n_visits < 2:
            scheme = "window"
        o = objs.loc[r.syn]
        rows.append(dict(syn=f"SYN{num:06d}", num=num, source=o.source, copy=int(o["copy"]), kind=o.kind, s_frame=o.s_frame,
                         template=int(o.template), amp_inj=o.amp_inj, period_inj_hr=o.period_inj_hr, phase0=o.phase0,
                         e_ref=o.e_ref, m_ref=o.m_ref, parent=r.syn, parent_run=r.run, scheme=scheme,
                         P_ref_hr=r.period_inj_hr if r.kind == "rotator" else r.P_ref_hr, art="+".join(sorted(types)),
                         a_event=float(np.exp(rng.uniform(*np.log(AMP_EVENT)))) if "event" in types else 0.0,
                         a_drift=float(np.exp(rng.uniform(*np.log(AMP_DRIFT)))) if "drift" in types else 0.0,
                         a_ramp=float(np.exp(rng.uniform(*np.log(AMP_RAMP)))) if "ramp" in types else 0.0))
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/objects.csv", index=False)
    print(f"{len(out)} copies: kinds {out.kind.value_counts().to_dict()}, artefacts {out.art.value_counts().to_dict()}, "
          f"cut schemes {out.scheme.value_counts().to_dict()}")


def add_artefacts(g, o, rng):
    """g: one object's frames (sorted by mjd); returns the detrended flux with o's artefacts added."""
    t = g.mjd.to_numpy(float)
    f = g.flux_detrended.to_numpy(float).copy()
    vis = g[ca.VISIT_KEY].astype(str).agg("_".join, axis=1).to_numpy()
    e = g.e_flux.to_numpy(float)
    ev = pd.Series(e).groupby(vis).transform("median").to_numpy()
    if o.a_event > 0:
        for _ in range(int(rng.integers(1, 4))):
            fwhm = np.exp(rng.uniform(*np.log(EVENT_FWHM_HR))) / 24
            k = int(rng.integers(len(t)))              # centred on a frame: events happen while observing
            t0 = t[k]
            sgn = 1.0 if rng.uniform() < EVENT_POS else -1.0
            f += sgn * o.a_event * ev[k] * np.exp(-0.5 * ((t - t0) / (fwhm / 2.3548)) ** 2)
    if o.a_drift > 0:
        for s in np.unique(g.sector):
            m = (g.sector == s).to_numpy()
            tau = np.exp(rng.uniform(*np.log(DRIFT_D)))
            f[m] += o.a_drift * ev[m] * np.sin(2 * np.pi * t[m] / tau + rng.uniform(0, 2 * np.pi))
    if o.a_ramp > 0:
        for v in np.unique(vis):
            m = vis == v
            tt = t[m]
            span = max(np.ptp(tt), 1e-6)
            f[m] += rng.choice([-1.0, 1.0]) * o.a_ramp * ev[m] * ((tt - tt.min()) / span - 0.5)
    return f


def make(bucket):
    objs = pd.read_csv(f"{OUT}/objects.csv")
    objs = objs[objs.num % N_BUCKETS == bucket]
    if not len(objs):
        print(f"bucket {bucket}: no copies"); return
    tabs, n_fail = [], 0
    for run, oo in objs.groupby("parent_run"):
        files = glob.glob(f"{B}/{run}/phot/bucket={bucket}/*.parquet")
        if not files:
            continue
        ph = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        ph = ph[ph.designation.isin(set(oo.parent))]
        by = dict(tuple(ph.groupby("designation")))
        for _, o in oo.iterrows():
            g = by.get(o.parent)
            if g is None:
                n_fail += 1; continue
            g = g.sort_values("mjd")
            rng = np.random.default_rng([SEED, int(o.num)])
            if o.scheme != "none":
                keep = ca.cut_one(g, o, rng)
                if keep is None:
                    n_fail += 1; continue
                g = g[keep]
            g = g.assign(designation=o.syn, flux_detrended_clean=g.flux_detrended.values)
            g["flux_detrended"] = add_artefacts(g, o, rng)
            tabs.append(g)
    os.makedirs(f"{OUT}/phot/bucket={bucket}", exist_ok=True)
    if tabs:
        pd.concat(tabs, ignore_index=True).to_parquet(f"{OUT}/phot/bucket={bucket}/art.parquet", index=False)
    print(f"bucket {bucket}: {len(tabs)} copies written, {n_fail} without enough frames")


def post(bucket):
    ca.RUN, ca.OUT = RUN, OUT
    ca.post(bucket)


def calib():
    """Artefact-sensitive feature quantiles: real catalogue, artefact-free synthetic, artefact copies by amplitude."""
    rep = pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False,
                      usecols=["designation", "period_hr", "is_reliable"] + CALIB_FEATS[:3])
    rep = rep[rep.period_hr.notna()].drop_duplicates("designation")
    ev = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ED}/cand_*.csv") + glob.glob(f"{ED}/test_*.csv")]).drop_duplicates("designation")
    rep = rep.merge(ev[["designation"] + CALIB_FEATS[3:]], on="designation", how="left")
    syn = pd.read_csv(f"{B}/synthetic_training_set_all_detect.csv", low_memory=False, usecols=["syn", "period_hr", "error"] + CALIB_FEATS[:3])
    syn = syn[syn.error.isna() & syn.period_hr.notna()]
    evs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ED}/syn_*.csv") if "_cut1_" not in f and f"_{RUN}_" not in f]).drop_duplicates("syn")
    syn = syn.merge(evs[["syn"] + CALIB_FEATS[3:]], on="syn", how="left")
    a = pd.concat([pd.read_csv(f, low_memory=False) for f in glob.glob(f"{OUT}/pipe_*.csv")], ignore_index=True)
    a = a[a.get("error", pd.Series(np.nan, index=a.index)).isna() & a.period_hr.notna()]
    a = a.merge(pd.read_csv(f"{OUT}/objects.csv")[["syn", "art", "a_event", "a_drift", "a_ramp", "scheme"]], on="syn")
    eva = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ED}/syn_{RUN}_*.csv")]).drop_duplicates("syn")
    a = a.merge(eva[["syn"] + CALIB_FEATS[3:]], on="syn", how="left")
    a["amp"] = a[["a_event", "a_drift", "a_ramp"]].max(axis=1)
    qs = [0.5, 0.9, 0.99]
    rows = []
    def add(lab, df):
        r = dict(set=lab, n=len(df))
        for c in CALIB_FEATS:
            x = df[c].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            for q in qs:
                r[f"{c[:14]}_q{int(q * 100)}"] = float(np.quantile(x, q)) if len(x) else np.nan
        rows.append(r)
    add("real (all with a period)", rep)
    add("real reliable", rep[rep.is_reliable.astype(str).str.lower().eq("true")])
    add("synthetic, no artefacts", syn)
    for typ in TYPES:
        col = f"a_{typ}"
        sub = a[(a[col] > 0) & (a.art == typ)]
        if len(sub) < 20:
            continue
        edges = np.quantile(sub[col], [0, 1 / 3, 2 / 3, 1])
        for lo, hi in zip(edges[:-1], edges[1:]):
            add(f"{typ} {lo:.2g}-{hi:.2g} e", sub[(sub[col] >= lo) & (sub[col] <= hi)])
    t = pd.DataFrame(rows).set_index("set")
    pd.set_option("display.width", 250)
    for c in CALIB_FEATS:
        print(f"\n{c} (quantiles {qs})")
        print(t[["n"] + [k for k in t.columns if k.startswith(c[:14] + "_q")]].round(3).to_string())
    a["reach"] = (a.kind == "rotator") & np.any([((a.period_hr / a.period_inj_hr) / q - 1).abs() < SCAN for q in (1, 2, 0.5)], axis=0)
    print("\nartefact copies: rotator reach by type, and long adopted periods (>= 50 h) among blanks/constants")
    print(a[a.kind == "rotator"].groupby("art").reach.agg(["size", "mean"]).round(3).to_string())
    nb = a[a.kind != "rotator"]
    print(nb.groupby("art").apply(lambda x: pd.Series(dict(n=len(x), long=(x.period_hr >= 50).mean(),
                                                            deployed_pass=x.is_reliable.astype(str).str.lower().eq("true").mean()))).round(3).to_string())


def assemble(base_file, out_file):
    import injection_detect as idt
    base = pd.read_csv(base_file, low_memory=False)
    ok0 = base.error.isna() & base.period_hr.notna() & base.det_fap.notna() & ~base.run.astype(str).str.startswith(("cut", "art"))
    thr = float(np.quantile(np.log10(np.clip(base.det_fap[ok0 & (base.kind == "blank")], 1e-300, 1)), idt.NULL_FRAC))
    c = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(f"{OUT}/pipe_*.csv"))], ignore_index=True)
    meta = pd.read_csv(f"{OUT}/objects.csv")[["syn", "parent", "parent_run", "scheme", "art", "a_event", "a_drift", "a_ramp"]]
    det = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{B}/detect/{RUN}_*.parquet")], ignore_index=True).drop(columns=["run"])
    c = c.merge(meta, on="syn", how="left").merge(det, on="syn", how="left").assign(run=RUN)
    n0 = len(c)
    c = c[(c.a_drift <= DRIFT_CAP) & (c.a_ramp <= RAMP_CAP)].reset_index(drop=True)
    print(f"amplitude caps drift <= {DRIFT_CAP} e, ramp <= {RAMP_CAP} e: {len(c)} of {n0} copies kept")
    ok = c.get("error", pd.Series(np.nan, index=c.index)).isna() & c.period_hr.notna() & c.det_fap.notna()
    c["detected"] = ok & (np.log10(np.clip(c.det_fap, 1e-300, 1)) < thr)
    within = np.any([((c.period_hr / c.P_ref_hr) / q - 1).abs() < idt.SCAN for q in idt.HARM], axis=0)
    c["reach"] = ok & (c.kind == "rotator") & within
    c["reach_chance"] = ok & c.stand_in.fillna(False).astype(bool) & within
    c["reach_detect"] = c.reach & c.detected
    allc = pd.concat([base, c], ignore_index=True)
    allc.to_csv(out_file, index=False)
    cr = c[ok & (c.kind == "rotator")]
    print(f"artefact copies {len(c)}, through the pipeline {int(ok.sum())}")
    print(cr.groupby("art").reach.agg(["size", "mean"]).round(3).to_string())
    print(f"-> {out_file} ({len(allc):,} rows)")
    # does the training set move towards the real catalogue? (features from the pipeline rows; drops from event_dependence)
    rep = pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False, usecols=["designation", "period_hr"] + CALIB_FEATS[:3])
    rep = rep[rep.period_hr.notna()].drop_duplicates("designation")
    okb = allc.error.isna() & allc.period_hr.notna()
    for lab, df in (("real", rep), ("synthetic before (full + cut)", allc[okb & (allc.run != RUN)]), ("synthetic after (+ artefacts)", allc[okb])):
        print(f"  {lab:30s} " + "  ".join(f"{c_[:12]} q50/q90/q99 " + "/".join(f"{v:.3g}" for v in np.nanquantile(
            df[c_].astype(float).replace([np.inf, -np.inf], np.nan), [.5, .9, .99])) for c_ in CALIB_FEATS[:3]))


if __name__ == "__main__":
    a_ = sys.argv[1]
    if a_ == "plan":
        plan()
    elif a_ == "make":
        make(int(sys.argv[2]))
    elif a_ == "post":
        post(int(sys.argv[2]))
    elif a_ == "calib":
        calib()
    else:
        assemble(sys.argv[2], sys.argv[3])
