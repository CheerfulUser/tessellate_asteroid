"""Cut-data augmentation of the blank-path synthetic set: copies of existing synthetic objects with parts of their
data removed, so the classifier sees long periods covered by few cycles, large phase gaps and few visits -- the
cases real artefacts pass on (1999 XG19, 2001 OT35, 2000 AU31: strong peak, long period, 0 phase coverings). No new
photometry: each copy is its parent's stored injected photometry (blank_injection/<run>/phot) with frames dropped,
then the unchanged period pipeline (blank_injection.pipe), the detection statistic (injection_detect.power) and the
event-dependence drops. Labels as for the parents (reach of P_inj x {1, 2, 1/2} within +-SCAN); CV groups stay the
parent's source, so a copy never sits in a different fold from its parent.

Cut schemes (one per copy, drawn evenly):
  visits   keep k random visits (k = 1..min(4, n_visits - 1)); parents with >= 2 visits
  window   keep one contiguous time window of length u x P_ref (u log-uniform WIN_U), P_ref = P_inj (rotators) or
           the parent's stand-in period (blanks / constants), start uniform over the data
  gap      remove one contiguous time block of GAP_FRAC of the baseline, placed uniformly
A copy is kept when >= MIN_FRAMES frames remain (up to MAX_TRIES draws).
Parents: N_CUT copies; rotators in the parents' own share, half of them drawn from P_inj >= LONG_HR.

Stages (output run blank_injection/cut1, numbers NUM_OFFSET + 200 i + b so a copy shares its parent's bucket b):
  plan                  parents, scheme and seed per copy -> cut1/objects.csv
  cut <bucket>          the copies' photometry -> cut1/phot/bucket=<b>/cut.parquet
  (then RUN=cut1 blank_injection.py pipe <bucket>)
  post <bucket>         detection power (detect/cut1_<b>.parquet) and drops (event_dependence/syn_cut1_<b>.csv)
  assemble              cut1 rows labelled like injection_detect.label, appended to the parents' set
                        -> synthetic_training_set_all_detect_cut.csv
Runs on ozstar.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import sector_report_pipeline as p

B = "/fred/oz335/rridden/asteroids_store/blank_injection"
ED = "/fred/oz335/rridden/asteroids_store/event_dependence"
RUN = "cut1"
OUT = f"{B}/{RUN}"
PARENT_RUNS = ("full", "full2", "amp", "faint")
N_CUT = int(os.environ.get("N_CUT", 30000))
NUM_OFFSET = 300000
N_BUCKETS = 200
LONG_HR = 50.0
WIN_U = (0.3, 4.0)
GAP_FRAC = (0.3, 0.7)
MIN_FRAMES = 150
MAX_TRIES = 20
SEED = 20261015
SCHEMES = ("visits", "window", "gap")
VISIT_KEY = ["sector", "cam", "ccd", "cut", "part"]


def plan():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(f"{B}/synthetic_training_set_all_detect.csv", low_memory=False,
                    usecols=["syn", "run", "kind", "period_inj_hr", "n_visits", "P_ref_hr", "error", "period_hr"])
    d = d[d.error.isna() & d.period_hr.notna() & d.run.isin(PARENT_RUNS)]
    objs = pd.concat([pd.read_csv(f"{B}/{r}/objects.csv") for r in PARENT_RUNS]).set_index("syn")
    rot = d[d.kind == "rotator"]
    n_rot = int(round(N_CUT * len(rot) / len(d)))
    long_ = rot[rot.period_inj_hr >= LONG_HR]
    pick = pd.concat([long_.sample(n_rot // 2, replace=n_rot // 2 > len(long_), random_state=rng.integers(1 << 31)),
                      rot.sample(n_rot - n_rot // 2, replace=True, random_state=rng.integers(1 << 31)),
                      d[d.kind != "rotator"].sample(N_CUT - n_rot, replace=True, random_state=rng.integers(1 << 31))])
    rows, per_bucket = [], {}
    for _, r in pick.iterrows():
        b = int(r.syn[3:]) % N_BUCKETS
        i = per_bucket.get(b, 0)
        per_bucket[b] = i + 1
        num = NUM_OFFSET + N_BUCKETS * i + b
        scheme = SCHEMES[int(rng.integers(len(SCHEMES)))]
        if scheme == "visits" and r.n_visits < 2:
            scheme = "window"
        o = objs.loc[r.syn]
        rows.append(dict(syn=f"SYN{num:06d}", num=num, source=o.source, copy=int(o["copy"]), kind=o.kind, s_frame=o.s_frame,
                         template=int(o.template), amp_inj=o.amp_inj, period_inj_hr=o.period_inj_hr, phase0=o.phase0,
                         e_ref=o.e_ref, m_ref=o.m_ref, parent=r.syn, parent_run=r.run, scheme=scheme,
                         P_ref_hr=r.period_inj_hr if r.kind == "rotator" else r.P_ref_hr))
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/objects.csv", index=False)
    print(f"{len(out)} copies: kinds {out.kind.value_counts().to_dict()}, schemes {out.scheme.value_counts().to_dict()}, "
          f"rotators with P_inj >= {LONG_HR:.0f} h {(out[out.kind == 'rotator'].period_inj_hr >= LONG_HR).mean():.0%}")


def cut_one(g, o, rng):
    """Frames of one parent kept for copy o, or None."""
    t = g.mjd.to_numpy()
    vis = g[VISIT_KEY].astype(str).agg("_".join, axis=1).to_numpy()
    uv = np.unique(vis)
    for _ in range(MAX_TRIES):
        if o.scheme == "visits":
            k = int(rng.integers(1, min(4, len(uv) - 1) + 1))
            keep = np.isin(vis, rng.choice(uv, size=k, replace=False))
        elif o.scheme == "window":
            w = np.exp(rng.uniform(*np.log(WIN_U))) * o.P_ref_hr / 24
            t0 = rng.uniform(t.min() - w, t.max())
            keep = (t >= t0) & (t <= t0 + w)
        else:
            span = t.max() - t.min()
            w = rng.uniform(*GAP_FRAC) * span
            t0 = rng.uniform(t.min(), t.max() - w)
            keep = (t < t0) | (t > t0 + w)
        if keep.sum() >= MIN_FRAMES:
            return keep
    return None


def cut(bucket):
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
            keep = cut_one(g, o, np.random.default_rng([SEED, int(o.num)]))
            if keep is None:
                n_fail += 1; continue
            tabs.append(g[keep].assign(designation=o.syn))
    os.makedirs(f"{OUT}/phot/bucket={bucket}", exist_ok=True)
    if tabs:
        pd.concat(tabs, ignore_index=True).to_parquet(f"{OUT}/phot/bucket={bucket}/cut.parquet", index=False)
    print(f"bucket {bucket}: {len(tabs)} copies written, {n_fail} without enough frames")


def post(bucket):
    import injection_detect as idt
    import event_dependence as ev
    idt.power(RUN, bucket)
    f = f"{OUT}/pipe_{bucket:03d}.csv"
    if not os.path.exists(f):
        return
    pr = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("syn", "period_hr", "error"))
    pr = pr[pr.get("error", pd.Series(np.nan, index=pr.index)).isna() & pr.period_hr.notna()].set_index("syn")
    ph = pd.concat([pd.read_parquet(x) for x in glob.glob(f"{OUT}/phot/bucket={bucket}/*.parquet")], ignore_index=True)
    rows = []
    for d, g in ph.groupby("designation"):
        if d not in pr.index:
            continue
        tr = p.normalise(p.frames_to_points(p._store_frame(g))).sort_values("mjd").reset_index(drop=True)
        tr = tr[p.preclip_mask(tr)]
        rows.append(dict(syn=d, **ev.power_drops(tr.mjd.values, tr.rel_flux.values, tr.rel_flux_err.values, float(pr.loc[d, "period_hr"]))))
    pd.DataFrame(rows).to_csv(f"{ED}/syn_{RUN}_{bucket:03d}.csv", index=False)
    print(f"{RUN} {bucket}: drops for {len(rows)}")


def assemble():
    import injection_detect as idt
    base = pd.read_csv(f"{B}/synthetic_training_set_all_detect.csv", low_memory=False)
    ok0 = base.error.isna() & base.period_hr.notna() & base.det_fap.notna()
    thr = float(np.quantile(np.log10(np.clip(base.det_fap[ok0 & (base.kind == "blank")], 1e-300, 1)), idt.NULL_FRAC))
    c = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(f"{OUT}/pipe_*.csv"))], ignore_index=True)
    meta = pd.read_csv(f"{OUT}/objects.csv")[["syn", "parent", "parent_run", "scheme"]]
    det = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{B}/detect/{RUN}_*.parquet")], ignore_index=True).drop(columns=["run"])
    c = c.merge(meta, on="syn", how="left").merge(det, on="syn", how="left").assign(run=RUN)
    ok = c.get("error", pd.Series(np.nan, index=c.index)).isna() & c.period_hr.notna() & c.det_fap.notna()
    c["detected"] = ok & (np.log10(np.clip(c.det_fap, 1e-300, 1)) < thr)
    within = np.any([((c.period_hr / c.P_ref_hr) / q - 1).abs() < idt.SCAN for q in idt.HARM], axis=0)
    c["reach"] = ok & (c.kind == "rotator") & within
    c["reach_chance"] = ok & c.stand_in.fillna(False).astype(bool) & within
    c["reach_detect"] = c.reach & c.detected
    allc = pd.concat([base, c], ignore_index=True)
    allc.to_csv(f"{B}/synthetic_training_set_all_detect_cut.csv", index=False)
    cr = c[ok & (c.kind == "rotator")]
    print(f"cut copies {len(c)}, through the pipeline {int(ok.sum())}; detection threshold log10 FAP < {thr:.2f} (from the parents' blanks)")
    print(cr.groupby("scheme").agg(n=("reach", "size"), reach=("reach", "mean"), cycles_med=("n_cycles_observed", "median"),
                                   npc0=("n_phase_coverings", lambda x: (x == 0).mean())).round(3).to_string())
    lp = cr.period_hr >= LONG_HR
    print(f"rotator copies with an adopted period >= {LONG_HR:.0f} h: {int(lp.sum())}, reach {cr.reach[lp].mean():.1%} "
          f"(parents' set: {base[ok0 & (base.kind == 'rotator') & (base.period_hr >= LONG_HR)].reach.mean():.1%}); "
          f"with 0 phase coverings {int((lp & (cr.n_phase_coverings == 0)).sum())}, reach {cr.reach[lp & (cr.n_phase_coverings == 0)].mean():.1%}")
    print(f"-> {B}/synthetic_training_set_all_detect_cut.csv ({len(allc):,} rows)")


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "plan":
        plan()
    elif a == "cut":
        cut(int(sys.argv[2]))
    elif a == "post":
        post(int(sys.argv[2]))
    else:
        assemble()
