"""Synthetic injection-recovery test and training set: real high-S/N lightcurve profiles injected into the real
photometry of faint objects, run through the unchanged pipeline (plot_asteroid), labelled by whether the injected
period is recovered -- truth from construction, not from anyone's eye. Runs on ozstar.

  templates   N_TEMPLATES random reliable objects with amp_snr > 20 and > 1000 points (store report): each clean
              lightcurve (per-visit median = 1) folded at its adopted period and fitted with a TEMPLATE_HARM-term
              Fourier series, normalised to zero mean and unit peak-to-peak. Shape descriptors per template:
              amplitude (relative flux), fraction of power in each harmonic, number of maxima, depth difference of
              the two deepest minima. Hosts: N_HOSTS_PER_BAND random objects (>= MIN_HOST_POINTS points) in each
              per-frame S/N band of HOST_BANDS, so the S/N that real noise gives a physical amplitude spans the range.
              -> asteroids_store/injection/templates.csv, hosts.csv
  inject      task <index> of <total>: for each of its injections, a random host and template, period log-uniform
              in [P_MIN_HR, P_MAX_HR], random phase; BLANK_FRAC of them get no signal. The host's rel_flux is
              multiplied by (1 + A s(phase)) with A the template's own measured amplitude (a first run that set A from
              a drawn S/N needed amplitudes of 1-90x the flux on faint hosts); recorded
              S/N = A rms(s) / median(rel_flux_err) * sqrt(N). Pipeline summary row + injection parameters +
              recovered (adopted period within TOL of the injected), recovered_half, recovered_double.
              -> asteroids_store/injection/inj_<index>.csv
  summary     recovery vs S/N, period and profile type; classifier pass rate and precision on the synthetic set.

Draws come from a generator seeded by (SEED, index), so every injection is reproducible.
    python3 -u injection_test.py templates
    python3 -u injection_test.py inject <index> <total> [n_per_task]
    python3 -u injection_test.py summary
    ASTEROID_SAVE_PLOTS=1 python3 -u injection_test.py blind [n]     # blind labelling batch (default 300)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import sector_report_pipeline as p
from tessellate import asteroid_store as st

S = "/fred/oz335/rridden/asteroids_store"
OUT = f"{S}/injection"
CLEAN = f"{S}/clean_lightcurves"
N_TEMPLATES = 500
N_HOSTS_PER_BAND = 1000
HOST_BANDS = [0, 0.5, 1, 2, 5, 20, np.inf]    # per-frame S/N (largest visit avg_sig in the store tracks)
TEMPLATE_HARM = 6
MIN_TEMPLATE_POINTS = 1000
MIN_TEMPLATE_AMP_SNR = 20
MIN_HOST_POINTS = 50
P_MIN_HR, P_MAX_HR = 1.0, 500.0
SNR_BINS = [0, 3, 5, 10, 20, 50, 100, np.inf]
BLANK_FRAC = 0.10
TOL = 0.02
SEED = 20261013


def fourier(phase, coef):
    n = (len(coef) - 1) // 2
    m = np.full_like(phase, coef[0], dtype=float)
    for k in range(1, n + 1):
        m += coef[2 * k - 1] * np.cos(2 * np.pi * k * phase) + coef[2 * k] * np.sin(2 * np.pi * k * phase)
    return m


def templates():
    os.makedirs(OUT, exist_ok=True)
    r = pd.read_csv(p.SUMMARY_PATH, low_memory=False,
                    usecols=["designation", "period_hr", "amp_snr", "n_points", "is_reliable"]).drop_duplicates("designation")
    pool = r[r.is_reliable.astype(bool) & (r.amp_snr > MIN_TEMPLATE_AMP_SNR) & (r.n_points > MIN_TEMPLATE_POINTS)]
    pool = pool[[os.path.exists(f"{CLEAN}/{p.safe_name(d)}_clean_lc.csv") for d in pool.designation]]
    rows = []
    for _, t in pool.sample(frac=1, random_state=SEED).iterrows():
        lc = pd.read_csv(f"{CLEAN}/{p.safe_name(t.designation)}_clean_lc.csv")
        lc["rel"] = lc.flux / lc.groupby("visit").flux.transform("median")
        ph = (lc.mjd.values % (t.period_hr / 24)) / (t.period_hr / 24)
        A = np.column_stack([np.ones(len(ph))] + [f(2 * np.pi * k * ph) for k in range(1, TEMPLATE_HARM + 1) for f in (np.cos, np.sin)])
        coef, *_ = np.linalg.lstsq(A, lc.rel.values, rcond=None)
        g = np.linspace(0, 1, 1000, endpoint=False)
        m = fourier(g, coef) - coef[0]
        amp = float(np.ptp(m))
        if not np.isfinite(amp) or amp <= 0:
            continue
        c = coef.copy(); c[0] = 0.0; c[1:] /= amp           # zero mean, unit peak-to-peak
        pw = np.array([c[2 * k - 1] ** 2 + c[2 * k] ** 2 for k in range(1, TEMPLATE_HARM + 1)])
        s = fourier(g, c)
        n_max = int(np.sum((s > np.roll(s, 1)) & (s >= np.roll(s, -1))))
        mins = np.sort(s[(s < np.roll(s, 1)) & (s <= np.roll(s, -1))])
        rows.append(dict(template=len(rows), source=t.designation, source_period_hr=t.period_hr, amp_rel=amp,
                         n_maxima=n_max, min_depth_diff=float(mins[1] - mins[0]) if len(mins) > 1 else np.nan,
                         **{f"harm{k}_frac": float(pw[k - 1] / pw.sum()) for k in range(1, TEMPLATE_HARM + 1)},
                         **{f"c{j}": float(c[j]) for j in range(len(c))}))
        if len(rows) == N_TEMPLATES:
            break
    tp = pd.DataFrame(rows)
    tp.to_csv(f"{OUT}/templates.csv", index=False)
    tr = pd.concat([pd.read_parquet(f, columns=["designation", "avg_sig", "n_points"])
                    for f in glob.glob(f"{p.DATA_ROOT}/asteroid_store/tracks/sector*.parquet")])
    g_ = tr[tr.n_points > 0].groupby("designation").agg(frame_sn=("avg_sig", "max"), n=("n_points", "sum"))
    g_ = g_[g_.n >= MIN_HOST_POINTS]
    g_["band"] = pd.cut(g_.frame_sn, HOST_BANDS)
    hosts = pd.concat([b.sample(min(N_HOSTS_PER_BAND, len(b)), random_state=SEED) for _, b in g_.groupby("band", observed=True)]).reset_index()
    hosts["band"] = hosts.band.astype(str)
    hosts.to_csv(f"{OUT}/hosts.csv", index=False)
    print(f"template pool {len(pool):,} -> {len(tp)} templates; n_maxima {tp.n_maxima.value_counts().sort_index().to_dict()}; "
          f"amplitude quantiles 10/50/90 {tp.amp_rel.quantile([.1, .5, .9]).round(3).tolist()}; hosts per band "
          f"{hosts.band.value_counts().sort_index().to_dict()} (pool {g_.band.value_counts().sort_index().to_dict()})")


def inject(index, total, n_per_task):
    tp = pd.read_csv(f"{OUT}/templates.csv")
    hosts = pd.read_csv(f"{OUT}/hosts.csv")
    rng = np.random.default_rng([SEED, index])
    cols = [f"c{j}" for j in range(2 * TEMPLATE_HARM + 1)]
    rows = []
    for k in range(n_per_task):
        hi = int(rng.integers(len(hosts)))
        h = hosts.designation.iloc[hi]
        ti = int(rng.integers(len(tp)))
        P_hr = float(np.exp(rng.uniform(np.log(P_MIN_HR), np.log(P_MAX_HR))))
        phi0 = float(rng.uniform())
        blank = bool(rng.uniform() < BLANK_FRAC)
        inj = dict(injection=f"{index}_{k}", host=h, host_band=hosts.band.iloc[hi],
                   template=ti, period_inj_hr=P_hr, phase0=phi0, blank=blank)
        try:
            track = p.normalise(p.frames_to_points(p._store_frame(
                st.load_lightcurve(h, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))))
            if len(track) < p.MIN_POINTS_FOR_PERIODOGRAM:
                rows.append(dict(inj, error="too few points")); continue
            s = fourier(((track.mjd.values * 24 / P_hr) + phi0) % 1, tp.loc[ti, cols].values.astype(float))
            amp = 0.0 if blank else float(tp.amp_rel.iloc[ti])       # the template's own measured amplitude
            track = track.assign(rel_flux=track.rel_flux.values * (1 + amp * s))
            inj["amp_inj"] = amp
            inj["snr_inj"] = float(amp * np.sqrt(np.mean(s ** 2)) / np.median(track.rel_flux_err) * np.sqrt(len(track)))
            row = p.plot_asteroid(h, track, os.devnull)
            pr = row.get("period_hr", np.nan)
            row.update(inj, recovered=bool(not blank and np.isfinite(pr) and abs(pr / P_hr - 1) < TOL),
                       recovered_half=bool(not blank and np.isfinite(pr) and abs(2 * pr / P_hr - 1) < TOL),
                       recovered_double=bool(not blank and np.isfinite(pr) and abs(pr / (2 * P_hr) - 1) < TOL))
            rows.append(row)
        except Exception as ex:
            rows.append(dict(inj, error=f"{type(ex).__name__}: {ex}"))
    pd.DataFrame(rows).to_csv(f"{OUT}/inj_{index:04d}.csv", index=False)
    print(f"task {index}: {len(rows)} injections, {sum(r.get('recovered', False) for r in rows)} recovered", flush=True)


def make_track(row, tp):
    """Rebuild one injection's track from its saved parameters (host, template, period, amplitude, phase)."""
    cols = [f"c{j}" for j in range(2 * TEMPLATE_HARM + 1)]
    track = p.normalise(p.frames_to_points(p._store_frame(
        st.load_lightcurve(row.host, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))))
    s = fourier(((track.mjd.values * 24 / row.period_inj_hr) + row.phase0) % 1, tp.loc[int(row.template), cols].values.astype(float))
    return track.assign(rel_flux=track.rel_flux.values * (1 + row.amp_inj * s))


def blind(n_blind):
    """Blind labelling batch: BLANK_FRAC of n_blind blanks, the rest injected, equal numbers per S/N bin, shuffled,
    plotted exactly as the labelling plots (plot_asteroid, ASTEROID_SAVE_PLOTS=1) under neutral names
    blind_NNNN.png; the key (truth per file) stays in OUT/blind_key.csv."""
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(f"{OUT}/inj_*.csv"))], ignore_index=True)
    d = d[d.get("error", pd.Series(np.nan, index=d.index)).isna() & d.amp_inj.notna()]
    tp = pd.read_csv(f"{OUT}/templates.csv")
    n_bl = int(round(BLANK_FRAC * n_blind))
    inj = d[~d.blank.astype(bool)].assign(snr_bin=lambda x: pd.cut(x.snr_inj, SNR_BINS))
    per = (n_blind - n_bl) // inj.snr_bin.nunique()
    sel = pd.concat([d[d.blank.astype(bool)].sample(n_bl, random_state=SEED)]
                    + [g.sample(min(per, len(g)), random_state=SEED) for _, g in inj.groupby("snr_bin", observed=True)])
    sel = sel.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sel["file"] = [f"blind_{i + 1:04d}.png" for i in range(len(sel))]
    os.makedirs(f"{OUT}/blind", exist_ok=True)
    keep = ["file", "injection", "host", "template", "period_inj_hr", "snr_inj", "amp_inj", "phase0", "blank",
            "recovered", "recovered_half", "recovered_double", "period_hr", "reliability_score", "is_reliable"]
    sel[[c for c in keep if c in sel.columns]].to_csv(f"{OUT}/blind_key.csv", index=False)
    for _, r in sel.iterrows():
        try:
            p.plot_asteroid(r.host, make_track(r, tp), f"{OUT}/blind/{r.file}")
        except Exception as ex:
            print(f"  {r.file}: {type(ex).__name__}: {ex}", flush=True)
    print(f"blind batch: {len(sel)} plots ({n_bl} blank) -> {OUT}/blind/; key {OUT}/blind_key.csv")


def summary():
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(f"{OUT}/inj_*.csv"))], ignore_index=True)
    tp = pd.read_csv(f"{OUT}/templates.csv")
    d = d.merge(tp[["template", "n_maxima", "harm1_frac", "harm2_frac", "harm3_frac", "min_depth_diff", "amp_rel"]], on="template", how="left")
    ok = d.get("error", pd.Series(np.nan, index=d.index)).isna() & d.period_hr.notna()
    d = d[ok]
    inj, blank = d[~d.blank.astype(bool)], d[d.blank.astype(bool)]
    rel = d.is_reliable.astype(bool)
    print(f"{len(d)} runs ({len(inj)} injected, {len(blank)} blank); recovered {inj.recovered.mean():.1%} "
          f"(half {inj.recovered_half.mean():.1%}, double {inj.recovered_double.mean():.1%})")
    print(f"classifier passes: injected {rel[inj.index].mean():.1%} (of which recovered {inj[rel[inj.index]].recovered.mean():.1%}); "
          f"blanks {rel[blank.index].mean():.1%}")
    inj = inj.assign(snr_bin=pd.cut(inj.snr_inj, SNR_BINS), p_bin=pd.cut(inj.period_inj_hr, [1, 3, 10, 30, 100, 500]))
    print(inj.groupby("host_band").agg(n=("recovered", "size"), recovered=("recovered", "mean"),
                                       passes=("is_reliable", lambda x: x.astype(bool).mean())).round(3).to_string())
    print(inj.groupby("snr_bin", observed=True).agg(n=("recovered", "size"), recovered=("recovered", "mean"),
                                                     passes=("is_reliable", lambda x: x.astype(bool).mean())).round(3).to_string())
    print(inj.groupby("p_bin", observed=True).agg(n=("recovered", "size"), recovered=("recovered", "mean")).round(3).to_string())
    print(inj.groupby(["snr_bin", "n_maxima"], observed=True).recovered.mean().unstack().round(2).to_string())
    d.to_csv(f"{OUT}/synthetic_training_set.csv", index=False)


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "templates":
        templates()
    elif a == "summary":
        summary()
    elif a == "blind":
        blind(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        inject(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 20)
