"""Binary-asteroid identification from the TESS lightcurves (definitions agreed 2026-10-14). On the pipeline's cleaned
points (normalise, preclip) at its adopted period P1:

Two-period test: subtract a SUB_K-harmonic fit at P1, per apparition (gaps > APP_GAP_D; the shape changes with
geometry, which otherwise leaves power at P1's harmonics); Lomb-Scargle of the residuals over the pipeline's period
range (P2_RANGE), up to MIN_P2_CYCLES cycles in the longest apparition, excluding frequencies within HARM_TOL of
(m / n) f1 (m, n <= MASK_MN) and within HARM_PEAKS peak widths (1 / longest
apparition) of every integer harmonic k f1; P2 = the best remaining peak; statistic log10 of
(1 - z)^((N - 3) / 2) x M (M = independent frequencies in the band, (f_max - f_min) x baseline). The threshold is
the NULL_FRAC quantile of the same statistic, per P2 bin, on synthetic single-period rotators whose period was
reached (no second period exists there), clean and with realistic artefacts (art1): a candidate is below it, with
the two periods at least SEP_MIN apart. Ranges are broad on purpose: the known binaries are too few to tune on.

Eclipse-shape test: at P1 and 2 P1, unbinned weighted fits of (a) a smooth Fourier series (FOURIER_K harmonics at P1, 2 FOURIER_K at 2 P1 so it
keeps P1's harmonics; a 2-harmonic smooth model calls sharp-minimum rotation curves eclipses) and (b) a
flat level with two Gaussian dips (free phase, depth, width sigma <= DIP_W_MAX; 7 parameters); dbic = BIC(a) - BIC(b) (> DBIC_MIN =
eclipse-like); dip_ratio = shallower / deeper depth (unequal dips at 2 P1 mark P1 as half the orbit).

Judged on the LCDB binary table (lcdb_binary_check.csv: 238 systems with a TESS period) by match type (orbit /
half orbit / primary / none) against controls (LCDB non-binaries, U >= 3, TESS period reaches period_updated,
deployed reliable): recall, false-positive rate (target <= 2 %; an upper bound, controls hold unknown binaries), and
whether a detected P2 is the system's LCDB primary or orbital period.

  targets                 -> asteroids_store/binary_search/targets.csv
  real <index> <total>    targets' features -> binary_search/real_<index>.csv
  null <run> <bucket>     synthetic reached rotators of one photometry bucket -> binary_search/null_<run>_<bucket>.csv
  evaluate
  catalogue <i> <n>       every deployed-reliable object -> binary_search/cat_<i>.csv
  candidates              two-period flag (null threshold, separation >= SEP_MIN) + eclipse support (amp_snr >=
                          ECL_AMP_SNR), LCDB context -> binary_candidates.csv, catalogue_binary_features.csv
  verify <i> <n>          subtraction-independent checks (flat residuals in P1 phase, joint two-period fit, recurrence
                          in independent stretches) for the LCDB targets and the catalogue candidates -> verify_<i>.csv
  verify_report           full rule: stricter of the synthetic and real-control thresholds + all three checks
  iterate "<d>" <P1>      iterative search for further signals until a candidate is rejected (one object)
  plots <n>               the top n candidates, known LCDB binaries among them, n / 2 random others -> examples/
Runs on ozstar.
"""
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.optimize import least_squares

import sector_report_pipeline as p

S = "/fred/oz335/rridden/asteroids_store"
OUT = f"{S}/binary_search"
LCDB = "/fred/oz335/rridden/asteroids/lcdb_updated.csv"
SUMMARY = f"{S}/lcdb_lc_summary.csv"
FOURIER_K = 4         # smooth model of the eclipse test (2 FOURIER_K at 2 P1)
SUB_K = 8             # harmonics subtracted at P1 before the residual search (templates carry up to 6)
# P2 searched over the pipeline's own period range (not tuned on the small known-binary sample), up to the
# longest period the data sample at least MIN_P2_CYCLES times within one apparition
P2_RANGE = (24 * p.MIN_PERIOD_DAYS, 24 * p.MAX_PERIOD_DAYS)
MIN_P2_CYCLES = 2.0
MASK_MN = 6           # (m / n) f1 masked for m <= MASK_MN, n = 2..MASK_MN
HARM_TOL = 0.02
HARM_PEAKS = 3.0      # integer harmonics k f1 masked within this many peak widths (1 / baseline)
APP_GAP_D = 60.0      # d: gaps longer than this separate apparitions
APP_MIN_PTS = 80      # an apparition gets its own harmonic fit with at least this many points
DIP_W_MAX = 0.07      # Gaussian dip sigma (phase) at most this: broad dips would mimic a smooth double wave
OVERSAMPLE = 5
NULL_FRAC = 0.01
DBIC_MIN = 10.0
N_CONTROLS = 3000
SCAN = 0.08829
FOLD_BINS = 40


def design(phase, k):
    cols = [np.ones_like(phase)]
    for j in range(1, k + 1):
        cols += [np.cos(2 * np.pi * j * phase), np.sin(2 * np.pi * j * phase)]
    return np.column_stack(cols)


def wfit(A, y, e):
    c, *_ = np.linalg.lstsq(A / e[:, None], y / e, rcond=None)
    return A @ c


def apparitions(t):
    return np.concatenate([[0], np.cumsum(np.diff(t) > APP_GAP_D)])


def two_period(t, y, e, P1, full=False):
    f1 = 24 / P1
    A = design((t * f1) % 1, SUB_K)
    c_all, *_ = np.linalg.lstsq(A / e[:, None], y / e, rcond=None)
    r = y - A @ c_all
    app = apparitions(t)
    models = []                                       # (apparition, its points, the coefficients subtracted)
    for a in np.unique(app):                          # the lightcurve shape changes between apparitions: subtract
        m = app == a                                  # each apparition's own fit where it has enough points
        if m.sum() >= APP_MIN_PTS:
            c_a, *_ = np.linalg.lstsq(A[m] / e[m, None], y[m] / e[m], rcond=None)
            r[m] = y[m] - A[m] @ c_a
            models.append((a, m, c_a))
        else:
            models.append((a, m, c_all))
    T = np.ptp(t)
    T_app = max(np.ptp(t[app == a]) for a in np.unique(app))
    fmin = max(24 / P2_RANGE[1], MIN_P2_CYCLES / max(T_app, 1e-3))
    fmax = 24 / P2_RANGE[0]
    f = np.arange(fmin, fmax, 1 / (OVERSAMPLE * T))
    bad = np.zeros(len(f), bool)
    for m in range(1, MASK_MN + 1):
        for n in range(2, MASK_MN + 1):
            bad |= np.abs(f / (m / n * f1) - 1) < HARM_TOL
    for k in range(1, int(fmax / f1) + 2):            # every integer harmonic (rotation curves carry > K harmonics)
        bad |= np.abs(f - k * f1) < HARM_PEAKS / max(T_app, 1.0)
    if bad.all():
        return dict(P2_hr=np.nan, p2_stat=np.nan)
    if full:                                          # the whole band, for plots (masked frequencies flagged)
        pw_all = LombScargle(t, r, e).power(f)
        pw = pw_all[~bad]
    else:
        pw = LombScargle(t, r, e).power(f[~bad])
    fk = f[~bad]
    i = int(np.argmax(pw))
    z, N, M = float(pw[i]), len(t), max((fmax - fmin) * T, 1.0)
    stat = (N - 3) / 2 * np.log10(max(1 - z, 1e-300)) + np.log10(M)
    A = design((t * fk[i]) % 1, 2)
    amp = np.ptp(wfit(A, r, e))
    out = dict(P2_hr=24 / fk[i], p2_power=z, p2_stat=stat, p2_amp=amp)
    if full:
        out.update(_f=f, _pw=pw_all, _bad=bad, _r=r, _models=models)
    return out


def dips(phase, c, d1, p1, w1, d2, p2, w2):
    g = lambda d, ph, w: d * np.exp(-0.5 * (((phase - ph + 0.5) % 1 - 0.5) / w) ** 2)
    return c - g(d1, p1, w1) - g(d2, p2, w2)


def eclipse(t, y, e, P, k_smooth):
    ph = (t * 24 / P) % 1
    N = len(t)
    smooth = wfit(design(ph, k_smooth), y, e)
    chi_a = float(np.sum(((y - smooth) / e) ** 2))
    edges = np.linspace(0, 1, FOLD_BINS + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, FOLD_BINS - 1)
    med = np.array([np.median(y[idx == k]) if np.any(idx == k) else np.nan for k in range(FOLD_BINS)])
    cen = (edges[:-1] + edges[1:]) / 2
    o = np.argsort(np.where(np.isfinite(med), med, np.inf))
    k1 = o[0]
    far = [k for k in o[1:] if abs(((cen[k] - cen[k1]) + 0.5) % 1 - 0.5) > 0.15]
    k2 = far[0] if far else (k1 + FOLD_BINS // 2) % FOLD_BINS
    top = np.nanmax(med)
    x0 = [top, max(top - med[k1], 1e-3), cen[k1], 0.04, max(top - med[k2], 1e-4), cen[k2], 0.04]
    lo = [-np.inf, 0, -1, 0.005, 0, -1, 0.005]
    hi = [np.inf, np.inf, 2, DIP_W_MAX, np.inf, 2, DIP_W_MAX]
    try:
        res = least_squares(lambda q: (y - dips(ph, *q)) / e, x0, bounds=(lo, hi), max_nfev=2000)
        q = res.x
        chi_b = float(np.sum(res.fun ** 2))
    except Exception:
        return dict(dbic=np.nan, dip_ratio=np.nan, dip_width=np.nan)
    dbic = (chi_a + (2 * k_smooth + 1) * np.log(N)) - (chi_b + 7 * np.log(N))
    d_hi, d_lo = max(q[1], q[4]), min(q[1], q[4])
    return dict(dbic=dbic, dip_ratio=d_lo / d_hi if d_hi > 0 else np.nan, dip_depth=d_hi,
                dip_width=q[3] if q[1] >= q[4] else q[6], dip_sep=abs(((q[2] - q[5]) + 0.5) % 1 - 0.5))


def features(t, y, e, P1):
    out = dict(**two_period(t, y, e, P1))
    for lab, P, k in (("1", P1, FOURIER_K), ("2", 2 * P1, 2 * FOURIER_K)):     # at 2 P1 the smooth model keeps P1's harmonics
        out.update({f"{k_}_P{lab}": v for k_, v in eclipse(t, y, e, P, k).items()})
    return out


def track_points(df):
    tr = p.normalise(p.frames_to_points(p._store_frame(df))).sort_values("mjd").reset_index(drop=True)
    tr = tr[p.preclip_mask(tr)]
    return tr.mjd.values, tr.rel_flux.values, tr.rel_flux_err.values


def from_store(args):
    d, P1 = args
    from tessellate import asteroid_store as st
    try:
        t, y, e = track_points(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
        return dict(designation=d, n=len(t), **features(t, y, e, P1))
    except Exception as ex:
        return dict(designation=d, error=f"{type(ex).__name__}: {ex}")


def targets():
    os.makedirs(OUT, exist_ok=True)
    b = pd.read_csv(f"{S}/lcdb_binary_check.csv")
    b = b.assign(group="binary_" + b.kind)[["designation", "period_hr", "group", "primary", "orbit", "secondary", "match",
                                             "bintype", "dep_pass", "cl_pass", "amp_snr"]]
    lines = open(SUMMARY).readlines()
    cols = lines[16].strip().split(",")
    sm = pd.read_csv(SUMMARY, skiprows=22, header=None, names=cols, low_memory=False)
    sm["Number"] = pd.to_numeric(sm.Number, errors="coerce")
    nonbin = set(sm.Number[sm.IsBinary.astype(str).str.strip().eq("-")])
    lc = pd.read_csv(LCDB, low_memory=False)
    lc = lc[lc.Number.isin(nonbin) & ~lc.Number.isin(pd.read_csv(f"{S}/lcdb_binary_check.csv").Number)]
    lc = lc[pd.to_numeric(lc.U_clean, errors="coerce").ge(3) & lc.designation.notna()]
    rep = pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False,
                      usecols=["designation", "period_hr", "is_reliable", "amp_snr"]).drop_duplicates("designation")
    c = lc[["designation", "period_updated"]].merge(rep, on="designation")
    c = c[c.is_reliable.astype(str).str.lower().eq("true") & c.period_hr.notna()]
    r = c.period_hr / c.period_updated
    c = c[np.any([(r / q - 1).abs() < SCAN for q in (1, 2, 0.5)], axis=0)]
    c = c.sample(min(N_CONTROLS, len(c)), random_state=0).assign(group="control", primary=lambda x: x.period_updated, dep_pass=True)
    t = pd.concat([b, c.drop(columns=["period_updated", "is_reliable"])], ignore_index=True)
    t.to_csv(f"{OUT}/targets.csv", index=False)
    print(t.group.value_counts().to_string())


def real(index, total):
    t = pd.read_csv(f"{OUT}/targets.csv").iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=20) as pool:
        r = pd.DataFrame(pool.map(from_store, list(zip(t.designation, t.period_hr))))
    t.merge(r, on="designation").to_csv(f"{OUT}/real_{index:03d}.csv", index=False)
    print(f"task {index}: {len(r)}")


def null(run, bucket):
    """Reached single-period rotators: clean (full2) and with realistic artefacts (art1: drifts, ramps, events within
    the calibrated caps), so the null holds the slow residual structure real objects have."""
    os.makedirs(OUT, exist_ok=True)
    B = f"{S}/blank_injection"
    table = "synthetic_training_set_all_detect_cut_art.csv" if run.startswith("art") else "synthetic_training_set_all_detect.csv"
    d = pd.read_csv(f"{B}/{table}", low_memory=False,
                    usecols=["syn", "run", "kind", "reach", "period_hr", "period_inj_hr", "s_frame", "amp_snr", "error"])
    d = d[(d.run == run) & d.error.isna() & d.reach.astype(bool) & (d.kind == "rotator")].set_index("syn")
    files = glob.glob(f"{B}/{run}/phot/bucket={bucket}/*.parquet")
    ph = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ph = ph[ph.designation.isin(d.index)]
    rows = []
    for s, g in ph.groupby("designation"):
        try:
            t, y, e = track_points(g)
            rows.append(dict(syn=s, n=len(t), period_hr=d.loc[s, "period_hr"], amp_snr=d.loc[s, "amp_snr"],
                             **features(t, y, e, float(d.loc[s, "period_hr"]))))
        except Exception as ex:
            print(f"  {s}: {type(ex).__name__}: {ex}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/null_{run}_{bucket:03d}.csv", index=False)
    print(f"null {run} {bucket}: {len(rows)}")


SEP_MIN = 1.2         # longer / shorter period at least this (P2 within 20 % of P1 is leftover rotation structure)
P2_BINS = np.array([0.9, 3, 10, 30, 100, 300, 1000])   # h: the null threshold is set per P2 bin (residual noise
MIN_NULL_PER_BIN = 200                                  # reddens at long periods); a sparse bin uses the pooled one
ECL_AMP_SNR = 2.0     # the eclipse test is used only at this amp_snr or more (below it fits noise and events)
N_CAT_TASKS = 50


def catalogue(index, total):
    """The two-period and eclipse features for every deployed-reliable object."""
    rep = pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False,
                      usecols=["designation", "period_hr", "amp_snr", "is_reliable", "n_sectors"]).drop_duplicates("designation")
    rep = rep[rep.is_reliable.astype(str).str.lower().eq("true") & rep.period_hr.notna()].sort_values("designation").iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=20) as pool:
        r = pd.DataFrame(pool.map(from_store, list(zip(rep.designation, rep.period_hr))))
    rep.drop(columns=["is_reliable"]).merge(r, on="designation").to_csv(f"{OUT}/cat_{index:03d}.csv", index=False)
    print(f"catalogue task {index}: {len(r)}")


def null_threshold():
    """NULL_FRAC quantile of p2_stat on the null, per P2 bin (pooled where a bin is sparse)."""
    nl = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/null_*.csv")], ignore_index=True)
    pooled = float(np.nanquantile(nl.p2_stat, NULL_FRAC))
    b = np.clip(np.digitize(nl.P2_hr, P2_BINS) - 1, 0, len(P2_BINS) - 2)
    thr = np.array([float(np.nanquantile(nl.p2_stat[b == k], NULL_FRAC)) if (b == k).sum() >= MIN_NULL_PER_BIN else pooled
                    for k in range(len(P2_BINS) - 1)])
    print(f"null: {len(nl)} objects; per-P2-bin thresholds {dict(zip([f'{a:g}-{c:g} h' for a, c in zip(P2_BINS[:-1], P2_BINS[1:])], np.round(thr, 1)))} "
          f"(null counts {np.bincount(b, minlength=len(P2_BINS) - 1).tolist()}; pooled {pooled:.1f})")
    return thr


def control_threshold():
    """NULL_FRAC quantile of p2_stat per P2 bin on the REAL controls (LCDB non-binaries, evaluate's binary_features.csv),
    pooled where a bin has fewer than 100: real objects carry residual structure the synthetic null does not."""
    f = f"{OUT}/binary_features.csv"
    if not os.path.exists(f):
        return None
    c = pd.read_csv(f)
    c = c[c.group == "control"]
    b = np.clip(np.digitize(c.P2_hr, P2_BINS) - 1, 0, len(P2_BINS) - 2)
    pooled = float(np.nanquantile(c.p2_stat, NULL_FRAC))
    return np.array([float(np.nanquantile(c.p2_stat[b == k], NULL_FRAC)) if (b == k).sum() >= 100 else pooled
                     for k in range(len(P2_BINS) - 1)])


def strict_threshold():
    t_syn, t_ctl = null_threshold(), control_threshold()
    thr = t_syn if t_ctl is None else np.minimum(t_syn, t_ctl)
    print("thresholds used (stricter of synthetic and real controls):", np.round(thr, 1))
    return thr


def flag(d, thr):
    s, l = np.minimum(d.P2_hr, d.period_hr), np.maximum(d.P2_hr, d.period_hr)
    k = np.clip(np.digitize(d.P2_hr.fillna(1.0), P2_BINS) - 1, 0, len(P2_BINS) - 2)
    two = (d.p2_stat < thr[k]) & (l / s >= SEP_MIN)
    ecl = (d.amp_snr >= ECL_AMP_SNR) & ((d.dbic_P1 > DBIC_MIN) | (d.dbic_P2 > DBIC_MIN))
    return two, ecl


def candidates():
    import lcdb_binary_check as lb
    thr = null_threshold()
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/cat_*.csv")], ignore_index=True)
    d = d[d.get("error", pd.Series(np.nan, index=d.index)).isna()].copy()
    d["two_period"], d["eclipse"] = flag(d, thr)
    d["Number"] = pd.to_numeric(d.designation.str.extract(r"^\((\d+)\)")[0], errors="coerce")
    bt = lb.binary_table()
    d = d.merge(bt[["Number", "sumbin", "bintype", "primary", "orbit", "secondary"]].rename(columns={"primary": "lcdb_primary"}), on="Number", how="left")
    lc = pd.read_csv(LCDB, low_memory=False, usecols=["designation", "period_updated", "U_clean"]).drop_duplicates("designation")
    d = d.merge(lc, on="designation", how="left")
    d["known_binary"] = d.lcdb_primary.notna()
    d["P2_matches_lcdb"] = [harmonic_of(a, b) or harmonic_of(a, c) for a, b, c in zip(d.P2_hr, d.lcdb_primary, d.orbit)]
    c = d[d.two_period].sort_values("p2_stat")
    c.to_csv(f"{OUT}/binary_candidates.csv", index=False)
    d.to_csv(f"{OUT}/catalogue_binary_features.csv", index=False)
    print(f"deployed-reliable objects searched {len(d):,}; two-period rule: p2_stat below the per-P2-bin null threshold, periods separated >= {SEP_MIN}x")
    print(f"two-period candidates {len(c):,} ({len(c) / len(d):.1%}); with eclipse support (amp_snr >= {ECL_AMP_SNR:.0f}) {int(c.eclipse.sum())}")
    print(f"  known LCDB binaries in the search {int(d.known_binary.sum())}, flagged {int((d.known_binary & d.two_period).sum())} "
          f"(P2 = their LCDB primary or orbit: {int((d.known_binary & d.two_period & d.P2_matches_lcdb).sum())})")
    print(f"  candidates with an LCDB period but no binary flag {int((c.period_updated.notna() & ~c.known_binary).sum())}; with no LCDB period {int(c.period_updated.isna().sum())}")
    s, l = np.minimum(c.P2_hr, c.period_hr), np.maximum(c.P2_hr, c.period_hr)
    print("  shorter period (h) quantiles 10/50/90:", np.round(np.quantile(s, [.1, .5, .9]), 2), "; longer:", np.round(np.quantile(l, [.1, .5, .9]), 1))
    print("  candidates by shorter period (h):", c.groupby(pd.cut(s, P2_BINS), observed=True).size().to_dict())
    print("  candidates by longer period (h):", c.groupby(pd.cut(l, P2_BINS), observed=True).size().to_dict())
    print("  known binaries flagged by orbit (h):", d[d.known_binary & d.two_period].groupby(pd.cut(d.orbit, [0, 30, 100, 1e4]), observed=True).size().to_dict(),
          "; known binaries searched by orbit:", d[d.known_binary].groupby(pd.cut(d.orbit, [0, 30, 100, 1e4]), observed=True).size().to_dict())
    print("  by amp_snr:", c.groupby(pd.cut(c.amp_snr, [0, 1, 2, 4, 1e9]), observed=True).size().to_dict())
    print(c[["designation", "period_hr", "P2_hr", "p2_stat", "p2_amp", "amp_snr", "eclipse", "known_binary", "lcdb_primary", "orbit", "period_updated"]].head(40).round(3).to_string(index=False))


VERIFY_BINS = 20         # P1-phase bins for the flat-residual check
JOINT_K2 = 4             # harmonics of P2 in the joint fit
JOINT_SCAN = 41          # P2 frequencies tried within +-2 peak widths in the joint fit
FLAT_P = 1e-3            # residuals count as flat in P1 phase when their binned chi2 has p above this
JOINT_P = 1e-3           # the P2 series must improve the joint fit at F-test p below this


def verify_checks(t, y, e, P1, tp):
    """Subtraction-independent checks of a P2 (agreed 2026-10-15): residuals flat in P1 phase, P2 survives a joint
    two-period fit, and P2 recurs in independent stretches of data (apparitions, or the two halves of a single one)."""
    from scipy.stats import chi2 as chi2d, f as fdist
    out = {}
    P2, r = tp["P2_hr"], tp["_r"]
    f1, f2 = 24 / P1, 24 / P2
    T = np.ptp(t)
    # 1 flat residuals in P1 phase
    ph = (t * f1) % 1
    k = np.clip((ph * VERIFY_BINS).astype(int), 0, VERIFY_BINS - 1)
    w = 1 / e ** 2
    num = np.bincount(k, w * r, VERIFY_BINS); den = np.bincount(k, w, VERIFY_BINS)
    ok = den > 0
    c2 = float(np.sum(num[ok] ** 2 / den[ok]))           # sum over bins of (weighted mean / its error)^2
    out["resid_p1_chi2r"] = c2 / max(ok.sum(), 1)
    out["resid_p1_p"] = float(chi2d.sf(c2, ok.sum()))
    # 2 joint fit: P1 shape per apparition (as subtracted) + one shared P2 series, P2 refined within +-2 peak widths
    blocks = []
    for a_, m_, _ in tp["_models"]:
        if m_.sum() >= APP_MIN_PTS:
            blocks.append(design((t * f1) % 1, SUB_K) * m_[:, None])
    small = np.zeros(len(t), bool)
    for a_, m_, _ in tp["_models"]:
        if m_.sum() < APP_MIN_PTS:
            small |= m_
    if small.any():
        blocks.append(design((t * f1) % 1, SUB_K) * small[:, None])
    A1 = np.column_stack(blocks)
    sw = 1 / e
    def chi(A):
        c, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        return float(np.sum(((y - A @ c) * sw) ** 2)), c
    c_1, _ = chi(A1)
    best = (np.inf, f2, None)
    for fj in f2 + np.linspace(-2, 2, JOINT_SCAN) / T:
        A2 = design((t * fj) % 1, JOINT_K2)[:, 1:]
        cc, coef = chi(np.column_stack([A1, A2]))
        if cc < best[0]:
            best = (cc, fj, coef)
    c_12, fj, coef = best
    k2 = 2 * JOINT_K2
    dof = len(t) - A1.shape[1] - k2
    F = ((c_1 - c_12) / k2) / (c_12 / max(dof, 1))
    grid = np.linspace(0, 1, 200)
    amp_j = float(np.ptp(design(grid, JOINT_K2)[:, 1:] @ coef[-k2:]))
    amp_s = float(np.ptp(design(grid, JOINT_K2) @ np.linalg.lstsq(design((t * f2) % 1, JOINT_K2) * sw[:, None], r * sw, rcond=None)[0]))
    T_app = max(np.ptp(t[m_]) for a_, m_, _ in tp["_models"])      # shifts below one apparition's peak width are
    # only aliases between apparitions
    out.update(P2_joint_hr=24 / fj, joint_shift_widths=float(abs(fj - f2) * T_app), joint_F_p=float(fdist.sf(F, k2, max(dof, 1))),
               amp_joint=amp_j, amp_seq=amp_s, joint_amp_ratio=amp_j / amp_s if amp_s > 0 else np.nan)
    out["joint_ok"] = bool(out["joint_shift_widths"] <= 1.0 and out["joint_amp_ratio"] >= 0.5 and out["joint_F_p"] < JOINT_P)
    # 3 recurrence in independent stretches
    segs = [m_ for a_, m_, _ in tp["_models"] if m_.sum() >= APP_MIN_PTS and np.ptp(t[m_]) * 24 >= 2 * P2]
    if len(segs) < 2:                                    # one usable apparition: its two halves in time
        big = [m_ for a_, m_, _ in tp["_models"] if m_.sum() >= APP_MIN_PTS]
        segs = []
        if big:
            m_ = max(big, key=lambda x: x.sum())
            tm = np.median(t[m_])
            for h in (m_ & (t <= tm), m_ & (t > tm)):
                if h.sum() >= APP_MIN_PTS // 2 and np.ptp(t[h]) * 24 >= 2 * P2:
                    segs.append(h)
    n_rec = 0
    for m_ in segs:
        ts = t[m_]
        Ts = np.ptp(ts)
        f = np.arange(max(24 / P2_RANGE[1], MIN_P2_CYCLES / max(Ts, 1e-3)), 24 / P2_RANGE[0], 1 / (OVERSAMPLE * Ts))
        bad = np.zeros(len(f), bool)
        for mm in range(1, MASK_MN + 1):
            for nn in range(2, MASK_MN + 1):
                bad |= np.abs(f / (mm / nn * f1) - 1) < HARM_TOL
        for kk in range(1, int(f.max() / f1) + 2):
            bad |= np.abs(f - kk * f1) < HARM_PEAKS / max(Ts, 1.0)
        f = f[~bad]
        if not len(f):
            continue
        pw = LombScargle(ts, r[m_], e[m_]).power(f)
        n_rec += int(abs(f[int(np.argmax(pw))] - f2) <= 1.5 / Ts)
    out.update(n_stretch=len(segs), n_stretch_rec=n_rec)
    out["recur_ok"] = bool(len(segs) >= 2 and n_rec >= 2 and n_rec / len(segs) >= 0.5)
    out["flat_ok"] = bool(out["resid_p1_p"] > FLAT_P)
    out["verified"] = out["flat_ok"] and out["joint_ok"] and out["recur_ok"]
    return out


def verify_one(args):
    d, P1 = args
    from tessellate import asteroid_store as st
    try:
        t, y, e = track_points(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
        tp = two_period(t, y, e, P1, full=True)
        if not np.isfinite(tp.get("P2_hr", np.nan)):
            return dict(designation=d, error="no P2")
        return dict(designation=d, **{k: v for k, v in tp.items() if not k.startswith("_")}, **verify_checks(t, y, e, P1, tp))
    except Exception as ex:
        return dict(designation=d, error=f"{type(ex).__name__}: {ex}")


def verify(index, total):
    """The checks for the LCDB targets (binaries + controls) and the catalogue candidates."""
    t = pd.read_csv(f"{OUT}/targets.csv")[["designation", "period_hr", "group"]]
    cf = f"{OUT}/binary_candidates.csv"
    if os.path.exists(cf):
        c = pd.read_csv(cf)[["designation", "period_hr"]].assign(group="candidate")
        t = pd.concat([t, c[~c.designation.isin(t.designation)]], ignore_index=True)
    t = t.iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=20) as pool:
        r = pd.DataFrame(pool.map(verify_one, list(zip(t.designation, t.period_hr))))
    t.merge(r, on="designation").to_csv(f"{OUT}/verify_{index:03d}.csv", index=False)
    print(f"verify task {index}: {len(r)}")


def verify_report():
    """Full rule: p2_stat below the stricter of the synthetic-null and real-control per-bin thresholds, periods
    separated, and verified (flat residuals, joint fit, recurrence)."""
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/verify_*.csv")], ignore_index=True)
    d = d[d.get("error", pd.Series(np.nan, index=d.index)).isna()].copy()
    thr_syn = null_threshold()
    ctl = d[d.group == "control"]
    b = np.clip(np.digitize(ctl.P2_hr, P2_BINS) - 1, 0, len(P2_BINS) - 2)
    pooled = float(np.nanquantile(ctl.p2_stat, NULL_FRAC))
    thr_ctl = np.array([float(np.nanquantile(ctl.p2_stat[b == k], NULL_FRAC)) if (b == k).sum() >= 100 else pooled
                        for k in range(len(P2_BINS) - 1)])
    thr = np.minimum(thr_syn, thr_ctl)
    print("per-P2-bin thresholds: synthetic", np.round(thr_syn, 1), "; real controls", np.round(thr_ctl, 1), "(control counts",
          np.bincount(b, minlength=len(P2_BINS) - 1).tolist(), ") -> used", np.round(thr, 1))
    d["two_period"], _ = flag(d.assign(amp_snr=np.nan, dbic_P1=np.nan, dbic_P2=np.nan), thr)
    d["passes"] = d.two_period & d.verified.astype(bool)
    lb = pd.read_csv(f"{S}/lcdb_binary_check.csv")[["designation", "orbit", "primary"]]
    d = d.merge(lb, on="designation", how="left")
    d["P2_matches"] = [harmonic_of(a, b_) or harmonic_of(a, c_) for a, b_, c_ in zip(d.P2_hr, d.primary, d.orbit)]
    print("\nby group: two-period flag, then each check among the flagged, then the full rule")
    g = d.groupby("group")
    tab = pd.DataFrame({"n": g.size(), "flagged": g.two_period.mean(),
                        "flat|flag": g.apply(lambda x: x.flat_ok[x.two_period].mean()),
                        "joint|flag": g.apply(lambda x: x.joint_ok[x.two_period].mean()),
                        "recur|flag": g.apply(lambda x: x.recur_ok[x.two_period].mean()),
                        "no_stretches|flag": g.apply(lambda x: (x.n_stretch[x.two_period] < 2).mean()),
                        "passes": g.passes.mean(), "passes_n": g.passes.sum()})
    print(tab.round(3).to_string())
    kb = d[d.group.str.startswith("binary")]
    print("\nknown binaries passing, by orbit (h):", kb.groupby(pd.cut(kb.orbit, [0, 30, 100, 1e4]), observed=True).passes.agg(["size", "sum"]).to_dict())
    print(f"known binaries passing with P2 = LCDB primary or orbit: {int((kb.passes & kb.P2_matches).sum())} of {int(kb.passes.sum())}")
    c = d[(d.group == "candidate") & d.passes]
    print(f"\ncatalogue candidates passing the full rule: {len(c)} of {int((d.group == 'candidate').sum())}")
    d.to_csv(f"{OUT}/verify_all.csv", index=False)


# ---- iterative search for further companions / signals (agreed 2026-10-15): find the strongest remaining signal,
# check it, and if it passes, model it, remove it and search again, until a candidate is rejected. Each round searches
# with two detectors, because a narrow eclipse spreads its power over many harmonics and Lomb-Scargle can miss it:
#   periodic   Lomb-Scargle with every known period's harmonics and small-integer ratios masked; its statistic must beat
#              the per-P2-bin synthetic null threshold (the first-round calibration, used as is in later rounds)
#   events     Box Least Squares (dips) over EV_P_RANGE; significance from day-chunk shuffles of the same residuals
#              (times kept, values permuted between chunks: noise kept, any periodic event destroyed): a Gumbel fit to
#              the shuffled maxima gives p, which must be below EV_P_MAX
# The candidate's period is refined (chi2 over +-3 peak widths; t0 too for events). It is then checked: it must
# survive a joint linear fit with the primary (per apparition), every component found so far and itself (F-test
# p < JOINT_P), improve the fit in independent stretches, and improve it more than the same component does at
# CTRL_MARGIN x the best of N_CTRL_PERIODS unrelated control periods (residual structure of bright primaries fits any
# period); an event train also needs MIN_EVENTS events in the data. The first failure ends the search, with the
# reason recorded. Periodic thresholds are the synthetic null's (STRICT=1: the stricter of that and the real LCDB
# controls, which in the 1-3 h band reject even Pogson's independently known 3.26 h signal).
MAX_ROUNDS = 6
EV_P_RANGE = (3.0, 480.0)
EV_DURATIONS = np.array([0.5, 1.0, 2.0, 4.0, 8.0])     # h
EV_SHUFFLES = 30
EV_P_MAX = 1e-3           # Gumbel tail of the shuffled maxima (a max statistic): p below this
ITER_KS = (8, 16, 24)
N_CTRL_PERIODS = 20
CTRL_MARGIN = 2.0         # the candidate's fit improvement must exceed CTRL_MARGIN x the best control period's
                          # (six test objects: real components 7.7-35x, Eugenia's residual structure 1.007x)
MIN_EVENTS = 3            # an event train must have at least this many events covered by the data     # harmonics tried for a periodic component (narrow eclipses need many), chosen by BIC
EV_CHUNK_D = 1.0
ITER_K = 8


def _masked_freqs(f, periods, T_app):
    bad = np.zeros(len(f), bool)
    for P in periods:
        fr = 24 / P
        for mm in range(1, MASK_MN + 1):
            for nn in range(2, MASK_MN + 1):
                bad |= np.abs(f / (mm / nn * fr) - 1) < HARM_TOL
        for kk in range(1, int(f.max() / fr) + 2):
            bad |= np.abs(f - kk * fr) < HARM_PEAKS / max(T_app, 1.0)
    return bad


def _bls_stat(t, r, e, periods_h, durations_h, known, T_hr=None):
    """Best Box Least Squares dip (SNR objective) over the period grid; each duration is searched only at periods at
    least 4x longer. Periods at a known period's harmonics / small-integer ratios are excluded."""
    from astropy.timeseries import BoxLeastSquares
    bls = BoxLeastSquares(t * 24.0, r, e)
    ok = np.ones(len(periods_h), bool)
    for P in known:
        for mm in range(1, MASK_MN + 1):
            for nn in range(1, MASK_MN + 1):
                Pr = P * mm / nn
                # a known component's period and its simplest ratios are only known to ~ a peak width (capped at 5 %)
                tol = min(max(HARM_TOL, 3 * Pr / T_hr), 0.05) if (T_hr and mm <= 2 and nn <= 2) else HARM_TOL
                ok &= np.abs(periods_h / Pr - 1) > tol
    best = (-np.inf, None)
    for dur in durations_h:
        m_ = ok & (periods_h > 4 * dur)
        if not m_.any():
            continue
        res = bls.power(periods_h[m_], dur, objective="snr")
        pw = np.where(np.array(res.depth) > 0, np.array(res.power), 0)
        i = int(np.argmax(pw))
        if pw[i] > best[0]:
            best = (float(pw[i]), dict(P=float(periods_h[m_][i]), dur=float(dur), t0=float(res.transit_time[i]),
                                       depth=float(res.depth[i])))
    return best


def _component_design(th, comp):
    """Linear design columns of a found component (th = time in hours). Events: a box at t0 and, when off2 is set, a
    second box off2 of an orbit later (the other mutual event), each with its own depth."""
    if comp["kind"] == "events":
        cols = []
        for off in [0.0] + ([comp["off2"]] if comp.get("off2") is not None else []):
            ph = ((th - comp["t0"]) / comp["P"] - off + 0.5) % 1 - 0.5
            cols.append((np.abs(ph * comp["P"]) < comp["dur"] / 2).astype(float))
        return np.column_stack(cols)
    return design((th / comp["P"]) % 1, comp["K"])[:, 1:]


def iterate_signals(t, y, e, P1, null_thr, rng=None):
    from astropy.timeseries import LombScargle
    from scipy.stats import f as fdist
    rng = rng or np.random.default_rng(0)
    th = t * 24.0
    sw = 1 / e
    tp = two_period(t, y, e, P1, full=True)
    prim = np.column_stack([design((t * 24 / P1) % 1, SUB_K) * m_[:, None] for _, m_, _ in tp["_models"]])
    found, stop = [], None
    app = apparitions(t)
    T, T_app = np.ptp(t), max(np.ptp(t[app == a]) for a in np.unique(app))
    def resid(comps):
        A = np.column_stack([prim] + [_component_design(th, c) for c in comps])
        c, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        return y - A @ c, A, float(np.sum(((y - A @ c) * sw) ** 2))
    for rnd in range(1, MAX_ROUNDS + 1):
        r, A0, chi0 = resid(found)
        known = [P1] + [c["P"] for c in found]
        # periodic detector
        fmin = max(24 / P2_RANGE[1], MIN_P2_CYCLES / max(T_app, 1e-3)); fmax = 24 / P2_RANGE[0]
        f = np.arange(fmin, fmax, 1 / (OVERSAMPLE * T))
        bad = _masked_freqs(f, known, T_app)
        pw = LombScargle(t, r, e).power(f)
        pw[bad] = 0
        i = int(np.argmax(pw)); z = float(pw[i])
        N, M = len(t), max((fmax - fmin) * T, 1.0)
        stat = (N - 3) / 2 * np.log10(max(1 - z, 1e-300)) + np.log10(M)
        Pls = 24 / f[i]
        kbin = int(np.clip(np.digitize(Pls, P2_BINS) - 1, 0, len(P2_BINS) - 2))
        ls_ok = stat < null_thr[kbin] and all(max(Pls, P) / min(Pls, P) >= SEP_MIN for P in known)
        # event detector
        Pgrid = np.exp(np.linspace(np.log(EV_P_RANGE[0]), np.log(min(EV_P_RANGE[1], 24 * T_app / MIN_P2_CYCLES)), 3000))
        s_obs, ev = _bls_stat(t, r, e, Pgrid, EV_DURATIONS, known, 24 * T)
        if ev is None:                                    # every trial period masked
            s_obs, ev = -np.inf, dict(P=np.nan, dur=np.nan, t0=np.nan, depth=np.nan)
        chunk = np.floor((t - t.min()) / EV_CHUNK_D).astype(int)
        uniq = np.unique(chunk)
        null = []
        for _ in range(EV_SHUFFLES):
            perm = rng.permutation(uniq)
            mp_ = dict(zip(uniq, perm))
            order = np.argsort([mp_[c] for c in chunk], kind="stable")
            null.append(_bls_stat(t, r[order], e[order], Pgrid, EV_DURATIONS, known, 24 * T)[0])
        nl = np.array(null)
        beta = max(np.std(nl), 1e-9) * np.sqrt(6) / np.pi
        mu = np.mean(nl) - 0.5772 * beta
        ev_p = float(-np.expm1(-np.exp(-(s_obs - mu) / beta)))     # Gumbel survival of the shuffled maxima
        ev_ok = ev_p < EV_P_MAX and all(max(ev["P"], P) / min(ev["P"], P) >= SEP_MIN for P in known)
        rec = dict(round=rnd, ls_P=Pls, ls_stat=stat, ls_thr=float(null_thr[kbin]), bls_P=ev["P"], bls_depth=ev["depth"],
                   bls_dur=ev["dur"], bls_stat=s_obs, bls_p=ev_p)
        if not (ls_ok or ev_ok):
            stop = dict(rec, reason="no significant signal (periodic statistic above the null threshold, events p >= "
                                    f"{EV_P_MAX})")
            break
        # candidate component: the periodic one if it passes (Fourier at P or 2P, lower BIC), else the events
        if ls_ok:
            best = None
            for Pc in (Pls, 2 * Pls):
              for Kc in ITER_KS:
                cand = dict(kind="periodic", P=Pc, K=Kc)
                _, _, ch = resid(found + [cand])
                bic = ch + 2 * Kc * np.log(N)
                if best is None or bic < best[0]:
                    best = (bic, cand)
            cand = best[1]
        else:
            # the box search catches one event per trial period, so a two-event orbit can come out at 2P (or 3P): try
            # P, P/2, P/3, each with one box or with a second box at a free offset, and keep the lowest BIC
            best = None
            for div in (1, 2, 3):
                Pc = ev["P"] / div
                if Pc < 4 * ev["dur"]:
                    continue
                opts = [None] + list(np.arange(0.2, 0.81, 0.02))
                for off2 in opts:
                    cand = dict(kind="events", P=Pc, t0=ev["t0"], dur=ev["dur"], off2=off2)
                    _, _, ch = resid(found + [cand])
                    bic = ch + (1 if off2 is None else 3) * np.log(N)
                    if best is None or bic < best[0]:
                        best = (bic, cand)
            cand = best[1]
        # refine the period: the periodogram / box-search peak is only good to a fraction of a peak width, and an
        # unrefined component leaves its own signal behind for the next round (Balam: 32.75 h kept, 33.45 h left)
        widths = np.linspace(-3, 3, 121)
        bestP = None
        for dw in widths:
            fP = 24 / cand["P"] + dw / T          # frequency step in units of 1 / baseline (c/d)
            if fP <= 0:
                continue
            for dt0 in ((0.0,) if cand["kind"] == "periodic" else np.linspace(-0.5, 0.5, 5) * cand["dur"]):
                cc = dict(cand, P=24 / fP, **({"t0": cand["t0"] + dt0} if cand["kind"] == "events" else {}))
                _, _, ch = resid(found + [cc])
                if bestP is None or ch < bestP[0]:
                    bestP = (ch, cc)
        cand = bestP[1]
        # joint-fit check
        r1, A1, chi1 = resid(found + [cand])
        k2 = _component_design(th, cand).shape[1]
        F = ((chi0 - chi1) / k2) / (chi1 / max(N - A1.shape[1], 1))
        Fp = float(fdist.sf(F, k2, max(N - A1.shape[1], 1)))
        # recurrence: each independent stretch alone must prefer the candidate (its own joint fit improves, F p < 0.01)
        segs = [app == a for a in np.unique(app) if (app == a).sum() >= APP_MIN_PTS]
        if len(segs) < 2 and segs:
            m_ = segs[0]; tm = np.median(t[m_]); segs = [m_ & (t <= tm), m_ & (t > tm)]
        n_rec = 0
        for m_ in segs:
            Ab = np.column_stack([prim[m_]] + [_component_design(th[m_], c) for c in found])
            Ac = np.column_stack([Ab, _component_design(th[m_], cand)])
            def chi(A):
                c_, *_ = np.linalg.lstsq(A * sw[m_, None], y[m_] * sw[m_], rcond=None)
                return float(np.sum(((y[m_] - A @ c_) * sw[m_]) ** 2))
            cb, cc = chi(Ab), chi(Ac)
            Fs = ((cb - cc) / k2) / (cc / max(m_.sum() - Ac.shape[1], 1))
            n_rec += int(fdist.sf(Fs, k2, max(m_.sum() - Ac.shape[1], 1)) < 0.01)
        # control periods: the same component at N_CTRL_PERIODS unrelated periods (0.6-1.6 x P, away from every known
        # period's small-integer ratios) must improve the fit LESS than the candidate does; residual structure (bright
        # primaries) improves the fit at any period
        crng = np.random.default_rng(int(cand["P"] * 1e4) % (2 ** 31))
        trial = cand["P"] * np.exp(crng.uniform(np.log(0.6), np.log(1.6), 400))
        badc = np.zeros(len(trial), bool)
        for Pk in known + [cand["P"]]:
            for mm in range(1, 5):
                for nn in range(1, 5):
                    badc |= np.abs(trial / (Pk * mm / nn) - 1) < 0.05
        ctrl_gain = []
        for Pc in trial[~badc][:N_CTRL_PERIODS]:
            _, _, chc = resid(found + [dict(cand, P=float(Pc))])
            ctrl_gain.append(chi0 - chc)
        gain = chi0 - chi1
        rec.update(kind=cand["kind"], P=cand["P"], joint_F_p=Fp, n_stretch=len(segs), n_stretch_rec=n_rec,
                   gain=float(gain), ctrl_gain_max=float(max(ctrl_gain)) if ctrl_gain else np.nan,
                   ctrl_gain_med=float(np.median(ctrl_gain)) if ctrl_gain else np.nan)
        if not (Fp < JOINT_P and len(segs) >= 2 and n_rec >= 2 and n_rec / len(segs) >= 0.5):
            stop = dict(rec, reason="candidate failed its checks (joint fit or recurrence)")
            break
        if ctrl_gain and not gain > CTRL_MARGIN * max(ctrl_gain):
            stop = dict(rec, reason=f"candidate failed the control-period test (fit gain not > {CTRL_MARGIN}x the best of "
                                    f"{len(ctrl_gain)} unrelated periods)")
            break
        if cand["kind"] == "events":
            n_ev = 0
            for off in [0.0] + ([cand["off2"]] if cand.get("off2") is not None else []):
                cyc = np.round((th - cand["t0"]) / cand["P"] - off)
                inev = np.abs(((th - cand["t0"]) / cand["P"] - off - cyc) * cand["P"]) < cand["dur"] / 2
                n_ev += len(np.unique(cyc[inev]))
            rec["n_events"] = int(n_ev)
            if n_ev < MIN_EVENTS:
                stop = dict(rec, reason=f"event train with only {n_ev} events covered (< {MIN_EVENTS})")
                break
        found.append(dict(cand, **{k: v for k, v in rec.items() if k not in cand}))
    else:
        stop = dict(reason=f"reached MAX_ROUNDS = {MAX_ROUNDS}")
    return found, stop


def injection_limit(d, P1, thr, depths=(0.005, 0.01, 0.02, 0.03, 0.05), n_per=12, seed=1):
    """How deep must a further companion's mutual events be for the iterative search to find them in THIS object's
    data? Two-event trains (box events of equal depth, half an orbit apart, duration 2-6 h, period log-uniform
    20-300 h) are added to the real lightcurve; the search runs as for the real object; an injection counts as found
    when a component's period is within 2 % of the injected period or its half / double."""
    from tessellate import asteroid_store as st
    t, y, e = track_points(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
    rng = np.random.default_rng(seed)
    th = t * 24.0
    rows = []
    for dep in depths:
        for _ in range(n_per):
            P = float(np.exp(rng.uniform(np.log(20), np.log(300))))
            dur = float(rng.uniform(2, 6))
            t0 = float(rng.uniform(0, P))
            ph = ((th - t0) / P) % 1
            inj = np.zeros(len(t))
            for c0 in (0.0, 0.5):
                inj += (np.abs(((ph - c0 + 0.5) % 1 - 0.5) * P) < dur / 2) * dep
            found, stop = iterate_signals(t, y - inj, e, P1, thr, np.random.default_rng(int(rng.integers(1 << 30))))
            hit = any(min(abs(c["P"] / (P * q_) - 1) for q_ in (1, 0.5, 2)) < 0.02 for c in found)
            rows.append(dict(depth=dep, P=P, dur=dur, found=hit, n_found=len(found)))
            print(f"  depth {dep:.3f} P {P:7.2f} h dur {dur:.1f} h: {'found' if hit else 'missed'} ({len(found)} components)", flush=True)
    r = pd.DataFrame(rows)
    print(r.groupby("depth").found.agg(["size", "mean"]).to_string())
    return r


def iterate_one(args):
    d, P1, thr = args
    from tessellate import asteroid_store as st
    try:
        t, y, e = track_points(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
        found, stop = iterate_signals(t, y, e, P1, thr, np.random.default_rng(abs(hash(d)) % (2 ** 32)))
        return dict(designation=d, P1=P1, n_found=len(found), found=json.dumps(found), stop=json.dumps(stop))
    except Exception as ex:
        return dict(designation=d, error=f"{type(ex).__name__}: {ex}")


def plot_one(d, row, fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tessellate import asteroid_store as st
    t, y, e = track_points(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
    P1 = float(row.period_hr)
    tp = two_period(t, y, e, P1, full=True)
    P2 = tp["P2_hr"]
    Pl = max(P1, P2)
    vc = verify_checks(t, y, e, P1, tp)
    fig, ax = plt.subplots(2, 3, figsize=(16, 7.5))

    def fold(a, yy, P, lab):
        ph = (t * 24 / P) % 1                          # the phase origin the subtracted model uses
        a.plot(ph, yy, ".", ms=1.5, color="0.6", alpha=0.5, zorder=1)
        edges = np.linspace(0, 1, 31)
        idx = np.clip(np.digitize(ph, edges) - 1, 0, 29)
        mb = np.array([np.mean(yy[idx == k]) if np.any(idx == k) else np.nan for k in range(30)])
        sb = np.array([np.std(yy[idx == k]) / np.sqrt(max((idx == k).sum(), 1)) for k in range(30)])
        a.errorbar((edges[:-1] + edges[1:]) / 2, mb, sb, fmt="o", color="tab:red", ms=4, zorder=2)
        lo, hi = np.nanpercentile(yy, [1, 99])
        pad = 0.3 * (hi - lo)
        a.set_ylim(lo - pad, hi + pad)
        a.set_xlabel("Phase")
        a.set_title(lab, fontsize=10)

    fold(ax[0, 0], y, P1, f"Data folded at the pipeline period P1 = {P1:.3f} h, with the subtracted model")
    ax[0, 0].set_ylabel("Relative flux")
    grid = np.linspace(0, 1, 400)
    cols = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:cyan", "tab:brown", "black", "tab:pink"]
    for i, (a_, m_, c_) in enumerate(tp["_models"]):
        ax[0, 0].plot(grid, design(grid, SUB_K) @ c_, color=cols[i % len(cols)], lw=1.6, zorder=3,
                      label=f"Apparition {i + 1} ({int(m_.sum())} pts)" + ("" if m_.sum() >= APP_MIN_PTS else ", all-data fit"))
    ax[0, 0].legend(fontsize=7, loc="lower right", ncol=2)
    pp = 24 / tp["_f"]
    ax[0, 1].plot(pp, tp["_pw"], color="tab:blue", lw=0.6)
    ax[0, 1].plot(pp[tp["_bad"]], tp["_pw"][tp["_bad"]], ".", ms=1, color="0.7", label="Masked (P1 harmonics)")
    ax[0, 1].axvline(P2, color="tab:red", ls="--", lw=1, label=f"P2 = {P2:.3f} h")
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlabel("Period (h)")
    ax[0, 1].set_ylabel("Residual Lomb-Scargle power")
    ax[0, 1].legend(fontsize=8, loc="upper right")
    ax[0, 1].set_title(f"Residuals after the P1 fit (per apparition): statistic {tp['p2_stat']:.1f}", fontsize=10)
    fold(ax[1, 0], tp["_r"], P2, f"Residuals folded at P2 = {P2:.3f} h")
    ax[1, 0].set_ylabel("Residual relative flux")
    other = tp["_r"] if Pl == P2 else y
    fold(ax[1, 1], other, 2 * Pl, f"{'Residuals' if Pl == P2 else 'Data'} folded at 2 x the longer period = {2 * Pl:.2f} h")
    ax[1, 1].set_ylabel("Relative flux")
    fold(ax[0, 2], tp["_r"], P1, f"Residuals folded at P1: binned chi2/dof {vc['resid_p1_chi2r']:.2f}, p = {vc['resid_p1_p']:.1g} "
         f"({'flat' if vc['flat_ok'] else 'NOT flat'})")
    ax[0, 2].axhline(0, color="k", lw=0.8)
    ax[0, 2].set_ylabel("Residual relative flux")
    ax[1, 2].axis("off")
    ax[1, 2].text(0.02, 0.95, "\n".join([
        "Checks that the P2 is not a subtraction artefact:",
        f"1  Residuals flat in P1 phase: {'yes' if vc['flat_ok'] else 'no'} (p = {vc['resid_p1_p']:.2g})",
        f"2  Joint two-period fit: P2 = {vc['P2_joint_hr']:.3f} h (shift {vc['joint_shift_widths']:.2f} peak widths),",
        f"    amplitude {vc['amp_joint']:.3f} vs {vc['amp_seq']:.3f} sequential, F-test p = {vc['joint_F_p']:.1g}: "
        f"{'survives' if vc['joint_ok'] else 'FAILS'}",
        f"3  Recurs in independent stretches: {vc['n_stretch_rec']} of {vc['n_stretch']} "
        f"({'yes' if vc['recur_ok'] else 'no' if vc['n_stretch'] >= 2 else 'too few stretches'})",
        "",
        f"Verified: {'YES' if vc['verified'] else 'NO'}"]), va="top", family="monospace", fontsize=9, transform=ax[1, 2].transAxes)
    known = (f"LCDB binary: primary {row.lcdb_primary:.3f} h, orbit {row.orbit:.2f} h" if row.known_binary
             else f"LCDB period {row.period_updated:.3f} h (not flagged binary)" if np.isfinite(row.period_updated) else "No LCDB period")
    fig.suptitle(f"{d}: amp_snr {row.amp_snr:.2f}; eclipse support {'yes' if row.eclipse else 'no'}; {known}", fontsize=11)
    fig.tight_layout()
    fig.savefig(fn, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plots(n, cand=None, out=None, only=None):
    c = pd.read_csv(cand or f"{OUT}/binary_candidates.csv")
    out = out or f"{OUT}/examples"
    os.makedirs(out, exist_ok=True)
    top = c.head(n)
    kb = c[c.known_binary].head(max(6, n // 4))
    rest = c.iloc[n:]
    rnd = rest.sample(min(n // 2, len(rest)), random_state=1)
    sel = pd.concat([top.assign(why="top"), kb.assign(why="known"), rnd.assign(why="random")]).drop_duplicates("designation")
    if only:                                          # just these designations (e.g. to redraw a few)
        sel = c[c.designation.isin(only)].assign(why="redraw")
    for _, r in sel.iterrows():
        tag = "LCDBbinary" if r.known_binary else ("LCDB" if np.isfinite(r.period_updated) else "new")
        fn = (f"{out}/{r.why}__stat{r.p2_stat:.0f}__P1_{r.period_hr:.2f}h__P2_{r.P2_hr:.2f}h__{tag}__"
              f"{p.safe_name(r.designation)}.png")
        try:
            plot_one(r.designation, r, fn)
        except Exception as ex:
            print(f"  {r.designation}: {type(ex).__name__}: {ex}")
    print(f"{len(sel)} plots -> {out}")


def harmonic_of(P2, ref):
    if not (np.isfinite(P2) and np.isfinite(ref)):
        return False
    return any(abs(P2 / (ref * q) - 1) < SCAN for q in (1, 2, 0.5))


def evaluate():
    nl = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/null_*.csv")], ignore_index=True)
    thr = null_threshold()
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/real_*.csv")], ignore_index=True)
    d = d[d.get("error", pd.Series(np.nan, index=d.index)).isna()].copy()
    d["two_period"], _ = flag(d, thr)
    d["eclipse"] = (d.dbic_P1 > DBIC_MIN) | (d.dbic_P2 > DBIC_MIN)
    d["either"] = d.two_period | d.eclipse
    d["P2_is_primary"] = [harmonic_of(a, b) for a, b in zip(d.P2_hr, d.primary)]
    d["P2_is_orbit"] = [harmonic_of(a, b) for a, b in zip(d.P2_hr, d.orbit)]
    nf, _ = flag(nl, thr)
    print(f"null flagged by the two-period rule: {nf.mean():.2%}; "
          f"eclipse test on the null: dbic > {DBIC_MIN:.0f} at P1 or 2P1 in {((nl.dbic_P1 > DBIC_MIN) | (nl.dbic_P2 > DBIC_MIN)).mean():.1%}")
    g = d.groupby("group").agg(n=("two_period", "size"), two_period=("two_period", "mean"), eclipse=("eclipse", "mean"),
                               either=("either", "mean"), P2_is_primary=("P2_is_primary", "mean"), P2_is_orbit=("P2_is_orbit", "mean"))
    print(g.round(3).to_string())
    for lab, m in (("binary, reliable (deployed or C+L+ev)", d.group.str.startswith("binary") & (d.dep_pass.astype(bool) | d.cl_pass.fillna(False).astype(bool))),
                   ("control", d.group == "control")):
        x = d[m]
        print(f"{lab} ({len(x)}): two-period {x.two_period.mean():.1%}, eclipse {x.eclipse.mean():.1%}, either {x.either.mean():.1%}; "
              f"two-period with P2 = LCDB primary or orbit {(x.two_period & (x.P2_is_primary | x.P2_is_orbit)).mean():.1%}")
    tp = d[d.group.str.startswith("binary") & d.two_period]
    print("\nbinaries flagged by the two-period test:")
    print(tp[["designation", "group", "primary", "orbit", "period_hr", "P2_hr", "p2_stat", "p2_amp", "P2_is_primary", "P2_is_orbit", "dbic_P1", "dbic_P2", "dip_ratio_P2"]]
          .sort_values("p2_stat").round(3).to_string(index=False))
    d.to_csv(f"{OUT}/binary_features.csv", index=False)


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "targets":
        targets()
    elif a == "real":
        real(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "null":
        null(sys.argv[2], int(sys.argv[3]))
    elif a == "catalogue":
        catalogue(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "candidates":
        candidates()
    elif a == "verify":
        verify(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "verify_report":
        verify_report()
    elif a == "iterate":                 # iterate "<designation>" <P1_hr>   (one object, printed)
        thr = strict_threshold() if os.environ.get("STRICT") else null_threshold()
        r_ = iterate_one((sys.argv[2], float(sys.argv[3]), thr))
        print(json.dumps({k: (json.loads(v) if k in ("found", "stop") else v) for k, v in r_.items()}, indent=1, default=float))
    elif a == "inject_limit":            # inject_limit "<designation>" <P1_hr>
        r_ = injection_limit(sys.argv[2], float(sys.argv[3]), strict_threshold() if os.environ.get("STRICT") else null_threshold())
        r_.to_csv(f"{OUT}/inject_limit_{p.safe_name(sys.argv[2])}.csv", index=False)
    elif a == "plots":
        plots(int(sys.argv[2]), *(sys.argv[3:5] if len(sys.argv) > 4 else []), only=sys.argv[5:] or None)
    else:
        evaluate()
