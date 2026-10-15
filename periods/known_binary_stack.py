"""Stack the TESS data of the known (LCDB) binaries at their published orbital period and compare with the published
values. Per system with an orbital period: the primary is removed per apparition with a SUB_K-harmonic fit at the
LCDB primary period (the pipeline period when none), the residuals are folded at the LCDB orbit and averaged in
N_BINS phase bins (stacking every orbit TESS covered), and
  flat_p        chi2 of the binned stack against zero (p small = structure at the orbit period)
  dip fit       flat level + two Gaussian dips (binary_search.dips): depths, widths, separation in phase
  expected      the deepest total mutual event, q^2 / (1 + q^2) with q = Ds/Dp from the LCDB (secondary fully hidden,
                or covering q^2 of the primary's disc); partial events are shallower
  limit         3 x the median error of a binned point: the depth the stack could detect
Per-apparition stacks are kept too, so recurring events can be told from one-off structure.

  run <i> <n>   -> asteroids_store/known_binaries/stack_<i>.csv and stacks/<key>.json (binned stacks for figures)
  report        the table, sorted by expected depth / limit
  plots         one figure per system -> known_binaries/figures/
  control <i> <n>, control_report   the same at N_CONTROL control periods per system: is the structure specific to
                the orbit, or does it appear at any period (red residuals, e.g. bright objects with changing shapes)?
Runs on ozstar.
"""
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

import sector_report_pipeline as p
import binary_search as bs

S = "/fred/oz335/rridden/asteroids_store"
OUT = f"{S}/known_binaries"
N_BINS = 40
SUB_K = bs.SUB_K


def binary_table():
    f = f"{S}/lcdb_lc_binary.csv"
    lines = open(f).readlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Number,Name"))
    dat = next(i for i, l in enumerate(lines) if l.startswith("# Data"))
    b = pd.read_csv(f, skiprows=dat + 1, header=None, names=lines[hdr].strip().split(","))
    for c in ("SumPeriod", "OrbPer", "DsDp", "ADp", "EventMin", "EventMax"):
        b[c] = pd.to_numeric(b[c], errors="coerce").where(lambda x: x > 0)
    b = b[b.Number > 0]
    return b.groupby("Number").agg(name=("Name", "first"), bintype=("BinType", lambda s: "".join(sorted(set(s) - {"-"}))),
                                   primary=("SumPeriod", "first"), orbit=("OrbPer", "median"), q=("DsDp", "median"),
                                   a_Dp=("ADp", "median"), ev_min=("EventMin", "median"), ev_max=("EventMax", "median")).reset_index()


def targets():
    t = pd.read_csv(f"{S}/lcdb_binary_check.csv")[["designation", "Number", "period_hr", "amp_snr"]]
    t = t.merge(binary_table(), on="Number")
    return t[t.orbit.notna()].reset_index(drop=True)


def stack(t, r, e, P):
    ph = (t * 24 / P) % 1
    k = np.clip((ph * N_BINS).astype(int), 0, N_BINS - 1)
    w = 1 / e ** 2
    sw = np.bincount(k, w, N_BINS)
    m = np.where(sw > 0, np.bincount(k, w * r, N_BINS) / np.where(sw > 0, sw, 1), np.nan)
    se = np.where(sw > 0, 1 / np.sqrt(np.where(sw > 0, sw, 1)), np.nan)
    # inflate the formal errors by the residual scatter (red noise, underestimated e)
    chi_pt = np.sum(w * (r - m[k]) ** 2) / max(len(r) - N_BINS, 1)
    se = se * np.sqrt(max(chi_pt, 1.0))
    return m, se, ph


def one(row):
    from tessellate import asteroid_store as st
    from scipy.stats import chi2 as chi2d
    from scipy.optimize import least_squares
    d = row["designation"]
    try:
        t, y, e = bs.track_points(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
        P1 = row["primary"] if np.isfinite(row["primary"]) else row["period_hr"]
        A = bs.design((t * 24 / P1) % 1, SUB_K)
        r = y - bs.wfit(A, y, e)
        app = bs.apparitions(t)
        for a in np.unique(app):
            m_ = app == a
            if m_.sum() >= bs.APP_MIN_PTS:
                r[m_] = y[m_] - bs.wfit(A[m_], y[m_], e[m_])
        P = row["orbit"]
        mb, se, ph = stack(t, r, e, P)
        ok = np.isfinite(mb)
        c2 = float(np.nansum((mb[ok] / se[ok]) ** 2))
        out = dict(designation=d, n=len(t), n_orbits=float(np.ptp(t) * 24 / P), P1_used=P1,
                   flat_chi2r=c2 / max(ok.sum(), 1), flat_p=float(chi2d.sf(c2, ok.sum())),
                   limit=3 * float(np.nanmedian(se)), n_apparitions=int(len(np.unique(app))))
        # dip fit on the unbinned residuals
        cen = (np.arange(N_BINS) + 0.5) / N_BINS
        o = np.argsort(np.where(ok, mb, np.inf))
        k1 = o[0]
        far = [k for k in o[1:] if abs(((cen[k] - cen[k1]) + 0.5) % 1 - 0.5) > 0.15]
        k2 = far[0] if far else (k1 + N_BINS // 2) % N_BINS
        x0 = [0.0, max(-mb[k1], 1e-3), cen[k1], 0.03, max(-mb[k2], 1e-4), cen[k2], 0.03]
        res = least_squares(lambda q: (r - bs.dips(ph, *q)) / e, x0, bounds=([-np.inf, 0, -1, 0.005, 0, -1, 0.005],
                                                                           [np.inf, np.inf, 2, bs.DIP_W_MAX, np.inf, 2, bs.DIP_W_MAX]))
        q = res.x
        chi_flat = float(np.sum(((r - np.average(r, weights=1 / e ** 2)) / e) ** 2))
        out.update(dip1=q[1], dip1_phase=q[2] % 1, dip1_w=q[3], dip2=q[4], dip2_phase=q[5] % 1, dip2_w=q[6],
                   dip_dbic=(chi_flat + np.log(len(t))) - (float(np.sum(res.fun ** 2)) + 7 * np.log(len(t))))
        per_app = []
        for a in np.unique(app):
            m_ = app == a
            if m_.sum() >= bs.APP_MIN_PTS:
                ma, sa, _ = stack(t[m_], r[m_], e[m_], P)
                per_app.append(dict(t_mid=float(np.median(t[m_])), mean=np.nan_to_num(ma, nan=0).tolist(), err=np.nan_to_num(sa, nan=0).tolist()))
        os.makedirs(f"{OUT}/stacks", exist_ok=True)
        json.dump(dict(designation=d, P1=P1, orbit=P, centres=cen.tolist(), mean=np.nan_to_num(mb, nan=0).tolist(),
                       err=np.nan_to_num(se, nan=0).tolist(), dip=q.tolist(), per_app=per_app),
                  open(f"{OUT}/stacks/{p.safe_name(d)}.json", "w"))
        return out
    except Exception as ex:
        return dict(designation=d, error=f"{type(ex).__name__}: {ex}")


N_CONTROL = 20


def control_one(row):
    """The same stack and two-dip fit at N_CONTROL control periods (orbit x 0.6-1.6, away from the orbit, the primary
    and their small-integer ratios): residual structure that is not specific to the orbit shows up at these too."""
    from tessellate import asteroid_store as st
    from scipy.optimize import least_squares
    d = row["designation"]
    try:
        t, y, e = bs.track_points(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
        P1 = row["primary"] if np.isfinite(row["primary"]) else row["period_hr"]
        A = bs.design((t * 24 / P1) % 1, SUB_K)
        r = y - bs.wfit(A, y, e)
        app = bs.apparitions(t)
        for a in np.unique(app):
            m_ = app == a
            if m_.sum() >= bs.APP_MIN_PTS:
                r[m_] = y[m_] - bs.wfit(A[m_], y[m_], e[m_])
        rng = np.random.default_rng(int(row["Number"]))
        P0 = row["orbit"]
        cand = P0 * np.exp(rng.uniform(np.log(0.6), np.log(1.6), 400))
        bad = np.zeros(len(cand), bool)
        for ref in (P0, P1):
            for mm in range(1, 5):
                for nn in range(1, 5):
                    bad |= np.abs(cand / (ref * mm / nn) - 1) < 0.05
        cand = cand[~bad][:N_CONTROL]
        chi_flat = float(np.sum(((r - np.average(r, weights=1 / e ** 2)) / e) ** 2))
        out = []
        for P in cand:
            mb, se, ph = stack(t, r, e, P)
            ok = np.isfinite(mb)
            cen = (np.arange(N_BINS) + 0.5) / N_BINS
            o = np.argsort(np.where(ok, mb, np.inf)); k1 = o[0]
            far = [k for k in o[1:] if abs(((cen[k] - cen[k1]) + 0.5) % 1 - 0.5) > 0.15]
            k2 = far[0] if far else (k1 + N_BINS // 2) % N_BINS
            x0 = [0.0, max(-mb[k1], 1e-3), cen[k1], 0.03, max(-mb[k2], 1e-4), cen[k2], 0.03]
            res = least_squares(lambda q: (r - bs.dips(ph, *q)) / e, x0, bounds=([-np.inf, 0, -1, 0.005, 0, -1, 0.005],
                                                                               [np.inf, np.inf, 2, bs.DIP_W_MAX, np.inf, 2, bs.DIP_W_MAX]))
            out.append(dict(designation=d, P_control=P, flat_chi2r=float(np.nansum((mb[ok] / se[ok]) ** 2)) / max(ok.sum(), 1),
                            dip_dbic=(chi_flat + np.log(len(t))) - (float(np.sum(res.fun ** 2)) + 7 * np.log(len(t))),
                            deepest=float(max(res.x[1], res.x[4]))))
        return out
    except Exception as ex:
        return [dict(designation=d, error=f"{type(ex).__name__}: {ex}")]


def control(index, total):
    t = targets().iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=10) as pool:
        r = pool.map(control_one, [row for _, row in t.iterrows()])
    pd.DataFrame([x for xs in r for x in xs]).to_csv(f"{OUT}/control_{index:03d}.csv", index=False)
    print(f"control task {index}: {len(r)}")


def control_report():
    d = pd.read_csv(f"{OUT}/known_binary_stacks.csv")
    c = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/control_*.csv")], ignore_index=True)
    c = c[c.get("error", pd.Series(np.nan, index=c.index)).isna()]
    g = c.groupby("designation").agg(ctrl_dbic_med=("dip_dbic", "median"), ctrl_dbic_max=("dip_dbic", "max"),
                                     ctrl_deep_med=("deepest", "median"), n_ctrl=("dip_dbic", "size"))
    d = d.merge(g, on="designation", how="left")
    d["beats_controls"] = d.dip_dbic > d.ctrl_dbic_max
    d["orbit_specific"] = d.detected & d.beats_controls
    print(f"detected at the orbit {int(d.detected.sum())}; of those, orbit-specific (dip dBIC above every control period's) "
          f"{int(d.orbit_specific.sum())}")
    print(d.groupby(pd.cut(d.exp_over_limit, [0, 0.3, 1, 3, 1e9]), observed=True).orbit_specific.agg(["size", "sum"]).to_string())
    pd.set_option("display.width", 250)
    cols = ["designation", "bintype", "primary", "orbit", "q", "expected", "limit", "deepest", "dip_dbic", "ctrl_dbic_med",
            "ctrl_dbic_max", "ctrl_deep_med", "detected", "orbit_specific", "amp_snr"]
    sel = d.designation.str.contains("Kalliope|Pogson|Elektra|Daphne|Sylvia|Arlon|Euphrosyne|Antiope")
    print(d[sel][cols].round(4).to_string(index=False))
    print("\nbright (amp_snr > 50):")
    print(d[d.amp_snr > 50][cols].round(4).to_string(index=False))
    d.to_csv(f"{OUT}/known_binary_stacks_controls.csv", index=False)


def overlap(d, r1, r2):
    """Area of overlap of two circles of radii r1, r2 whose centres are d apart."""
    d = np.asarray(d, float)
    out = np.zeros_like(d)
    full = d <= abs(r1 - r2)
    out[full] = np.pi * min(r1, r2) ** 2
    part = (d < r1 + r2) & ~full
    dd = d[part]
    a1 = r1 ** 2 * np.arccos(np.clip((dd ** 2 + r1 ** 2 - r2 ** 2) / (2 * dd * r1), -1, 1))
    a2 = r2 ** 2 * np.arccos(np.clip((dd ** 2 + r2 ** 2 - r1 ** 2) / (2 * dd * r2), -1, 1))
    a3 = 0.5 * np.sqrt(np.clip((-dd + r1 + r2) * (dd + r1 - r2) * (dd - r1 + r2) * (dd + r1 + r2), 0, None))
    out[part] = a1 + a2 - a3
    return out


def two_disc(t_hr, par):
    """Mutual events of two uniform discs (primary radius 1, secondary q) on a circular orbit of period P (h), radius
    a, impact parameter b, the secondary in front at time t0 (h); k2 scales the other event (shadows / albedo)."""
    off, q, a, b, t0, k2, P = par
    th = 2 * np.pi * (t_hr - t0) / P
    d = np.sqrt((a * np.sin(th)) ** 2 + (b * np.cos(th)) ** 2)
    loss = overlap(d, 1.0, q) / (np.pi * (1 + q ** 2))
    return off - np.where(np.cos(th) > 0, loss, k2 * loss)


ORB_SCAN = 0.01           # fractional half-range of the orbital-period grid around the starting period
ORB_STEPS = 401


def fit_orbit(designation, P_start, P1=None):
    """Fit the orbital period from the TESS data: primary removed per apparition; a grid of P (+-ORB_SCAN) with the
    two-disc event model fitted to each binned stack; the best then refined with P free on the unbinned residuals.
    Uncertainty from the covariance, inflated by sqrt(reduced chi2)."""
    from tessellate import asteroid_store as st
    from scipy.optimize import least_squares
    t, y, e = bs.track_points(st.load_lightcurve(designation, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
    tp = bs.two_period(t, y, e, P1, full=True) if P1 else None
    r = tp["_r"]
    th = t * 24.0
    grid = P_start * (1 + np.linspace(-ORB_SCAN, ORB_SCAN, ORB_STEPS))
    best = None
    for P in grid:
        mb, se, _ = stack(t, r, e, P)
        ok = np.isfinite(mb) & (se > 0)
        cen = (np.arange(N_BINS) + 0.5) / N_BINS
        f0 = cen[np.nanargmin(np.where(ok, mb, np.inf))]
        res = least_squares(lambda x: (mb[ok] - two_disc(cen[ok] * P, list(x) + [P])) / se[ok],
                            [0.0, 0.35, 4.0, 0.5, f0 * P, 0.7], bounds=([-1, 0.05, 1.2, 0, f0 * P - 0.1 * P, 0], [1, 1, 60, 1.9, f0 * P + 0.1 * P, 3]))
        if best is None or res.cost < best[0]:
            best = (res.cost, P, res.x)
    _, P0, x0 = best
    # unbinned refinement with P free; t0 near the middle of the data so it and P are nearly uncorrelated
    tmid = np.median(th)
    t0 = x0[4] + np.round((tmid - x0[4]) / P0) * P0
    # cycle-count aliases: over the baseline T, P and P + k P^2 / T fit the same events one orbit apart at the ends,
    # so each is refined and the lowest chi2 kept; the chi2 gap to the next one says how secure the cycle count is
    T_hr = np.ptp(th)
    fits = []
    for k in range(-3, 4):
        Pk = P0 + k * P0 ** 2 / T_hr
        rk = least_squares(lambda x: (r - two_disc(th, x)) / e, list(x0[:4]) + [t0, x0[5], Pk],
                           bounds=([-1, 0.05, 1.2, 0, t0 - 0.2 * Pk, 0, Pk - 0.5 * P0 ** 2 / T_hr],
                                   [1, 1, 60, 1.9, t0 + 0.2 * Pk, 3, Pk + 0.5 * P0 ** 2 / T_hr]))
        fits.append((2 * rk.cost, k, rk))
    fits.sort(key=lambda z: z[0])
    res = fits[0][2]
    alias_dchi2 = fits[1][0] - fits[0][0]
    J = res.jac
    chi2r = 2 * res.cost / max(len(r) - len(res.x), 1)
    cov = np.linalg.pinv(J.T @ J) * max(chi2r, 1.0)
    err = np.sqrt(np.diag(cov))
    off, q, a, b, t0f, k2, P = res.x
    mb, se, _ = stack(t, r, e, P)
    cen = (np.arange(N_BINS) + 0.5) / N_BINS
    return dict(designation=designation, P_orb_hr=float(P), P_orb_err_hr=float(err[6]), q=float(q), q_err=float(err[1]),
                a_Rp=float(a), a_err=float(err[2]), b_Rp=float(b), b_err=float(err[3]), k2=float(k2), t0_hr=float(t0f),
                offset=float(off), chi2r=float(chi2r), n=len(r), baseline_d=float(np.ptp(t)), P_start=float(P_start),
                grid_best=float(P0), alias_dchi2=float(alias_dchi2), alias_next_P=float(fits[1][2].x[6]),
                aliases=[(float(z[2].x[6]), float(z[0] - fits[0][0])) for z in fits], stack_phase=(((cen * P - t0f) / P) % 1).tolist(), stack_mean=np.nan_to_num(mb).tolist(),
                stack_err=np.nan_to_num(se).tolist(),
                stack_note="stack at the fitted P, phase measured from t0 (secondary in front at phase 0)")


def further_signals(designation, P1, orbit_json, test_periods=()):
    """Search for more companions: remove the primary (per apparition, as everywhere) AND the fitted mutual-event model
    of the first companion, then a Lomb-Scargle of what is left over the pipeline's period range, masking the
    harmonics and small-integer ratios of BOTH P1 and the fitted orbit. Reports the strongest remaining peaks with the
    same false-alarm statistic as the binary search, and the power at any test periods (e.g. an LCDB additional period)."""
    from tessellate import asteroid_store as st
    from astropy.timeseries import LombScargle
    o = json.load(open(orbit_json))
    t, y, e = bs.track_points(st.load_lightcurve(designation, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
    r = bs.two_period(t, y, e, P1, full=True)["_r"]
    par = [o["offset"], o["q"], o["a_Rp"], o["b_Rp"], o["t0_hr"], o["k2"], o["P_orb_hr"]]
    r2 = r - (two_disc(t * 24.0, par) - o["offset"]) - o["offset"]
    T = np.ptp(t)
    app = bs.apparitions(t)
    T_app = max(np.ptp(t[app == a]) for a in np.unique(app))
    fmin = max(24 / bs.P2_RANGE[1], bs.MIN_P2_CYCLES / max(T_app, 1e-3)); fmax = 24 / bs.P2_RANGE[0]
    f = np.arange(fmin, fmax, 1 / (bs.OVERSAMPLE * T))
    bad = np.zeros(len(f), bool)
    for ref in (24 / P1, 24 / o["P_orb_hr"]):
        for mm in range(1, bs.MASK_MN + 1):
            for nn in range(2, bs.MASK_MN + 1):
                bad |= np.abs(f / (mm / nn * ref) - 1) < bs.HARM_TOL
        for kk in range(1, int(fmax / ref) + 2):
            bad |= np.abs(f - kk * ref) < bs.HARM_PEAKS / max(T_app, 1.0)
    pw = LombScargle(t, r2, e).power(f)
    pk = pw.copy(); pk[bad] = 0
    N, M = len(t), max((fmax - fmin) * T, 1.0)
    stat = lambda z: (N - 3) / 2 * np.log10(max(1 - z, 1e-300)) + np.log10(M)
    peaks = []
    for _ in range(5):                           # the five strongest separate peaks
        i = int(np.argmax(pk)); peaks.append((24 / f[i], float(pk[i]), stat(pk[i])))
        pk[np.abs(f - f[i]) < 3 / T] = 0
    tests = []
    for P in test_periods:
        fi = 24 / P
        w = np.abs(f - fi) < 1 / T
        z = float(pw[w].max()) if w.any() else np.nan
        tests.append((P, z, (N - 3) / 2 * np.log10(max(1 - z, 1e-300)) if np.isfinite(z) else np.nan, bool(bad[w].any())))
    return dict(designation=designation, rms_before=float(np.std(r)), rms_after=float(np.std(r2)), peaks=peaks, tests=tests)


def check_signal(designation, P1, orbit_json, P3_start, K3=4):
    """Checks of a further periodic signal found after removing the primary and the first companion's events
    (agreed standard + checks specific to a possible satellite rotation):
      precise period   fitted with a K3-harmonic series at P3 (free), error from the covariance (x sqrt chi2r)
      combinations     distance from n f1 + k f_orb (|n| <= 2, |k| <= 8) in units of its frequency error: a P3 made by
                       rotation x orbit cross terms would sit on one of these
      true period      P3 (double wave) against P3 / 2: does a fit with the odd harmonics of P3 beat a fit without them?
      joint fit        P1 shape per apparition + the event model + the P3 series fitted together: P3 must survive
      flat             residuals after everything, binned in P1 phase and in orbital phase
      recurrence       the P3 peak in independent stretches (apparitions, or the halves of one)
      occultation      if P3 is the SECONDARY's rotation, its signal vanishes while the secondary is hidden behind the
                       primary: the P3 amplitude fitted in-occultation vs out-of-event (a third body would not care)
      amplitude        peak-to-peak of P3, and what it implies for the secondary's own amplitude given q"""
    from tessellate import asteroid_store as st
    from astropy.timeseries import LombScargle
    from scipy.optimize import least_squares
    from scipy.stats import chi2 as chi2d
    o = json.load(open(orbit_json))
    t, y, e = bs.track_points(st.load_lightcurve(designation, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))
    th = t * 24.0
    tp = bs.two_period(t, y, e, P1, full=True)
    par = [o["offset"], o["q"], o["a_Rp"], o["b_Rp"], o["t0_hr"], o["k2"], o["P_orb_hr"]]
    ev = two_disc(th, par) - o["offset"]
    r2 = tp["_r"] - ev
    r2 = r2 - np.average(r2, weights=1 / e ** 2)
    sw = 1 / e
    out = {}
    # precise period
    def fourier(tt, P, c):
        ph = 2 * np.pi * tt / P
        return sum(c[2 * j] * np.cos((j + 1) * ph) + c[2 * j + 1] * np.sin((j + 1) * ph) for j in range(len(c) // 2))
    tc = th - np.median(th)               # centred time: the period and the phase coefficients are then ~uncorrelated
    res = least_squares(lambda x: (r2 - fourier(tc, x[0], x[1:])) * sw, [P3_start] + [0.0] * (2 * K3), x_scale="jac")
    chi2r = 2 * res.cost / max(len(r2) - len(res.x), 1)
    cov = np.linalg.pinv(res.jac.T @ res.jac) * max(chi2r, 1)
    P3, P3e = float(res.x[0]), float(np.sqrt(cov[0, 0]))
    grid = np.linspace(0, P3, 400)
    amp3 = float(np.ptp(fourier(grid, P3, res.x[1:])))
    out.update(P3_hr=P3, P3_err_hr=P3e, P3_amp=amp3, chi2r=float(chi2r))
    # combinations n f1 + k f_orb
    f3, f3e = 24 / P3, 24 * P3e / P3 ** 2
    f1, fo = 24 / P1, 24 / o["P_orb_hr"]
    combos = sorted(((abs(n * f1 + k * fo - f3) / f3e, n, k, 24 / (n * f1 + k * fo)) for n in (-2, -1, 0, 1, 2)
                     for k in range(-8, 9) if n * f1 + k * fo > 0), key=lambda z: z[0])[:3]
    out["nearest_combinations"] = [dict(sigma=float(a), n=n, k=k, P_hr=float(P)) for a, n, k, P in combos]
    # true period: P3 with all harmonics vs only even harmonics (= a fit at P3 / 2)
    even = least_squares(lambda x: (r2 - fourier(th, P3 / 2, x)) * sw, [0.0] * (2 * (K3 // 2)))
    full = least_squares(lambda x: (r2 - fourier(th, P3, x)) * sw, [0.0] * (2 * K3))
    dchi = 2 * (even.cost - full.cost)
    out.update(odd_harmonics_dchi2=float(dchi), odd_harmonics_p=float(chi2d.sf(dchi, 2 * K3 - 2 * (K3 // 2))))
    # joint fit: per-apparition P1 shape + events (fixed) + P3 series
    blocks = [bs.design((t * 24 / P1) % 1, bs.SUB_K) * m_[:, None] for a_, m_, _ in tp["_models"]]
    A1 = np.column_stack(blocks)
    yev = y - ev
    def lin(A):
        c, *_ = np.linalg.lstsq(A * sw[:, None], yev * sw, rcond=None)
        return float(np.sum(((yev - A @ c) * sw) ** 2)), c
    c1, _ = lin(A1)
    A3 = bs.design((th / P3) % 1, K3)[:, 1:]
    c13, cc = lin(np.column_stack([A1, A3]))
    amp_j = float(np.ptp(bs.design(np.linspace(0, 1, 400), K3)[:, 1:] @ cc[-2 * K3:]))
    out.update(joint_dchi2=float(c1 - c13), joint_amp=amp_j, joint_amp_ratio=amp_j / amp3 if amp3 > 0 else np.nan)
    # flat after everything
    rr = r2 - fourier(th, P3, full.x)
    for lab, P in (("P1", P1), ("orbit", o["P_orb_hr"])):
        k = np.clip((((th / P) % 1) * 20).astype(int), 0, 19)
        w = 1 / e ** 2
        num, den = np.bincount(k, w * rr, 20), np.bincount(k, w, 20)
        okb = den > 0
        c2 = float(np.sum(num[okb] ** 2 / den[okb]))
        out[f"flat_{lab}_p"] = float(chi2d.sf(c2, okb.sum()))
    # recurrence
    app = bs.apparitions(t)
    big = [app == a for a in np.unique(app) if (app == a).sum() >= bs.APP_MIN_PTS]
    segs = big if len(big) >= 2 else []
    if len(segs) < 2 and big:
        m_ = max(big, key=lambda x: x.sum()); tm = np.median(t[m_]); segs = [m_ & (t <= tm), m_ & (t > tm)]
    rec = []
    for m_ in segs:
        ts = t[m_]; Ts = np.ptp(ts)
        ff = np.arange(0.5, 24.0, 1 / (5 * Ts))
        pw = LombScargle(ts, r2[m_], e[m_]).power(ff)
        fb = ff[np.argmax(pw)]
        rec.append(dict(best_P=float(24 / fb), matches=bool(min(abs(fb - f3), abs(fb - 2 * f3)) <= 1.5 / Ts)))
    out["recurrence"] = rec
    # occultation test: secondary hidden (behind the primary, overlap) vs out of any event
    phs = (th - o["t0_hr"]) / o["P_orb_hr"]
    thx = 2 * np.pi * phs
    dsep = np.sqrt((o["a_Rp"] * np.sin(thx)) ** 2 + (o["b_Rp"] * np.cos(thx)) ** 2)
    hidden = (np.cos(thx) < 0) & (dsep <= 1 - o["q"])          # secondary entirely behind the primary's disc
    behind_any = (np.cos(thx) < 0) & (dsep < 1 + o["q"])
    out_ev = dsep >= 1 + o["q"]
    def amp_in(m_):
        if m_.sum() < 2 * K3 + 10:
            return np.nan, np.nan
        A = bs.design((th[m_] / P3) % 1, K3)[:, 1:]
        c, *_ = np.linalg.lstsq(A * sw[m_, None], r2[m_] * sw[m_], rcond=None)
        resid = r2[m_] - A @ c
        cv = np.linalg.pinv((A * sw[m_, None]).T @ (A * sw[m_, None])) * max(np.sum((resid * sw[m_]) ** 2) / max(m_.sum() - A.shape[1], 1), 1)
        g = bs.design(np.linspace(0, 1, 400), K3)[:, 1:]
        a = float(np.ptp(g @ c))
        # amplitude error by propagating the coefficient covariance through draws
        draws = np.random.default_rng(0).multivariate_normal(c, cv, 300)
        return a, float(np.std([np.ptp(g @ d_) for d_ in draws]))
    front_any = (np.cos(thx) > 0) & (dsep < 1 + q_ if (q_ := o["q"]) else False)
    # which event is the occultation is not determined, so both are tested: the secondary's own signal must weaken in
    # whichever event hides it
    for lab, m_ in (("out_of_event", out_ev), ("event_B_any_overlap", behind_any), ("event_B_central", hidden),
                    ("event_A_any_overlap", front_any), ("event_A_central", (np.cos(thx) > 0) & (dsep <= 1 - o["q"]))):
        a, ae = amp_in(m_)
        out[f"amp_{lab}"] = a; out[f"amp_{lab}_err"] = ae; out[f"n_{lab}"] = int(m_.sum())
    # the fold of the signal for display: residuals after the primary and the events, phase = MJD hours / P3
    phz = (th / P3) % 1
    kb = np.clip((phz * N_BINS).astype(int), 0, N_BINS - 1)
    w = 1 / e ** 2
    sw_ = np.bincount(kb, w, N_BINS)
    mb = np.where(sw_ > 0, np.bincount(kb, w * r2, N_BINS) / np.where(sw_ > 0, sw_, 1), 0.0)
    sb = np.where(sw_ > 0, 1 / np.sqrt(np.where(sw_ > 0, sw_, 1)), 0.0) * np.sqrt(max(chi2r, 1))
    gp = np.linspace(0, 1, 201)
    out.update(fold_phase=((np.arange(N_BINS) + 0.5) / N_BINS).tolist(), fold_mean=mb.tolist(), fold_err=sb.tolist(),
               model_phase=gp.tolist(), model=fourier(gp * P3, P3, full.x).tolist(),
               fold_note="phase = (MJD x 24 / P3) mod 1")
    q = o["q"]
    frac = q ** 2 / (1 + q ** 2)
    out.update(secondary_light_fraction=frac, implied_secondary_amp=amp3 / frac if frac > 0 else np.nan)
    return out


def run(index, total):
    os.makedirs(OUT, exist_ok=True)
    t = targets().iloc[index::total]
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=10) as pool:
        r = pd.DataFrame(pool.map(one, [row for _, row in t.iterrows()]))
    t.merge(r, on="designation").to_csv(f"{OUT}/stack_{index:03d}.csv", index=False)
    print(f"task {index}: {len(r)}")


def report():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/stack_*.csv")], ignore_index=True)
    d = d[d.get("error", pd.Series(np.nan, index=d.index)).isna()].copy()
    d["expected"] = d.q ** 2 / (1 + d.q ** 2)
    d["exp_over_limit"] = d.expected / d.limit
    d["deepest"] = d[["dip1", "dip2"]].max(axis=1)
    d["detected"] = (d.flat_p < 1e-3) & (d.dip_dbic > 10)
    pd.set_option("display.width", 250)
    cols = ["designation", "bintype", "primary", "orbit", "n_orbits", "q", "a_Dp", "expected", "limit", "deepest", "dip1_phase",
            "dip2_phase", "flat_p", "dip_dbic", "detected", "amp_snr"]
    print(f"{len(d)} known binaries with an LCDB orbit stacked; structure at the orbit period (flat p < 1e-3 and dips "
          f"dbic > 10) in {int(d.detected.sum())}")
    print("\nby expected depth vs the stack's 3-sigma limit:")
    print(d.groupby(pd.cut(d.exp_over_limit, [0, 0.3, 1, 3, 1e9]), observed=True).detected.agg(["size", "sum"]).to_string())
    print("\nlarge primaries (primary period > 4 h or q < 0.15):")
    big = d[(d.primary > 4) | (d.q < 0.15)]
    print(big.sort_values("exp_over_limit", ascending=False)[cols].round(4).head(30).to_string(index=False))
    print("\nall, by expected/limit:")
    print(d.sort_values("exp_over_limit", ascending=False)[cols].round(4).head(60).to_string(index=False))
    d.to_csv(f"{OUT}/known_binary_stacks.csv", index=False)


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(f"{OUT}/known_binary_stacks.csv")
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    for _, r in d.iterrows():
        f = f"{OUT}/stacks/{p.safe_name(r.designation)}.json"
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        c, m, e = np.array(s["centres"]), np.array(s["mean"]), np.array(s["err"])
        m = np.where(e > 0, m, np.nan)                 # empty bins were stored as 0 with error 0
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ax[0].errorbar(c, m, e, fmt="o", color="tab:red", ms=4, label="All apparitions stacked")
        g = np.linspace(0, 1, 400)
        ax[0].plot(g, bs.dips(g, *s["dip"]), color="k", lw=1.2, label="Two-dip fit")
        if np.isfinite(r.expected):
            ax[0].axhline(-r.expected, color="tab:blue", ls="--", lw=1, label=f"Deepest total event, q = {r.q:.2f}")
        ax[0].axhline(-r.limit, color="0.5", ls=":", lw=1, label="3-sigma stack limit")
        ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set_xlabel("Orbital phase")
        ax[0].set_ylabel("Residual relative flux")
        ax[0].set_title(f"Residuals stacked at the LCDB orbit {r.orbit:.3f} h ({r.n_orbits:.0f} orbits covered)", fontsize=10)
        ax[0].legend(fontsize=7, loc="lower left")
        cols = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:cyan", "tab:brown"]
        for i, pa in enumerate(s["per_app"]):
            ea = np.array(pa["err"])
            ax[1].errorbar(c + 0.004 * i, np.where(ea > 0, pa["mean"], np.nan), ea, fmt="o", ms=3, color=cols[i % len(cols)],
                           label=f"Apparition MJD {pa['t_mid']:.0f}")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].set_xlabel("Orbital phase")
        ax[1].set_ylabel("Residual relative flux")
        ax[1].set_title("Each apparition stacked separately (events should recur)", fontsize=10)
        if s["per_app"]:
            ax[1].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"{r.designation}: LCDB primary {r.primary:.3f} h, orbit {r.orbit:.3f} h, Ds/Dp {r.q:.2f}, a/Dp {r.a_Dp:.2f}; "
                     f"flat p = {r.flat_p:.1g}, dip dBIC {r.dip_dbic:.0f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{OUT}/figures/{'DET' if r.detected else 'non'}__{r.exp_over_limit:.2f}__{p.safe_name(r.designation)}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"figures -> {OUT}/figures")


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "report":
        report()
    elif a == "control":
        control(int(sys.argv[2]), int(sys.argv[3]))
    elif a == "control_report":
        control_report()
    elif a == "further":                  # further "<designation>" <P1_hr> [test periods...]
        out = further_signals(sys.argv[2], float(sys.argv[3]), f"{OUT}/orbits/{p.safe_name(sys.argv[2])}.json",
                              [float(x) for x in sys.argv[4:]])
        json.dump(out, open(f"{OUT}/orbits/{p.safe_name(sys.argv[2])}_further.json", "w"))
        print(f"{out['designation']}: residual rms {out['rms_before']:.4f} -> {out['rms_after']:.4f} after the events")
        print("strongest remaining peaks (period h, power, statistic; the binary-search null thresholds are about -25 at "
              "1-3 h, -41 at 3-10 h, -51 at 10-30 h):")
        for P_, z, s_ in out["peaks"]:
            print(f"   {P_:9.4f} h  power {z:.4f}  statistic {s_:.1f}")
        for P_, z, s_, m_ in out["tests"]:
            print(f"   test period {P_:.4f} h: power {z:.4f}, single-frequency log10 FAP {s_:.1f}" + (" (inside a mask)" if m_ else ""))
    elif a == "check3":                   # check3 "<designation>" <P1_hr> <P3_start_hr>
        out = check_signal(sys.argv[2], float(sys.argv[3]), f"{OUT}/orbits/{p.safe_name(sys.argv[2])}.json", float(sys.argv[4]))
        json.dump(out, open(f"{OUT}/orbits/{p.safe_name(sys.argv[2])}_check3.json", "w"))
        for k, v in out.items():
            print(f"  {k}: {v}")
    elif a == "fitorb":                   # fitorb "<designation>" <P_start_hr> <P1_hr>
        out = fit_orbit(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
        os.makedirs(f"{OUT}/orbits", exist_ok=True)
        json.dump(out, open(f"{OUT}/orbits/{p.safe_name(sys.argv[2])}.json", "w"))
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items() if not k.startswith("stack")})
    else:
        plots()
