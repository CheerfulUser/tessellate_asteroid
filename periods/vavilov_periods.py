"""Period determination after Vavilov, Eggl & Greenstreet 2026 (arXiv 2610.07584), adapted to dense TESS photometry
and run on candidate frequencies only (hybrid): a 1-term Lomb-Scargle on the pre-clipped points proposes the
candidates, and full multi-harmonic fits decide between them.

Model (relative flux, per-visit normalised as in sector_report_pipeline): y = offset per visit + s_apparition x Fourier
series of order n, where s is a per-apparition amplitude scale (1 for the best-sampled apparition; apparitions with
fewer than APP_MIN_POINTS points or spanning less than one trial period share the reference scale): viewing geometry
changes the amplitude between apparitions, and one amplitude across them leaves the weighted fit following the precise
apparition and the binned fold the noisy one. Per trial frequency and order:
  - weighted least squares, iterative rejection of points beyond THETA x the fit's own scatter;
  - BIC = N_u ln(chi2 / N_u) + R + p ln N, with R the clustered rejection penalty (rejected points within
    CLUSTER_H hours form one cluster; theta^2 per point up to K0, logarithmic beyond).
Order and period: the lowest BIC over all candidates and orders. The paper's ladder of F / Vuong tests at a fixed 95 %
adds harmonics on almost every object at TESS point counts (thousands of correlated points), and BIC's ln N penalty
does not. The grid scan uses one amplitude; the best grid points are refitted with the apparition scales.
Rotation period: the fitted period, doubled when the curve has one derivative maximum per cycle.
Harmonic alternative: the best other candidate whose rotation period is a harmonic ratio (2, 3, 3/2 and inverses) of
the adopted one, with its BIC difference: orders n at n x P0 all contain a dominant P0 harmonic, so a 2:3 choice
between them (double- vs triple-peaked) can rest on a small BIC margin.
Rivals: candidates at non-harmonic periods; the number within DBIC_C and the adopted period's Vuong probability of
beating them. Significance: F-test of the adopted model against offsets only.

Simplifications against the paper: the Vuong correction uses the parameter-count and rejection-penalty terms only
(no overlapping-model case); correlated noise is handled only through the per-visit offsets and the error rescaling.
"""
import numpy as np
from scipy import stats

THETA = 3.0
CLUSTER_H = 1.0
K0 = 4
ALPHA = K0
DBIC_C = 20.0
MAX_ORDER = 6
N_CANDIDATES = 6
LOCAL_STEPS = 21          # refinement grid points across +/- 2 peak widths
APPARITION_GAP_D = 60.0   # visits separated by more than this belong to different apparitions
ALS_STEPS = 4             # alternating least-squares steps for the per-apparition amplitude scales
APP_MIN_POINTS = 300      # an apparition gets its own amplitude only with this many points spanning >= 1 trial period
CADENCE_D = 10 / 1440     # TESS sectors 27-55 full-frame cadence


def _design(t, vis_idx, n_vis, f, n):
    A = np.zeros((len(t), n_vis + 2 * n))
    A[np.arange(len(t)), vis_idx] = 1.0
    for k in range(1, n + 1):
        A[:, n_vis + 2 * k - 2] = np.cos(2 * np.pi * k * f * t)
        A[:, n_vis + 2 * k - 1] = np.sin(2 * np.pi * k * f * t)
    return A


def _rejection_penalty(t, rejected):
    tr = np.sort(t[rejected])
    if not len(tr):
        return 0.0
    clusters, start, k = [], tr[0], 0
    for x in tr:
        if x - start <= CLUSTER_H / 24:
            k += 1
        else:
            clusters.append(k); start, k = x, 1
    clusters.append(k)
    return THETA ** 2 * sum(c if c <= K0 else K0 + ALPHA * np.log(c / K0) for c in clusters)


def apparitions(t):
    """Index of the apparition each point belongs to: runs of points with no gap longer than APPARITION_GAP_D."""
    o = np.argsort(t)
    idx = np.empty(len(t), int)
    idx[o] = np.concatenate([[0], np.cumsum(np.diff(t[o]) > APPARITION_GAP_D)])
    return idx


def fit(t, y, e, vis_idx, n_vis, f, n, app_idx=None, max_iter=10):
    """Fit with iterative rejection. With app_idx (more than one apparition) the Fourier part is multiplied by a
    per-apparition amplitude scale (best-sampled apparition fixed at 1), solved by alternating least squares.
    Returns dict(chi2, nu, p, bic, keep, resid (normalised by e), s2, coef, scales, f, n, R)."""
    A0 = _design(t, vis_idx, n_vis, f, n)
    w = 1 / e
    n_app = 1
    if app_idx is not None and n > 0:
        cnt = np.bincount(app_idx)
        span = np.array([np.ptp(t[app_idx == a]) for a in range(len(cnt))])
        free = np.flatnonzero((cnt >= APP_MIN_POINTS) & (span >= 1 / f))   # apparitions that constrain an amplitude
        if len(free) > 1:
            ref = free[np.argmax(cnt[free])]
            remap = np.full(len(cnt), -1)
            remap[free] = np.arange(len(free))
            remap[remap < 0] = remap[ref]                # the rest share the reference amplitude
            app_idx, n_app, ref = remap[app_idx], len(free), remap[ref]
    sc = np.ones(n_app)
    keep = np.ones(len(t), bool)
    for _ in range(max_iter):
        for k in range(ALS_STEPS if n_app > 1 else 1):
            A = A0.copy()
            if n_app > 1:
                A[:, n_vis:] *= sc[app_idx][:, None]
            c, *_ = np.linalg.lstsq(A[keep] * w[keep, None], y[keep] * w[keep], rcond=None)
            if n_app > 1 and k < ALS_STEPS - 1:
                F = A0[:, n_vis:] @ c[n_vis:]
                r0 = y - A0[:, :n_vis] @ c[:n_vis]
                for a in range(n_app):
                    m = keep & (app_idx == a)
                    den = np.sum((w[m] * F[m]) ** 2)
                    sc[a] = max(np.sum(w[m] ** 2 * r0[m] * F[m]) / den, 0.0) if den > 0 else 1.0
                sc = sc / sc[ref] if sc[ref] > 0 else np.ones(n_app)
        r = (y - A @ c) / e
        p = A0.shape[1] + n_app - 1
        s = np.sqrt(np.sum(r[keep] ** 2) / max(keep.sum() - p, 1))
        new = np.abs(r) < THETA * s
        if (new == keep).all():
            break
        keep = new
    chi2, nu = float(np.sum(r[keep] ** 2)), int(keep.sum())
    R = _rejection_penalty(t, ~keep)
    bic = nu * np.log(chi2 / nu) + R + p * np.log(len(t))
    return dict(chi2=chi2, nu=nu, p=p, bic=float(bic), keep=keep, resid=r, s2=chi2 / max(nu - p, 1), coef=c, scales=sc,
                point_scale=sc[app_idx] if n_app > 1 else np.ones(len(t)), f=f, n=n, R=R)


def vuong_prob(a, b, t):
    """P(model a better than model b), Vuong 1989 as in Vavilov et al. Sect. 2.5 (simplified correction)."""
    both = a["keep"] & b["keep"]
    m = int(both.sum())
    if m < max(a["p"], b["p"]) + 5:
        return 0.5
    s2 = min(a["s2"], b["s2"])
    d = (b["resid"][both] ** 2 - a["resid"][both] ** 2) / (2 * s2)
    ra = _rejection_penalty(t, a["keep"] & ~b["keep"]) / a["s2"] * s2 if a["s2"] > 0 else 0.0
    rb = _rejection_penalty(t, b["keep"] & ~a["keep"]) / b["s2"] * s2 if b["s2"] > 0 else 0.0
    C = (a["p"] - b["p"]) * np.log(m) + (ra - rb)
    sd = d.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.5
    return float(stats.norm.cdf((d.sum() - C / 2) / (np.sqrt(m) * sd)))


def ftest_prob(lo, hi):
    """P(higher-order hi better than lo) at the same frequency (nested), on the points both fits accept: comparing
    each fit's chi2 over its own accepted set rewards the higher order for every extra point it rejects."""
    dp = hi["p"] - lo["p"]
    both = lo["keep"] & hi["keep"]
    m = int(both.sum())
    if dp <= 0 or m - hi["p"] <= 0:
        return 0.0
    c_lo, c_hi = np.sum(lo["resid"][both] ** 2), np.sum(hi["resid"][both] ** 2)
    F = ((c_lo - c_hi) / dp) / (c_hi / (m - hi["p"]))
    return float(stats.f.cdf(max(F, 0), dp, m - hi["p"]))


def maxima_of_derivative(res, n_grid=1000):
    ph = np.arange(n_grid) / n_grid
    c = res["coef"][-2 * res["n"]:]
    m = sum(c[2 * k - 2] * np.cos(2 * np.pi * k * ph) + c[2 * k - 1] * np.sin(2 * np.pi * k * ph) for k in range(1, res["n"] + 1))
    dm = np.gradient(m)
    return int(np.sum((dm > np.roll(dm, 1)) & (dm >= np.roll(dm, -1))))


def candidates(t, y, e, min_period_d, max_period_d, samples_per_peak):
    """Candidate frequencies (cycles/day) from a 1-term Lomb-Scargle: the N_CANDIDATES strongest peaks at least three
    peak widths apart, plus up to 4 families >= 5 % apart in frequency. Returns (candidates, peak width)."""
    from astropy.timeseries import LombScargle
    fr, pw = LombScargle(t, y, e).autopower(minimum_frequency=1 / max_period_d, maximum_frequency=1 / min_period_d,
                                            samples_per_peak=samples_per_peak)
    width = 1 / np.ptp(t)
    cands, fams = [], []
    order = np.argsort(pw)[::-1]
    for i in order:                                 # strongest peaks (side lobes of one family included)
        if all(abs(fr[i] - c) > 3 * width for c in cands):
            cands.append(fr[i])
        if len(cands) == N_CANDIDATES:
            break
    for i in order:                                 # plus up to 4 distinct families, >= 5 % apart in frequency
        if all(abs(fr[i] / c - 1) > 0.05 for c in cands + fams):
            fams.append(fr[i])
        if len(fams) == 4:
            break
    return cands + fams, width


def rotation_period_hr(res):
    """Fitted period, doubled when the curve has one derivative maximum per cycle (single-peaked)."""
    return (2 if maxima_of_derivative(res) == 1 else 1) * 24 / res["f"]


def determine(t, y, e, visit, f_candidates, peak_width):
    """Hybrid Vavilov-style determination over candidate frequencies (cycles/day)."""
    vis_codes, vis_idx = np.unique(visit, return_inverse=True)
    n_vis = len(vis_codes)
    app_idx = apparitions(t)
    n_app = int(app_idx.max()) + 1
    f_ny = 1 / (2 * CADENCE_D)
    cand = []
    for f0 in f_candidates:
        for h in (1.0, 0.5, 2.0):                     # the peak and its half / double rotation frequency
            f = f0 * h
            if 0 < f < f_ny and all(abs(f - c) > peak_width for c in cand):
                cand.append(f)
    best = {}                                          # (candidate index, order) -> best fit on the local grid
    for i, fc in enumerate(cand):
        grid = fc + np.linspace(-2, 2, LOCAL_STEPS) * peak_width
        grid = grid[grid > 0]
        for n in range(1, MAX_ORDER + 1):
            if n * grid.max() >= f_ny:
                continue
            fits = [fit(t, y, e, vis_idx, n_vis, f, n) for f in grid]       # scan with one amplitude (fast)
            j = int(np.argmin([r["bic"] for r in fits]))
            if n_app > 1:                              # refit the best grid point and its neighbours with scales
                fits = [fit(t, y, e, vis_idx, n_vis, grid[k], n, app_idx) for k in range(max(j - 1, 0), min(j + 2, len(grid)))]
            best[(i, n)] = min(fits, key=lambda r: r["bic"])
    if not best:
        return None
    a = min(best.values(), key=lambda r: r["bic"])     # order and period by BIC
    period_hr = rotation_period_hr(a)
    per_cand = {}                                      # each candidate's best order
    for (i, n), r in best.items():
        if i not in per_cand or r["bic"] < per_cand[i]["bic"]:
            per_cand[i] = r
    others = [r for r in per_cand.values() if abs(r["f"] - a["f"]) >= peak_width]
    harm_q = (0.5, 2, 1 / 3, 3, 2 / 3, 1.5)
    alts, rivals = [], []
    for r in others:
        P = rotation_period_hr(r)
        if any(abs(P / period_hr / q - 1) < 0.02 for q in harm_q):
            alts.append((r, P))
        elif abs(P / period_hr - 1) >= 0.02:
            rivals.append(r)
    alt = min(alts, key=lambda x: x[0]["bic"]) if alts else None
    close = [r for r in rivals if r["bic"] - a["bic"] < DBIC_C]
    flat = fit(t, y, e, vis_idx, n_vis, a["f"], 0)
    F = ((flat["chi2"] - a["chi2"]) / (a["p"] - flat["p"])) / (a["chi2"] / max(a["nu"] - a["p"], 1))
    log10F_flat = float(np.log10(F)) if F > 0 else -np.inf   # p-values underflow for strong signals; F does not
    sc = a["scales"]
    return dict(vav_period_hr=float(period_hr), vav_order=int(a["n"]), vav_doubled=bool(maxima_of_derivative(a) == 1),
                vav_alt_period_hr=float(alt[1]) if alt else np.nan, vav_alt_order=int(alt[0]["n"]) if alt else -1,
                vav_alt_dbic=float(alt[0]["bic"] - a["bic"]) if alt else np.nan,
                vav_n_rivals=len(close), vav_best_rival_dbic=float(min(r["bic"] for r in rivals) - a["bic"]) if rivals else np.nan,
                vav_p_beats_rivals=float(np.prod([vuong_prob(a, r, t) for r in close])) if close else 1.0,
                vav_log10F_flat=log10F_flat, vav_chi2_red=float(a["s2"]), vav_frac_rejected=float(1 - a["nu"] / len(t)),
                vav_n_apparitions=n_app, vav_scale_min=float(sc.min()), vav_scale_max=float(sc.max()),
                vav_n_candidates=len(cand))
