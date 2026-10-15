"""Compare the accepted period with the hybrid Vavilov period (vavilov_periods.determine) for named labelled objects:
pipeline label-style figure: light curve, 1- and 2-term Lomb-Scargle periodograms with the periods marked, then for
the accepted period (best-BIC order) and the Vavilov period (its order) the two-cycle fold with the model (per-visit
offsets removed, per-apparition amplitude) and the 15-bin binned fold. Writes <DIR>/compare/<label figure name>.png
(LABEL_DIR set) or <DIR>/compare/<name>.png Reads the cleaned points
(pre-clipped, per-visit normalised; columns mjd, rel_flux, rel_flux_err, visit) from <DIR>/<name>_points.csv and the
accepted and hybrid results from <DIR>/vavilov_results.csv (as written by vavilov_test.py); writes <DIR>/<name>.png
    python3 -u vavilov_compare_plot.py DIR ["(12056) Yoshigeru" ...]     # all objects in the results file by default
"""
import matplotlib
matplotlib.use("Agg")
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import vavilov_periods as vp

FOLD_N_BINS = 15            # sector_report_pipeline values
MIN_PERIOD_DAYS = 0.04
MAX_PERIOD_DAYS = 40.0
SAMPLES_PER_PEAK = 10
LABEL_DIR = os.environ.get("LABEL_DIR")     # label figures, to name each comparison after its label figure


def fit_at(t, y, e, vis_idx, n_vis, app_idx, period_hr, order=None):
    """Fit at the period that, after the doubling rule, gives this rotation period: the best-BIC order at P and,
    for single-peaked curves, at P / 2. With order given, that order at P."""
    if order:
        return vp.fit(t, y, e, vis_idx, n_vis, 24 / period_hr, order, app_idx)
    fits = [vp.fit(t, y, e, vis_idx, n_vis, 24 / P, n, app_idx) for P in (period_hr, period_hr / 2)
            for n in range(1, vp.MAX_ORDER + 1) if n * 24 / P < 1 / (2 * vp.CADENCE_D)]
    fits = [r for r in fits if abs(vp.rotation_period_hr(r) / period_hr - 1) < 0.02]
    return min(fits, key=lambda r: r["bic"])


def fold_columns(ax_raw, ax_bin, t, y, e, vis_idx, n_vis, r, period_hr, title, colour):
    """Pipeline-style folds over two cycles: all points with the model (per-apparition scale), and 15-bin means
    with y-limits set by the binned data."""
    yc = y - r["coef"][:n_vis][vis_idx] + 1          # visit offsets removed, back on a relative-flux scale of 1
    ph = (t * 24 / period_hr) % 1                     # same phase origin as the fitted model (t = 0)
    k = r["keep"]
    for sh in (0, 1):
        ax_raw.errorbar(ph[k] + sh, yc[k], e[k], fmt="o", ms=2, color=colour, alpha=0.15, elinewidth=0.4)
    g = np.linspace(0, 2, 1000)
    c = r["coef"][n_vis:]
    f_ratio = 24 / period_hr / r["f"]
    shape = sum(c[2 * j - 2] * np.cos(2 * np.pi * j * g / f_ratio) + c[2 * j - 1] * np.sin(2 * np.pi * j * g / f_ratio)
                for j in range(1, r["n"] + 1))
    scales = np.unique(np.round(r["point_scale"], 3))
    for sc in scales:
        ax_raw.plot(g, 1 + sc * shape, color="red", lw=1.5,
                    label=f"Order {r['n']} model" + (f", scale {sc:.2f}" if len(scales) > 1 else ""))
    lo, hi = np.nanpercentile(yc[k], [0.2, 99.8])
    ax_raw.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax_raw.set_xlim(0, 2)
    ax_raw.axvline(1, color="0.5", ls=":", lw=0.8)
    ax_raw.set_xlabel(f"Phase (P = {period_hr:.3f} h)")
    ax_raw.set_ylabel("Relative flux")
    ax_raw.set_title(f"{title}: P = {period_hr:.3f} h\nBIC = {r['bic']:.1f}, reduced chi2 = {r['s2']:.2f}", fontsize=9)
    ax_raw.legend(fontsize=7, loc="upper right")
    edges = np.linspace(0, 1, FOLD_N_BINS + 1)
    idx = np.digitize(ph[k], edges) - 1
    mb = np.array([yc[k][idx == i].mean() if (idx == i).sum() else np.nan for i in range(FOLD_N_BINS)])
    sb = np.array([yc[k][idx == i].std(ddof=1) / np.sqrt((idx == i).sum()) if (idx == i).sum() > 1 else np.nan
                   for i in range(FOLD_N_BINS)])
    xc = 0.5 * (edges[1:] + edges[:-1])
    for sh in (0, 1):
        ax_bin.errorbar(xc + sh, mb, sb, fmt="o-", ms=4, color=colour, capsize=2)
    ax_bin.axvline(1, color="0.5", ls=":", lw=0.8)
    ax_bin.set_xlim(0, 2)
    ax_bin.set_xlabel(f"Phase (P = {period_hr:.3f} h), 15-bin means")
    ax_bin.set_ylabel("Relative flux (binned)")


def plot_one(args):
    OUT, d, row = args
    name = d.replace('(', '').replace(')', '').replace(' ', '_')
    tr = pd.read_csv(f"{OUT}/{name}_points.csv").sort_values("mjd")
    t, y, e, v = tr.mjd.values, tr.rel_flux.values, tr.rel_flux_err.values, tr.visit.values
    _, vis_idx = np.unique(v, return_inverse=True)
    n_vis = vis_idx.max() + 1
    app_idx = vp.apparitions(t)
    P_lab, P_vav, P_alt = row["labelled_period_hr"], row["vav_period_hr"], row["vav_alt_period_hr"]
    fr, pw = LombScargle(t, y, e).autopower(minimum_frequency=1 / MAX_PERIOD_DAYS, maximum_frequency=1 / MIN_PERIOD_DAYS,
                                            samples_per_peak=SAMPLES_PER_PEAK)
    pw2 = LombScargle(t, y, e, nterms=2).power(fr)
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.1, 1])
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.errorbar(t, y, e, fmt="o", ms=2, color="tab:blue", alpha=0.2, elinewidth=0.4)
    ax_t.axhline(1, color="k", ls="--", lw=0.8)
    lo, hi = np.nanpercentile(y, [0.2, 99.8])
    ax_t.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    ax_t.set_xlabel("MJD")
    ax_t.set_ylabel("Relative flux")
    ax_t.set_title(f"{d}: {len(t)} points, {n_vis} visits, {app_idx.max() + 1} apparitions, label {row['label_final']}",
                   fontsize=10)
    ax_p = fig.add_subplot(gs[1, :])
    ax_p.plot(24 / fr, pw, color="0.5", lw=0.8, label="1-term")
    ax_p.plot(24 / fr, pw2, color="tab:orange", lw=0.8, label="2-term")
    ax_p.axvline(P_lab, color="tab:blue", ls="--", lw=1.2, label=f"Periodogram P = {P_lab:.2f} h")
    ax_p.axvline(P_vav, color="tab:red", ls="--", lw=1.2, label=f"New method P = {P_vav:.2f} h (order {int(row['vav_order'])})")
    if np.isfinite(P_alt):
        ax_p.axvline(P_alt, color="tab:green", ls=":", lw=1.2,
                     label=f"New method harmonic alternative P = {P_alt:.2f} h (dBIC {row['vav_alt_dbic']:.1f})")
    ax_p.set_xscale("log")
    ax_p.set_xlabel("Period (h)")
    ax_p.set_ylabel("Lomb-Scargle power")
    ax_p.legend(fontsize=8)
    r1 = fit_at(t, y, e, vis_idx, n_vis, app_idx, P_lab)
    f_ad = 24 / P_vav * (2 if row["vav_doubled"] else 1)
    r2 = vp.fit(t, y, e, vis_idx, n_vis, f_ad, int(row["vav_order"]), app_idx)
    fold_columns(fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[3, 0]), t, y, e, vis_idx, n_vis, r1, P_lab,
                 "Periodogram period", "tab:blue")
    fold_columns(fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[3, 1]), t, y, e, vis_idx, n_vis, r2, P_vav,
                 "New method period", "tab:red")
    label = glob.glob(f"{LABEL_DIR}/**/*_{name}.png", recursive=True) if LABEL_DIR else []
    title = os.path.basename(label[0])[:-4] if label else name      # the label figure's name: score_period_designation
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(f"{OUT}/compare", exist_ok=True)
    fn = f"{OUT}/compare/{title}.png"
    fig.savefig(fn, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return (f"{d}: accepted {P_lab:.3f} h BIC {r1['bic']:.1f} (n={r1['n']}); Vavilov {P_vav:.3f} h (n={int(row['vav_order'])}) "
            f"BIC {r2['bic']:.1f}, dBIC (accepted - Vavilov) {r1['bic'] - r2['bic']:.1f}")

def main():
    OUT = sys.argv[1]
    vt = pd.read_csv(f"{OUT}/vavilov_results.csv").drop_duplicates("designation").set_index("designation")
    names = sys.argv[2:] or list(vt.index)
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as ex:
        for line in ex.map(plot_one, [(OUT, d, vt.loc[d]) for d in names]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
