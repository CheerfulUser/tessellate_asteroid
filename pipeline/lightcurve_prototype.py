"""Prototype: period search and phase-angle coverage assessment for a single
high-SNR TESS asteroid lightcurve, as a precursor to full shape/spin
inversion (a separate downstream step, not attempted here).

Reads the ephemeris (*_Asteroids.parquet) and per-frame PSF photometry
(*_AsteroidPSFPhotometry.parquet) for one target, merges them on mjd,
runs a Lomb-Scargle period search, phase-folds the lightcurve, and reports
whether phase-angle coverage is adequate for convex inversion.

Usage:
    python lightcurve_prototype.py <cut_path> <designation> <sector>
"""
import matplotlib
matplotlib.use('Agg')
import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _fourier_design(phase, n_harm):
    cols = [np.ones_like(phase)]
    for k in range(1, n_harm + 1):
        cols.append(np.sin(2 * np.pi * k * phase))
        cols.append(np.cos(2 * np.pi * k * phase))
    return np.column_stack(cols)


def _fourier_reduced_chi2(t, flux, ferr, period_hr, n_harm=2):
    """Least-squares fit an n_harm-order Fourier series to the phase-folded
    lightcurve and return its reduced chi-square -- a period whose folded
    shape is genuinely periodic (vs an alias/harmonic of the true period)
    should fit much better than one that isn't.
    """
    phase = (t * 24.0 / period_hr) % 1.0
    A = _fourier_design(phase, n_harm)
    w = 1.0 / ferr
    Aw = A * w[:, None]
    fw = flux * w
    coeffs, *_ = np.linalg.lstsq(Aw, fw, rcond=None)
    resid = flux - A @ coeffs
    chi2 = np.sum((resid / ferr) ** 2)
    dof = len(flux) - A.shape[1]
    return chi2 / dof


def disambiguate_period(t, flux, ferr, ls_best_period_hr, n_harm=4, ls_model=None, min_power_ratio=0.5,
                          max_period_hr=None, min_chi2_improvement=0.05):
    """Asteroid lightcurves are very often double-humped (near-180-deg
    rotational symmetry from an elongated shape), so a raw Lomb-Scargle
    periodogram frequently locks onto the first harmonic (half the true
    rotation period) instead of the fundamental. Check the LS best period
    against its 1/2x, 2x, 1/3x and 3x multiples by comparing how well each
    folds the lightcurve (reduced chi-square of a Fourier fit), and prefer
    the best-fitting candidate.

    n_harm=4 (not 2): a 2-harmonic fit is a pure double-sinusoid and often
    isn't flexible enough to represent a real asteroid's actual folded shape
    (e.g. asymmetric hump depths/widths) even at the TRUE period -- verified
    on (348) May's real Sector 32 photometry, where n_harm=2 wrongly favoured
    the half-period alias (chi2 33 vs 52) while n_harm=4+ correctly and
    decisively favours the true, literature-matched period (chi2 14 vs 32),
    consistent through n_harm=8. Too few harmonics is a real failure mode,
    not just a soft preference.

    ls_model, min_power_ratio (deprecated, kept only for call-site compatibility -- NOT
    applied): an earlier version of this function disqualified any harmonic candidate whose
    own nterms=1 periodogram power was below min_power_ratio of the naive peak's, meant to
    catch a long-period alias (2P/3P) "connecting the dots" of sparse, widely-gapped visits
    with no real periodogram power (e.g. (9946) 1990 ON2). REMOVED after it was found to
    silently veto genuine double-peaked corrections at scale: a TRUE double-humped period's
    own nterms=1 power is inherently low (that's the mechanism of the alias, not evidence
    against it) -- confirmed on (22) Kalliope, a well-documented 4.148hr rotator, where the
    chi2 comparison correctly and decisively preferred the true period (chi2=446 vs 1805 for
    the naive half-period alias) but this guard blocked it anyway (true period's own power
    was only 11% of the naive peak's), and systematically on 710+ objects in a full production
    run landing within 1% of exactly half their literature period, all mislabelled
    double_peaked=False. Checked the guard was never actually doing its intended job either:
    the sparse-visit overfitting cases it was meant to stop ((4942) Munroe, (11921)
    Mitamasahiro, ...) have MODERATE power ratios (0.6+) that never tripped the >=0.5 floor in
    the first place -- it only ever fired on genuinely low-power cases, which are exactly the
    real double-peaked corrections. That failure mode is instead caught downstream, after a
    period is adopted, by the multi-branch is_reliable rule's loo_worst_chi2 check (leave-one-
    visit-out cross-validation directly measures "do widely-separated visits actually agree",
    which is the real signature of a wrong period -- see sector_report_pipeline_v3.py) rather
    than by restricting which period can be adopted in the first place.

    max_period_hr (optional): the periodogram search band's own upper bound (e.g. the
    caller's MAX_PERIOD_DAYS*24). Found the hard way on real data: EVERY one of 12 flagged
    wrong-period objects had its winning 2P/3P candidate land outside this band (63-131 hr,
    vs a 48 hr search cap) -- ls_model.power() at a frequency the periodogram was never
    actually evaluated over does not measure real periodicity there, it can pick up
    red-noise/trend leakage and return a value that misleadingly LOOKS like a real peak
    (confirmed on (4942) Munroe: the out-of-band 2P candidate at 63.5 hr read power=0.127,
    higher than the true in-band peak's 0.105). A candidate whose period falls outside the
    search band is disqualified outright, before the power-ratio comparison even runs.

    min_chi2_improvement: a non-'P' candidate must beat P's own chi2 by at least this relative
    margin ((chi2_P - chi2_cand) / chi2_P) to be allowed to win at all -- otherwise falls back
    to P. Added after finding (7701) Zrzavy: chi2_P=2.89 vs chi2_2P=2.82, a 2.2% "improvement"
    entirely consistent with noise, decided the same way as Kalliope's genuine 75% improvement
    once the min_power_ratio guard (which had blocked this class of case, but also incorrectly
    blocked genuine doublings like Kalliope's) was removed.

    CAUTION on the threshold value: an initial 0.15 (15%) was picked from only two reference
    points (Zrzavy's 2.2% bad near-tie vs Kalliope's 75% genuine correction) and turned out to
    be far too strict -- confirmed on real full-survey production data: 70% of the reliable,
    literature-matched population (2131/3038 objects) was landing at exactly half its published
    period, including well-known asteroids like (175) Andromache (margin=7.9%, needs 2P per
    literature) and (130) Elektra (margin=4.1%, needs 2P per literature) -- both genuine
    doublings with real but modest margins, both wrongly blocked by the 15% floor. Lowered to
    0.03 (3%): still clears Zrzavy's 2.2% near-tie, still passes Andromache/Elektra. This value
    sits in a narrow, only-two-known-points gap (2.2% bad, 4.1% good) -- a genuine near-tie
    with a margin between roughly 2-4% could still go either way; there is no evidence yet of
    such a case, but don't treat 0.03 as more validated than it is. Objects that DO confidently
    but wrongly favour a long harmonic by chi2 (e.g. (4942) Munroe at 45%, (11921) Mitamasahiro
    at 41% -- not near-ties) are unaffected by this margin either way and still need to be
    caught downstream by is_reliable's loo_worst_chi2/n_well_covered_cycles checks, not by
    disambiguation alone -- confirmed (7701) Zrzavy itself still needs the margin specifically
    because its OWN downstream stats at the wrong period (aov_F=470, amp_snr=11, loo=7.4) are
    clean enough to pass branch D regardless of which candidate is adopted -- downstream QC
    does NOT reliably catch this case, unlike Munroe/Anshan/Mitamasahiro.
    """
    # 3P and P/3 REMOVED: a three-peaked lightcurve requires genuine three-fold shape symmetry,
    # which has no physical basis in the asteroid population. Elongation is the common shape
    # asymmetry, so the second harmonic dominates and the periodogram locks onto P or 2P.
    # Confirmed against LCDB: 3P is the true answer for only 43 of 5158 labelled objects and
    # P/3 for 23 (1.3% combined, some of which are probably LCDB errors), while offering them
    # as candidates cost real accuracy -- chi2 selection scores 67.6% over the full five-way
    # set versus 82.2% restricted to the physical three. (75738) 2000 AF145 is a concrete
    # casualty: its LS peak is 48.64hr, LCDB says 98.08hr = 2P, and chi2 picked 3P = 145.9hr.
    candidates = {
        'P': ls_best_period_hr,
        '2P': ls_best_period_hr * 2,
        'P/2': ls_best_period_hr / 2,
    }
    results = {name: _fourier_reduced_chi2(t, flux, ferr, p, n_harm) for name, p in candidates.items()}

    if max_period_hr is not None:
        in_band = {name for name, p in candidates.items() if name == 'P' or p <= max_period_hr}
        results = {name: chi2 for name, chi2 in results.items() if name in in_band}

    # DEFAULT TO 2P, and let chi2 override only with a decisive margin.
    #
    # This inverts the previous logic, which defaulted to P and let chi2 override on any margin
    # at all. That arrangement chose the harmonic correctly only 16.1% of the time in
    # production, because asteroid lightcurves are predominantly double-peaked: an elongated
    # body presents two faces per rotation, so the periodogram locks onto the second harmonic
    # at HALF the true period. Measured against LCDB, the true period is 2x the naive LS peak
    # for 95.3% of labelled objects.
    #
    # Accuracy over 5092 objects with a clean harmonic relation to LCDB truth:
    #     chi2 argmin over {P/2,P,2P}        83.3%
    #     always 2P                          95.5%
    #     2P unless chi2 better by >5%       95.9%   <- this rule
    # Flat between 2% and 50%, so the margin is not finely tuned. chi2 still earns its place:
    # it recovers cases like (5118) Elnapoul, whose LS peak already sits at 2x the true period
    # (LCDB 36.14hr secure), where blind doubling would land four times too long.
    if 'P' in results:
        base = '2P' if '2P' in results else 'P'
        chi2_base = results[base]
        best_name = min(results, key=results.get)
        margin = (chi2_base - results[best_name]) / abs(chi2_base) if chi2_base else 0.0
        if margin < (min_chi2_improvement if min_chi2_improvement is not None else 0.05):
            best_name = base
    else:
        best_name = min(results, key=results.get)
    return candidates[best_name], best_name, candidates, results


def reject_outliers(t, flux, ferr, period_hr, n_harm=4, sigma=5.0, max_iter=3):
    """Flag points that don't fit the periodic model at the (already known)
    rotation period -- bad PSF centroiding, blended background stars, partial
    frames -- almost always land BELOW the true brightness at their phase,
    not scattered symmetrically around a local time-domain trend. A
    time-domain smoothness check can't tell that apart from genuine
    rotational amplitude, so reject against a Fourier fit of the phase-folded
    curve instead: fit, flag >sigma*MAD residuals, refit without them, repeat.

    Returns a boolean keep-mask (True = kept).
    """
    keep = np.ones(len(t), dtype=bool)
    for _ in range(max_iter):
        idx = np.where(keep)[0]
        phase = (t[idx] * 24.0 / period_hr) % 1.0
        A = _fourier_design(phase, n_harm)
        w = 1.0 / ferr[idx]
        coeffs, *_ = np.linalg.lstsq(A * w[:, None], flux[idx] * w, rcond=None)
        resid = flux[idx] - A @ coeffs
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad if mad > 0 else np.std(resid)
        bad = np.abs(resid) > sigma * scale
        if not bad.any():
            break
        keep[idx[bad]] = False
    return keep


def load_target(cut_path, designation):
    eph_files = glob.glob(f'{cut_path}/*_Asteroids.parquet')
    psf_files = glob.glob(f'{cut_path}/*_AsteroidPSFPhotometry.parquet')
    eph = pd.read_parquet(eph_files[0])
    psf = pd.read_parquet(psf_files[0])
    eph = eph[eph['designation'] == designation].sort_values('mjd')
    psf = psf[psf['designation'] == designation].sort_values('mjd')
    merged = pd.merge_asof(psf, eph[['mjd', 'ra', 'dec', 'phase_angle_deg', 'r_helio_au', 'delta_au', 'mag_expected']],
                            on='mjd', direction='nearest', tolerance=0.01)
    return merged.dropna(subset=['phase_angle_deg'])


def load_target_multi(cut_paths, designation):
    """Load and concatenate a target's data across multiple cuts (e.g. the
    same asteroid crossing several spatial cuts of a sector at different
    times), deduplicating on mjd in case cuts overlap spatially.
    """
    frames = [load_target(cp, designation) for cp in cut_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset='mjd').sort_values('mjd').reset_index(drop=True)
    return combined


def analyse(df, designation, sector, min_period_hr=1.0, max_period_hr=24.0):
    clean = df[~df.get('near_bright_star', False).astype(bool)] if 'near_bright_star' in df else df
    if 'flux_detrended' in clean.columns and clean['flux_detrended'].notna().any():
        flux_raw, flux_col = clean['flux_detrended'].values, 'flux_detrended'
    else:
        flux_raw, flux_col = clean['flux'].values, 'flux'
    t = clean['mjd'].values
    ferr_raw = clean['e_flux'].values if 'e_flux' in clean else np.full_like(flux_raw, np.nanstd(flux_raw) * 0.1)

    # mag_expected already bakes in the 1/(r_helio^2 * delta^2) distance dilution and
    # the two-term HG phase function -- as r_helio/delta/phase_angle drift over the
    # sector, that secular brightness trend can swamp or alias with a short rotational
    # period in the periodogram. Divide it out so only the rotational modulation remains.
    mag_ref = clean['mag_expected'].median()
    geom_corr = 10 ** (0.4 * (clean['mag_expected'].values - mag_ref))
    flux = flux_raw * geom_corr
    ferr = ferr_raw * geom_corr

    freq_min = 1.0 / (max_period_hr / 24.0)
    freq_max = 1.0 / (min_period_hr / 24.0)
    ls = LombScargle(t, flux, ferr)
    freq, power = ls.autopower(minimum_frequency=freq_min, maximum_frequency=freq_max,
                                samples_per_peak=10)
    best_freq = freq[np.argmax(power)]
    ls_period_hr = (1.0 / best_freq) * 24.0
    fap = ls.false_alarm_probability(power.max())

    best_period_hr, best_name, candidates, chi2_by_name = disambiguate_period(t, flux, ferr, ls_period_hr)

    # Now that the period is known, reject points that don't fit the
    # phase-folded model at that period (bad frames), then refit the
    # disambiguation chi2s on the cleaned data for the reported/plotted result.
    keep = reject_outliers(t, flux, ferr, best_period_hr)
    n_rejected = int((~keep).sum())
    t_rej, flux_rej, ferr_raw_rej = t[~keep], flux_raw[~keep], ferr_raw[~keep]
    clean = clean.iloc[keep].reset_index(drop=True)
    t, flux_raw, ferr_raw = t[keep], flux_raw[keep], ferr_raw[keep]
    flux, ferr = flux[keep], ferr[keep]
    chi2_by_name = {name: _fourier_reduced_chi2(t, flux, ferr, p, 4) for name, p in candidates.items()}

    phase = ((t * 24.0 / best_period_hr) % 1.0)
    phase_angle_range = clean['phase_angle_deg'].max() - clean['phase_angle_deg'].min()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ax = axes[0, 0]
    ax.errorbar(t - t.min(), flux_raw, yerr=ferr_raw, fmt='o', ms=3, color='0.6', ecolor='0.8',
                alpha=0.5, label='raw')
    ax.errorbar(t - t.min(), flux, yerr=ferr, fmt='o', ms=3, color='C0', ecolor='0.7',
                alpha=0.7, label='geometry-corrected')
    if n_rejected:
        ax.scatter(t_rej - t.min(), flux_rej, marker='x', s=25, color='C3', linewidths=1.2,
                   label=f'rejected ({n_rejected})', zorder=5)
    ax.set_xlabel('Time (days from start)')
    ax.set_ylabel(f'Flux ({flux_col})')
    ax.set_title(f'{designation} -- Sector {sector} lightcurve')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(24.0 / freq, power, color='C1')
    ax.axvline(ls_period_hr, color='0.4', ls=':', lw=1, label=f'LS raw: {ls_period_hr:.3f} hr')
    ax.axvline(best_period_hr, color='C3', ls='--', lw=1.5, label=f'adopted ({best_name}): {best_period_hr:.3f} hr')
    ax.set_xlabel('Period (hr)')
    ax.set_ylabel('Lomb-Scargle power')
    ax.set_title(f'Periodogram (FAP={fap:.2e})')
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    names = list(candidates.keys())
    chi2s = [chi2_by_name[n] for n in names]
    colors = ['C3' if n == best_name else 'C0' for n in names]
    ax.bar(names, chi2s, color=colors)
    ax.set_ylabel('Reduced chi-square (Fourier fit)')
    ax.set_title('Period-harmonic disambiguation')

    ax = axes[1, 0]
    ax.errorbar(phase, flux, yerr=ferr, fmt='o', ms=3, color='C0', ecolor='0.7', alpha=0.7)
    ax.errorbar(phase + 1, flux, yerr=ferr, fmt='o', ms=3, color='C0', ecolor='0.7', alpha=0.4)
    phase_grid = np.linspace(0, 2, 400)
    A_grid = _fourier_design(phase_grid % 1.0, 4)
    A_fit = _fourier_design(phase, 4)
    w = 1.0 / ferr
    coeffs, *_ = np.linalg.lstsq(A_fit * w[:, None], flux * w, rcond=None)
    ax.plot(phase_grid, A_grid @ coeffs, color='C3', lw=1.5)
    ax.set_xlabel('Rotation phase')
    ax.set_ylabel(f'Flux ({flux_col})')
    ax.set_title(f'Phase-folded at adopted period ({best_period_hr:.3f} hr)')

    ax = axes[1, 1]
    sc = ax.scatter(t - t.min(), clean['phase_angle_deg'], c=clean['delta_au'], cmap='cividis')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Geocentric distance (au)')
    ax.set_xlabel('Time (days from start)')
    ax.set_ylabel('Solar phase angle (deg)')
    ax.set_title(f'Phase-angle coverage (range={phase_angle_range:.2f} deg)')

    axes[1, 2].axis('off')

    fig.tight_layout()
    out_path = f'{OUT_DIR}/{designation.replace(" ", "_")}_S{sector}_prototype.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f'Target: {designation} (Sector {sector})')
    print(f'  N frames used: {len(t)} ({n_rejected} outliers rejected)')
    print(f'  Lomb-Scargle raw period: {ls_period_hr:.4f} hr (FAP={fap:.3e})')
    print(f'  Harmonic disambiguation (reduced chi2): ' +
          ', '.join(f'{n}={chi2_by_name[n]:.3f}' for n in candidates))
    print(f'  Adopted period: {best_period_hr:.4f} hr ({best_name} of the LS peak)')
    print(f'  Phase-angle range: {phase_angle_range:.2f} deg')
    print(f'  Verdict: {"adequate single-apparition coverage" if phase_angle_range > 15 else "narrow phase-angle coverage -- shape inversion will need archival/multi-sector data"}')
    print(f'  Plot: {out_path}')
    return out_path


if __name__ == '__main__':
    # Usage: lightcurve_prototype.py <designation> <sector> <cut_path> [<cut_path> ...]
    designation, sector = sys.argv[1], sys.argv[2]
    cut_paths = sys.argv[3:]
    df = load_target_multi(cut_paths, designation) if len(cut_paths) > 1 else load_target(cut_paths[0], designation)
    analyse(df, designation, sector)
