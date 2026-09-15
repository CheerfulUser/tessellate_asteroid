"""All-sector asteroid lightcurve report: stitches each asteroid's forced-PSF
photometry across every cut it was predicted to cross IN ANY OF SECTORS,
searches for a rotation period via Lomb-Scargle where there's enough data,
and saves a two-panel (lightcurve + periodogram) figure per object. Also
produces an all-sector spin-rate vs. diameter ("spin barrier") diagram, and
writes a per-object clean-lightcurve CSV (mjd, flux, ra, dec, delta_au,
phase_angle_deg) for every object with enough points + phase-angle spread to
attempt convex shape inversion downstream (see shape_model_pipeline.py).

Generalized from the single-sector prototype (asteroid_test/asteroid_report_
pipeline.py, hardcoded to Sector 32) to pool every completed sector -- the
per-designation grouping, stacking, and periodogram logic is unchanged; what
changes is that photometry/ephemeris paths are globbed across all SECTORS,
and each visit id is tagged with its sector so cuts from different sectors
never collide under the same visit key.

Run on ozstar (data-heavy: scans every cut's AsteroidPSFPhotometry table
across every sector).
"""
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

from tessellate.asteroid_photometry import _stack_track, _required_n, STACK_SIG_TARGET

# Robust period-determination pipeline (geometric-trend correction, harmonic
# disambiguation, phase-residual outlier rejection), validated against real
# TESS data (Polyxena, May) in shape_proto/lightcurve_prototype.py -- reused
# here rather than re-implemented so both scripts share one source of truth.
sys.path.insert(0, "/fred/oz335/rridden/code/runs/shape_proto")
from lightcurve_prototype import _fourier_design, disambiguate_period, reject_outliers  # noqa: E402

import os as _os
_sec_lo = int(_os.environ.get("ASTEROID_SEC_LO", 27))
_sec_hi = int(_os.environ.get("ASTEROID_SEC_HI", 55))
SECTORS = list(range(_sec_lo, _sec_hi + 1))
DATA_ROOT = "/fred/oz335/TESSdata"
_outdir_suffix = _os.environ.get("ASTEROID_OUTDIR_SUFFIX", "")
OUTDIR = "/fred/oz335/rridden/asteroids" + _outdir_suffix
LC_OUTDIR = f"{OUTDIR}/lightcurves"
CLEAN_LC_DIR = f"{OUTDIR}/clean_lightcurves"
SUMMARY_PATH = f"{OUTDIR}/all_sector_asteroid_report.csv"
SPIN_SIZE_PATH = f"{OUTDIR}/all_sector_spin_size.png"
SBDB_PATH = "/fred/oz335/rridden/asteroids/sbdb_physical_properties.json"

# Per-object PNG generation (lightcurve+periodogram+phase-fold, 3 subplots each) is
# expensive at full 29-sector scale (tens of thousands of objects) and produces far more
# figures than anyone will look at one-by-one. Default OFF for full-scale runs; set
# ASTEROID_SAVE_PLOTS=1 to re-enable for a small/test run. Every other output (summary row,
# clean-lc CSV, spin-size plot) is unaffected -- only the per-object PNG is skipped.
SAVE_PLOTS = _os.environ.get("ASTEROID_SAVE_PLOTS", "0") == "1"

# every point used anywhere in this report (lightcurve, periodogram, phase fold) must clear
# this significance -- raw single-frame TESS asteroid photometry is often noise-dominated
# (flux_detrended can go negative), which both misrepresents the lightcurve and injects
# spurious periodogram power. Re-uses the package's own shift-and-stack target so a "5 sigma
# point" means the same thing here as it does in forced photometry.
MIN_SNR = STACK_SIG_TARGET

MIN_POINTS_FOR_PLOT = 10
MIN_POINTS_FOR_PERIODOGRAM = 30
MIN_PERIOD_DAYS = 0.04       # ~1 hr
MAX_PERIOD_DAYS = 10.0       # 240 hr -- widened from 48hr (see below). Draft "Asteroids in
# TESSELLATE" paper searches 20 min-200 hr for comparison.
# NOTE: the 48hr cap was a holdover from single-sector (~27-day baseline) studies, where a
# period anywhere near the baseline can't be meaningfully constrained anyway. This run spans
# 29 sectors with per-object visit spans often in the hundreds of hours (e.g. one (5118)
# Elnapoul visit alone ran ~600hr), so a single-sector-sized cap silently discards real
# longer-period detections the multi-sector baseline can actually constrain. Confirmed on 12
# real flagged-wrong-period objects: 8/12 had their true period sitting entirely outside the
# old 48hr search band -- (5118) Elnapoul's real period (73.2hr, LS power=0.786, a factor of
# ~20 stronger than anything found inside the old band) was never even a candidate under the
# old cap; the search found a weak in-band alias (43.2hr, power=0.038) and the harmonic
# disambiguation step doubled THAT to 86.3hr, landing on a distinct wrong answer, not a chi2
# failure. n_cycles_observed>=2 (RELIABLE_MIN_CYCLES) already guards the opposite failure
# mode (fitting a period close to/exceeding the actual baseline), so widening this doesn't
# reopen that hole. Remaining 4/12 (Munroe, Anshan, Mitamasahiro, VM34) stayed genuinely
# ambiguous even with the wider band (no single dominant periodogram peak) -- consistent
# with non-principal-axis/tumbling rotators, not a search-window artifact; not resolved by
# this change and still need a dedicated discriminator before is_reliable can trust them.
SAMPLES_PER_PEAK = 10

# standard absolute-magnitude-to-diameter proxy (e.g. used by MPC/JPL when no direct diameter
# measurement exists): D [km] = 1329 / sqrt(pV) * 10^(-H/5). No real albedo is known per-object
# here, so a single canonical assumed geometric albedo is used for every object and clearly
# labelled as an assumption on the resulting plot -- NOT a measured diameter.
ASSUMED_ALBEDO = 0.15

# ---- Single canonical "trustworthy measurement" definition -------------------------------
# A period is only as trustworthy as (a) the false-alarm probability of its periodogram peak,
# (b) having enough clean (post-outlier-rejection) points for that FAP to mean anything -- a
# low FAP computed from a handful of points is not actually confident, just underdetermined --
# and (c) the observation baseline actually covering at least one full rotation: a "period"
# longer than the span of data it was fit to is not a measured periodicity, it is an
# unconstrained long-period/trend fit that a Lomb-Scargle periodogram can still report a
# low-FAP peak for (a slow drift across a short baseline looks like part of a long sinusoid).
# RELIABLE_FAP_MAX=0.01 was already the threshold independently chosen (and validated by hand
# on real objects) for both the spin-size plot and the shape-modeling clean-lc export before
# this consolidation -- unifying them here means there is exactly ONE place this number is
# set, rather than two constants that happened to agree by coincidence and could silently
# drift apart. is_reliable is computed once per object in plot_asteroid() and written directly
# into the summary CSV (along with the baseline_hr and n_cycles_observed it was judged
# against), so any downstream consumer (comparison scripts, paper tables, the spin-size plot)
# can filter on df.is_reliable instead of re-deriving this logic each time.
#
# Two more criteria added after manual QC review of a 500-object random sample found real
# failure modes that FAP alone missed, both confirmed against real objects (see
# periodogram_with_harmonic_check's docstring for the exact power_ratio methodology and cases):
#   (d) power_ratio: the adopted period's periodogram power (nterms=2, on outlier-cleaned
#       data) must not have collapsed relative to the naive (pre-harmonic-disambiguation) LS
#       peak, evaluated with the SAME model on the SAME data -- catches disambiguate_period
#       locking onto a long chi2-favoured harmonic (2P/3P) that "connects the dots" of sparse,
#       widely-gapped visits but carries no real periodogram power. 9 confirmed-bad objects all
#       had power_ratio in [0.003, 0.055]; RELIABLE_MIN_POWER_RATIO is stable across the whole
#       0.1-0.5 range against those plus 6 confirmed-good controls (3 with an originally-strong
#       peak, 2 with sparse/gapped sampling similar to the bad cases) with zero false positives
#       after fixing the comparison to be apples-to-apples -- an earlier nterms=1-vs-nterms=2
#       version of this ratio produced false positives on good objects whose period was never
#       even overridden, since the two models' power isn't on a directly comparable scale.
#   (e) max_phase_gap: the adopted period's phase-folded coverage must not leave a large
#       fraction of the rotation cycle completely unsampled -- catches objects whose points
#       happen to satisfy every other check by count/significance alone while being clustered
#       in one arc of the cycle (confirmed on a real object with ~52% of the phase cycle never
#       observed at all, (890029) 2013 EF87). RELIABLE_MAX_PHASE_GAP is set above the confirmed-
#       good control examples used to calibrate it (0.15-0.27) and below the confirmed-bad one.
RELIABLE_FAP_MAX = 0.01
RELIABLE_MIN_POINTS = MIN_POINTS_FOR_PERIODOGRAM
RELIABLE_MIN_CYCLES = 2.0  # observation baseline must span at least this many adopted periods
RELIABLE_MIN_POWER_RATIO = 0.3  # adopted period must retain >=30% of the naive peak's own power
RELIABLE_MAX_PHASE_GAP = 0.3  # no more than 30% of the rotation phase left completely unsampled

# ---- Learned reliability score (replaces the hand-tuned multi-branch threshold rule) --------
# The branch rule was calibrated against a 500-object hand-labelled sample that has since gone
# 42.8% stale (periods changed underneath the labels), and its fixed floors cannot express a
# criterion whose correct DIRECTION depends on the period regime -- e.g. more observed cycles
# is strongly better above 8hr (agreement 9%->28% beyond 30hr) but WORSE below 3hr (58%->43%),
# because a huge cycle count at a short period just means the claimed period is tiny relative
# to the baseline, which is the alias/noise regime. A single global floor cannot encode that;
# a tree ensemble does so automatically, which is why n_cycles_observed is simultaneously the
# top permutation-importance feature and only 0.580 as a standalone (monotonic) AUC.
#
# The same structural weakness produced three separate production false positives, each
# entering through a DIFFERENT branch that happened to lack the relevant check ((4942) Munroe
# via A_old, (37373) 2001 VM34 via B, (91592) 1999 TU3 via D). Six branches x six criteria is
# too many combinations to keep mutually consistent by hand.
#
# The model is trained on LCDB agreement -- objective, ~6x larger than the hand sample, and it
# cannot go stale because it is recomputed against whatever period the pipeline produces.
# Validation (train on LCDB U=2, test on the disjoint U=3 "secure" set): ROC AUC 0.845, and at
# the purity the old rule achieves (95%) it recovers 91.1% of trustworthy lightcurves versus
# the old rule's 78.1%. Brightness extrapolation was checked explicitly (train bright, test
# faint: AUC 0.890-0.928, no degradation), since LCDB only covers the bright ~2%.
#
# NOTE the score is a RANKING, not a calibrated probability: an isotonic calibration collapsed
# when applied to the full catalogue because the base rate differs from the training
# population. Read purity off the U=3 validation, never off the score value.
RELIABILITY_MODEL_PATH = _os.environ.get(
    "ASTEROID_RELIABILITY_MODEL", f"{OUTDIR}/reliability_classifier.joblib")
RELIABILITY_SCORE_MIN = 0.564  # v4 model. Chosen on a 600-object dev set and measured ONCE on
# a held-out 200 the model and threshold never saw: precision 90.9%, recall 92.6%, AUC 0.9803,
# against the previous model's 66.7% / 85.2% on the same objects. The earlier 0.45 came from a
# model trained on LCDB U=3 agreement, which the vetting showed was the wrong target -- 42% of
# its training negatives were LCDB errors, not ours.
#
# HARD FLOORS. The model extrapolates confidently outside its training range: applied to the
# catalogue it scored objects with ~30 points and aov_F<1 above 0.9. Hand vetting of 800 objects
# found ZERO believable periods below either floor, so they cost no recall and remove the
# failure mode outright. They are applied in the reliability decision, not post hoc.
RELIABILITY_MIN_POINTS_FLOOR = 100
RELIABILITY_MIN_AOVF_FLOOR = 1.5
# n_phase_coverings is still COMPUTED and reported (it is a useful diagnostic and the
# corrected version of a statistic that was structurally broken below ~8hr), but it is NOT a
# model feature: it can only be derived from a saved lightcurve, and only 6,092 of the 8,964
# LCDB-matched objects have one. Including it would have restricted training to that subset,
# which is 73.1% correct against 52.6% in the full overlap -- i.e. a pre-filtered, easier
# population, since clean_lightcurves/ was only ever written for objects passing an earlier
# reliability cut. Training on the full overlap is worth far more than the +0.008 AUC the
# feature contributed (validated: 0.9069 on the full overlap vs 0.8445 on the subset).
CLASSIFIER_FEATURES = ['aov_F', 'amp_snr', 'max_phase_gap', 'n_cycles_observed', 'n_points',
                        'loo_worst_chi2', 'power_ratio', 'fap', 'ls_power', 'naive_power',
                        'n_visits', 'n_sectors', 'baseline_hr', 'n_outliers_rejected',
                        'frac_points_stacked', 'phase_angle_range_deg', 'median_n_stack',
                        'pk_over_med', 'pk_prom', 'pk_snr']

_reliability_bundle = None
_reliability_tried = False


def get_reliability_model():
    """Lazily load the trained classifier once per worker process. Returns None if it is not
    available, in which case is_reliable falls back to the legacy branch rule rather than
    failing the run."""
    global _reliability_bundle, _reliability_tried
    if not _reliability_tried:
        _reliability_tried = True
        try:
            import joblib
            _reliability_bundle = joblib.load(RELIABILITY_MODEL_PATH)
        except Exception as e:
            print(f"  reliability model unavailable ({e}) -- falling back to the legacy "
                  f"branch rule", flush=True)
            _reliability_bundle = None
    return _reliability_bundle


def reliability_score(feats):
    """Score one object from its summary statistics. NaN if no model is loaded."""
    b = get_reliability_model()
    if b is None:
        return np.nan
    x = np.array([[float(feats.get(f, np.nan)) for f in b['features']]], dtype=float)
    x[~np.isfinite(x)] = np.nan
    try:
        return float(b['model'].predict_proba(x)[0, 1])
    except Exception:
        return np.nan

# confidence threshold for including an object in the spin-size population plot
SPIN_SIZE_FAP_MAX = RELIABLE_FAP_MAX

# shape-modeling candidacy: an object needs a confident period AND enough phase-angle spread
# across its (possibly multi-sector) observations for convex inversion to be meaningfully
# constrained -- single-sector, single-apparition objects (~1-2 deg spread, see real_shape_may.py's
# caveat) are still written out, just flagged low-confidence; this threshold only gates whether
# a clean-lc CSV gets written at all (need enough points to bother).
SHAPE_MODEL_MIN_POINTS = 20
SHAPE_MODEL_FAP_MAX = RELIABLE_FAP_MAX


def n_available_workers():
    """os.cpu_count() reports the PHYSICAL node's total core count, not the SLURM/cgroup
    allocation this job is actually restricted to -- confirmed the hard way: a job requesting
    --cpus-per-task=8 spawned 36 ProcessPoolExecutor workers (the node's real core count) and
    OOM'd, because 36-way concurrent per-batch memory use is ~4.5x what the resource request
    was sized for. os.sched_getaffinity(0) reports the actual cgroup-restricted CPU set
    (standard fix for this on SLURM/cgroup-based clusters); fall back to os.cpu_count() only
    where affinity isn't available (e.g. non-Linux)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 4)


def safe_name(designation):
    return re.sub(r"[^A-Za-z0-9]+", "_", designation).strip("_")


def find_photometry_cuts():
    paths = []
    for sector in SECTORS:
        paths.extend(glob.glob(f"{DATA_ROOT}/Sector{sector}/Cam*/Ccd*/Cut*of*/asteroids/*_AsteroidPSFPhotometry.parquet"))
    return sorted(paths)


def find_ephemeris_cuts():
    paths = []
    for sector in SECTORS:
        paths.extend(glob.glob(f"{DATA_ROOT}/Sector{sector}/Cam*/Ccd*/Cut*of*/asteroids/*_Asteroids.parquet"))
    return sorted(paths)


def load_all_photometry():
    """Load every cut's PSF photometry across every sector, tag with a unique visit id
    (sector/cam/ccd/cut) so each visit's flux can be independently median-normalised before
    stitching -- robust to any residual per-cut zeropoint/aperture differences, since what
    matters for a rotation lightcurve is the fractional variation, not the absolute flux
    scale.

    Also merges each visit's ephemeris mag_expected/ra/dec/delta_au/phase_angle_deg (nearest
    mjd, per designation) and divides mag_expected out of flux_detrended/e_flux before
    returning -- mag_expected bakes in the 1/(r_helio^2*delta^2) distance dilution and the
    phase-function brightness trend, which drifts within a single visit and, left uncorrected,
    can swamp or alias with a short rotation period in the periodogram (found and fixed on
    real (348) May/(595) Polyxena TESS data; see lightcurve_prototype.py). Corrected per visit,
    matching the per-visit normalisation already done downstream in normalise(). ra/dec/
    delta_au/phase_angle_deg are carried through unchanged (geometry, not flux) for the
    downstream shape-modeling stage, which needs them to build convexinv's Sun/observer
    vectors."""
    paths = find_photometry_cuts()
    print(f"{len(paths)} cuts with PSF photometry across sectors {SECTORS[0]}-{SECTORS[-1]}", flush=True)

    frames = []
    n_workers = n_available_workers()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_one_cut, path): path for path in paths}
        for i, fut in enumerate(as_completed(futures)):
            try:
                df = fut.result()
            except Exception as e:
                print(f"  skip {futures[fut]}: {e}", flush=True)
                continue
            if df is not None and len(df):
                frames.append(df)
            if (i + 1) % 200 == 0:
                print(f"  loaded {i+1}/{len(paths)} cuts", flush=True)

    if not frames:
        return pd.DataFrame(columns=["designation", "mjd", "flux_detrended", "e_flux", "visit",
                                      "ra", "dec", "delta_au", "phase_angle_deg", "sector"])
    return pd.concat(frames, ignore_index=True)


def _load_one_cut(path):
    """One cut's photometry+ephemeris load/merge/geometric-correction, isolated as a
    top-level function so it can run in a ProcessPoolExecutor worker -- this loop is
    embarrassingly parallel across the (now ~29x more) independent cut files and was
    previously the single-threaded bottleneck dominating the whole job's wall time."""
    m = re.search(r"Sector(\d+)/Cam(\d+)/Ccd(\d+)/Cut(\d+)of\d+", path)
    if not m:
        return None
    sector, cam, ccd, cut = m.groups()
    df = pd.read_parquet(path, columns=["designation", "mjd", "flux_detrended", "e_flux", "near_bright_star"])
    if len(df) == 0:
        return None
    df = df[~df["near_bright_star"]].copy()
    if len(df) == 0:
        return None

    eph_path = path.replace("_AsteroidPSFPhotometry.parquet", "_Asteroids.parquet")
    try:
        eph = pd.read_parquet(eph_path, columns=["designation", "mjd", "mag_expected", "ra", "dec",
                                                   "delta_au", "phase_angle_deg"])
    except Exception:
        eph = None
    if eph is not None and len(eph):
        df = df.sort_values("mjd")
        eph = eph.sort_values("mjd")
        df = pd.merge_asof(df, eph, on="mjd", by="designation", direction="nearest", tolerance=0.01)
        has_mag = df["mag_expected"].notna()
        mag_ref = df.groupby("designation")["mag_expected"].transform("median")
        geom_corr = 10 ** (0.4 * (df["mag_expected"] - mag_ref))
        df.loc[has_mag, "flux_detrended"] = df.loc[has_mag, "flux_detrended"] * geom_corr[has_mag]
        df.loc[has_mag, "e_flux"] = df.loc[has_mag, "e_flux"] * geom_corr[has_mag]
        df = df.drop(columns=["mag_expected"])
    else:
        for col in ("ra", "dec", "delta_au", "phase_angle_deg"):
            df[col] = np.nan

    df["visit"] = f"S{sector}_C{cam}_{ccd}_{cut}"
    df["sector"] = int(sector)
    return df


def load_h_magnitudes():
    """designation -> magnitude_H, from the (much lighter) ephemeris tables -- H is constant
    per object across every cut/sector it appears in, so first-seen is fine."""
    paths = find_ephemeris_cuts()
    h_by_designation = {}
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=["designation", "magnitude_H"])
        except Exception:
            continue
        for designation, h in df.drop_duplicates("designation").itertuples(index=False):
            if designation not in h_by_designation and np.isfinite(h):
                h_by_designation[designation] = h
    return h_by_designation


def diameter_km(H, albedo=ASSUMED_ALBEDO):
    return 1329.0 / np.sqrt(albedo) * 10 ** (-H / 5.0)


def _norm_designation(text):
    """Normalise a provisional designation for matching -- collapse whitespace, uppercase."""
    return re.sub(r"\s+", " ", text.strip().upper())


def _parse_sbdb_full_name(full_name):
    """SBDB's full_name is one of:
      '     1 Ceres (A801 AA)'      -- numbered, named, (packed old-style or provisional)
      '248401 (2005 SP75)'          -- numbered, unnamed, (provisional designation)
      'C/1995 O1 (Hale-Bopp)'       -- comet, not a minor planet -- excluded
    Returns (number:int|None, provisional_designation:str|None)."""
    full_name = full_name.strip()
    if re.match(r"^[CPD]/", full_name):
        return None, None  # comet designation prefix, not an asteroid
    m = re.match(r"^(\d+)\s+(?:.+?\s+)?\(([^)]+)\)$", full_name)
    if m:
        return int(m.group(1)), m.group(2)
    m = re.match(r"^(\d+)\s+(.+)$", full_name)
    if m:
        return int(m.group(1)), None
    return None, full_name


def _parse_my_designation(designation):
    """My own designation format (from MPCORB via skyfield): '(885112) 2018 HU2' (numbered)
    or '2013 BR80' (unnumbered). Returns (number:int|None, provisional_designation:str)."""
    m = re.match(r"^\((\d+)\)\s*(.*)$", designation.strip())
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, designation.strip()


def load_sbdb_physical_properties():
    """Real per-object diameter/albedo/taxonomic type/published rotation period from JPL's
    Small-Body DataBase (physical-properties fields, objects with a measured diameter only --
    ~140k of the >1.3M minor planets in SBDB). Matched by MPC number where numbered (most
    reliable), else by normalised provisional designation text."""
    if not os.path.exists(SBDB_PATH):
        print(f"no SBDB catalog found at {SBDB_PATH} -- spin-size plot will fall back to the "
              f"assumed-albedo proxy for every object", flush=True)
        return {}, {}

    with open(SBDB_PATH) as f:
        data = json.load(f)
    fields = data["fields"]
    idx = {name: i for i, name in enumerate(fields)}

    by_number, by_designation = {}, {}
    for row in data["data"]:
        number, prov = _parse_sbdb_full_name(row[idx["full_name"]])
        if number is None and prov is None:
            continue  # comet
        entry = dict(
            diameter_km=float(row[idx["diameter"]]) if row[idx["diameter"]] is not None else np.nan,
            albedo=float(row[idx["albedo"]]) if row[idx["albedo"]] is not None else np.nan,
            rot_per_hr=float(row[idx["rot_per"]]) if row[idx["rot_per"]] is not None else np.nan,
            spec_type=row[idx["spec_T"]] or row[idx["spec_B"]],
        )
        if number is not None:
            by_number[number] = entry
        elif prov is not None:
            by_designation[_norm_designation(prov)] = entry

    print(f"SBDB catalog: {len(by_number)} numbered + {len(by_designation)} unnumbered "
          f"objects with physical properties", flush=True)
    return by_number, by_designation


def match_sbdb(designation, by_number, by_designation):
    number, prov = _parse_my_designation(designation)
    if number is not None and number in by_number:
        return by_number[number]
    if prov:
        return by_designation.get(_norm_designation(prov))
    return None


def _stack_one_designation(designation, g, min_snr):
    """One designation's worth of (visit) groups, stacked -- isolated as a top-level function
    so ProcessPoolExecutor can dispatch across designations. Dispatching per-designation
    rather than per-(designation,visit) keeps task count (and pickling overhead) down to the
    number of unique objects rather than the much larger number of visits."""
    rows = []
    for visit, gv in g.groupby("visit"):
        gv = gv.sort_values("mjd").reset_index(drop=True)
        raw_sig = (gv["flux_detrended"] / gv["e_flux"]).replace([np.inf, -np.inf], np.nan)
        avg_sig = raw_sig.median()
        n_stack = _required_n(avg_sig, min_snr)
        n_stack = 1 if n_stack is None else min(n_stack, len(gv))
        stacked = _stack_track(gv, n_stack)
        stacked = stacked[stacked["sig"] >= min_snr].copy()
        if len(stacked) == 0:
            continue
        stacked["designation"] = designation
        stacked["visit"] = visit
        stacked["sector"] = gv["sector"].iloc[0]
        # bin-average geometry columns onto the same n_stack grouping the flux stacking used
        for col in ("ra", "dec", "delta_au", "phase_angle_deg"):
            vals = gv[col].values
            n = len(stacked)
            chunks = np.array_split(vals, n) if n else []
            stacked[col] = [np.nanmean(c) if len(c) else np.nan for c in chunks]
        rows.append(stacked)
    return pd.concat(rows, ignore_index=True) if rows else None


def stack_to_min_snr(all_phot, min_snr=MIN_SNR):
    """Per (designation, visit), combine consecutive frames with the package's own
    shift-and-stack (weighted mean flux, sqrt(n) SNR scaling; see _stack_track/_required_n
    in asteroid_photometry.py) until each output point clears min_snr. Frames that can never
    reach it even combined into one single bin (n_stack would exceed what's available) are
    stacked as far as possible and then dropped by the sig>=min_snr filter below, rather than
    kept as a sub-threshold point. n_frames on the output records exactly how many raw frames
    went into each point, so any stacking is auditable, not silent. ra/dec/delta_au/
    phase_angle_deg are carried through as the (unweighted) mean over each stacked bin -- the
    geometry barely changes frame-to-frame within one stack, so this is an adequate summary
    for the shape-modeling geometry, not a precision requirement.

    Dispatched across a ProcessPoolExecutor per designation (thousands of independent objects
    once pooling all sectors, was a single-threaded Python loop over every (designation,visit)
    pair before -- the same bottleneck load_all_photometry's per-cut loop had, fixed the same
    way).

    Submitted in bounded batches rather than all designations' futures at once: a dict
    comprehension over the full groupby (`{pool.submit(...): d for d, g in all_phot.groupby(...)}`)
    eagerly slices (copies) EVERY group out of all_phot and queues every one of them for
    inter-process pickling before a single worker has consumed one -- fine at Y3-only scale
    (~86k designations) but at full 29-sector scale (~492k designations, ~1.5B rows) this
    transiently held roughly 2-3x the raw dataset in memory on top of all_phot itself already
    being resident, and OOM-killed the job outright (confirmed: peak RAM jumped from ~221GB,
    where load_all_photometry left it, to >300GB within seconds of this function starting).
    Bounding how many groups are in flight at once keeps the extra peak close to
    (BATCH_SIZE x average group size) instead of (ALL designations x average group size)."""
    n_workers = n_available_workers()
    BATCH_SIZE = max(2000, n_workers * 50)
    out = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        batch = []
        for d, g in all_phot.groupby("designation"):
            batch.append((d, g))
            if len(batch) >= BATCH_SIZE:
                futures = {pool.submit(_stack_one_designation, d, g, min_snr): d for d, g in batch}
                batch = []
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None:
                        out.append(res)
                del futures
        if batch:
            futures = {pool.submit(_stack_one_designation, d, g, min_snr): d for d, g in batch}
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    out.append(res)
    if not out:
        return pd.DataFrame(columns=["designation", "visit", "mjd", "mjd_err", "flux", "e_flux", "sig",
                                      "n_frames", "sector", "ra", "dec", "delta_au", "phase_angle_deg"])
    return pd.concat(out, ignore_index=True)


def normalise(stacked_phot):
    """Per (designation, visit), normalise flux to that visit's own median of its (already
    >=MIN_SNR) stacked points -- puts every visit's segment on the same relative scale
    (median ~1) regardless of cut-to-cut/sector-to-sector zeropoint/calibration differences,
    so segments from different cuts (now potentially different sectors) stitch into one
    consistent relative-flux lightcurve. Every input point has flux >= MIN_SNR * e_flux > 0,
    so rel_flux can never go negative here.

    Fully vectorised via groupby.transform -- no per-group Python loop needed (unlike
    stack_to_min_snr, this doesn't call any external per-row function, so there's nothing a
    loop buys over a single vectorised pass)."""
    if len(stacked_phot) == 0:
        return pd.DataFrame(columns=["designation", "mjd", "rel_flux", "rel_flux_err", "visit", "n_frames",
                                      "sector", "ra", "dec", "delta_au", "phase_angle_deg", "flux", "e_flux"])
    med = stacked_phot.groupby(["designation", "visit"])["flux"].transform("median")
    keep = med.notna() & (med > 0)
    out = stacked_phot.loc[keep].copy()
    med = med.loc[keep]
    out["rel_flux"] = out["flux"] / med
    out["rel_flux_err"] = out["e_flux"] / med
    return out.reset_index(drop=True)


DISAMBIGUATION_N_HARM = 4  # see lightcurve_prototype.py: n_harm=2 is a pure double-sinusoid
# and isn't flexible enough to represent a real asteroid's folded shape even at the TRUE
# period -- verified wrong (favoured the half-period alias) on real (348) May TESS
# photometry at n_harm=2, corrected at n_harm=4+. Applies here for the same reason.


def periodogram_with_harmonic_check(mjd, rel_flux, rel_flux_err):
    """Standard single-sinusoid (nterms=1) Lomb-Scargle aliases to HALF the true rotation
    period for a double-peaked lightcurve (two similar-brightness maxima per rotation, from
    presenting two elongated-body faces per revolution) -- its periodogram peak lands at 2x
    the true rotation frequency, since that's the dominant Fourier component of a symmetric
    double-humped shape. Used here only to seed a starting-period guess; the actual P vs 2P
    (and P/2, 3P, P/3) decision is made by disambiguate_period's reduced-chi2 comparison at
    DISAMBIGUATION_N_HARM harmonics -- a plain nterms=2 LS fit uses the same insufficient
    2-harmonic model that was found to give the wrong answer on real TESS data (see above).

    Also rejects points that don't fit the phase-folded model at the adopted period (bad
    frames -- see reject_outliers in lightcurve_prototype.py) before finalising anything,
    and returns the cleaned (mjd, rel_flux, rel_flux_err) alongside the period result so the
    caller plots/reports on the same cleaned data the period was determined from.

    fap2 (false alarm probability at the adopted period) alone is NOT sufficient -- caught the
    hard way on real full-27-55-sector data: with thousands of points, fap2 can reach ~0 purely
    from sample size even when the "signal" is noise-level. The actual failure mode found: on
    sparse, widely-gapped multi-visit objects, disambiguate_period's chi2 comparison can prefer
    a long-period harmonic (2P/3P) that "connects the dots" between a few far-apart visits
    better than the true period does, even though that harmonic carries essentially zero real
    periodogram power. Diagnostic: power_ratio compares the adopted period's power against the
    naive (pre-harmonic-multiplication) peak's power, using the SAME nterms=2 model on the SAME
    outlier-cleaned data for both (an earlier version compared nterms=1-on-raw-data against
    nterms=2-on-cleaned-data and produced false positives on good objects whose period was never
    even overridden -- the two models' power isn't on a directly comparable scale). Confirmed on
    9 real bad objects ((4942) Munroe, (5118) Elnapoul, (7701) Zrzavy, (9946) 1990 ON2,
    (11921) Mitamasahiro, (12143) Harwit, (37373) 2001 VM34, (47429) 1999 XN172,
    (78681) 2002 TK136), all of which either adopted a 2P/3P harmonic that collapsed to
    power_ratio in [0.003, 0.055] once corrected, or (2 of the 9) were caught upstream by the
    ls_model guard in disambiguate_period falling back to the naive peak on its own -- against
    6 confirmed-good controls (3 with an originally strong, unmodified peak; 2 with sparse/
    gapped sampling similar to the bad cases but a genuine underlying signal) with zero false
    positives. power_ratio below RELIABLE_MIN_POWER_RATIO disqualifies an object regardless of
    its fap2. Also pass ls1 into disambiguate_period so it can reject a harmonic candidate
    outright if ITS OWN nterms=1 power is already far below the naive peak's, as a complementary
    (not sufficient alone) guard at the point of selection."""
    ls1 = LombScargle(mjd, rel_flux, rel_flux_err, nterms=1)
    freq1, power1 = ls1.autopower(minimum_frequency=1.0 / MAX_PERIOD_DAYS,
                                    maximum_frequency=1.0 / MIN_PERIOD_DAYS,
                                    samples_per_peak=SAMPLES_PER_PEAK)
    i1 = np.argmax(power1)
    # Peak significance measured against THIS object's own periodogram background. A flat
    # power value is not comparable between objects -- it depends on point count and
    # normalisation, so a clean lightcurve is reliable at a low absolute power while a noisy
    # one is meaningless at a higher one. Measured against LCDB the raw value is actually
    # INVERTED (78.3% agreement below 0.005, 10.4% above 0.2), i.e. it was only acting as a
    # backwards proxy for point count; peak/median reaches AUC 0.846 versus 0.734 raw.
    _pmed = float(np.median(power1))
    _pmad = 1.4826 * float(np.median(np.abs(power1 - _pmed)))
    _p99 = float(np.percentile(power1, 99))
    _pk = float(power1[i1])
    pk_over_med = _pk / _pmed if _pmed > 0 else np.nan
    pk_snr = (_pk - _pmed) / _pmad if _pmad > 0 else np.nan
    pk_prom = _pk / _p99 if _p99 > 0 else np.nan
    # sub-grid peak refinement (see refine_peak_frequency): the periodogram is sampled on a
    # discrete grid, so the true maximum lies between grid points. Reduces the median
    # fractional period error by ~9% overall, +12.6% at 4-8hr, measured against LCDB.
    freq1_refined = refine_peak_frequency(freq1, power1, int(i1))
    period1 = 1.0 / freq1_refined
    period1_hr = period1 * 24.0
    naive_power = float(power1[i1])   # peak HEIGHT stays the on-grid value; only the LOCATION
    # is refined -- interpolating the height would inflate it slightly and it is used for
    # comparisons against other on-grid powers.

    best_period_hr, best_name, candidates, chi2_by_name = disambiguate_period(
        mjd, rel_flux, rel_flux_err, period1_hr, n_harm=DISAMBIGUATION_N_HARM,
        ls_model=ls1, min_power_ratio=0.5, max_period_hr=MAX_PERIOD_DAYS * 24.0)

    keep = reject_outliers(mjd, rel_flux, rel_flux_err, best_period_hr, n_harm=DISAMBIGUATION_N_HARM)
    n_rejected = int((~keep).sum())

    ls2 = LombScargle(mjd[keep], rel_flux[keep], rel_flux_err[keep], nterms=2)
    freq2, power2 = ls2.autopower(minimum_frequency=1.0 / MAX_PERIOD_DAYS,
                                    maximum_frequency=1.0 / MIN_PERIOD_DAYS,
                                    samples_per_peak=SAMPLES_PER_PEAK)
    power2_best = float(np.interp(1.0 / (best_period_hr / 24.0), freq2[::-1], power2[::-1]))
    # FAP is evaluated at the PERIODOGRAM PEAK, not at the adopted period. Since harmonic
    # selection now defaults to 2P, the adopted period is usually the FUNDAMENTAL, where a
    # double-peaked rotator carries almost no power by construction (median ~2% of the peak).
    # Evaluating there produced fap=1.0 for 63% of objects -- including (85959) 1999 FV42 with
    # aov_F=339 and a peak 1785x its own noise floor -- destroying what had been the
    # fourth-ranked model feature. It was also train/serve skew: the classifier was fitted on
    # fap values from the old default-to-P logic, where adopted period == peak for most
    # objects. "Is there a significant periodicity at all" is a question about the peak.
    fap2 = float(ls1.false_alarm_probability(float(power1[i1])))
    double_peaked = best_name in ("2P", "3P")
    # power_ratio MUST compare like-for-like: the same nterms=2 model on the same
    # outlier-cleaned data, evaluated at both the adopted period and the naive period1 --
    # comparing against power1 (nterms=1, on the RAW pre-rejection data) produced false
    # positives on real objects that never even had their period overridden (best_name=='P'),
    # since nterms=1 and nterms=2 power aren't on a directly comparable scale even for a
    # perfectly good, unchanged period. Confirmed on 9 real bad objects + 6 real good controls
    # (3 with best_name=='P' and originally-strong ls_power) after this fix: 0 false positives,
    # stable result across power-ratio thresholds 0.1-0.5.
    power2_at_naive = float(np.interp(1.0 / period1, freq2[::-1], power2[::-1]))
    power_ratio = (power2_best / power2_at_naive) if power2_at_naive > 0 else 0.0

    # Phase coverage: fold the cleaned data at the adopted period and find the largest gap
    # between consecutive points in phase (with wraparound) -- a "reliable" period fit from a
    # handful of clustered visits can leave a large fraction of the rotation completely
    # unsampled (found on real data: (543719) 2014 OH373 and (890029) 2013 EF87, both with
    # ~45% of the phase cycle -- max_phase_gap~0.45 -- never observed at all, despite passing
    # every other check). A period constrained by dense phase coverage is far more trustworthy
    # than one from the same point COUNT concentrated in one arc of the cycle.
    if keep.sum() > 1:
        phase = (mjd[keep] % (best_period_hr / 24.0)) / (best_period_hr / 24.0)
        phase_sorted = np.sort(phase)
        gaps = np.diff(np.concatenate([phase_sorted, [phase_sorted[0] + 1.0]]))
        max_phase_gap = float(gaps.max())
    else:
        max_phase_gap = 1.0

    return dict(keep=keep, n_rejected=n_rejected,
                freq1=freq1, power1=power1, period1=period1,
                freq2=freq2, power2=power2, period2=best_period_hr / 24.0,
                power2_best=power2_best, naive_power=naive_power, power_ratio=power_ratio,
                pk_over_med=pk_over_med, pk_snr=pk_snr, pk_prom=pk_prom,
                max_phase_gap=max_phase_gap,
                fap2=fap2, double_peaked=bool(double_peaked),
                best_name=best_name, chi2_by_name=chi2_by_name)


def refine_peak_frequency(freq, power, i, half=3):
    """Sub-grid periodogram peak location by fitting a Gaussian to the peak.

    The Lomb-Scargle is evaluated on a discrete frequency grid, so the true maximum falls
    between samples. A Gaussian in power is a parabola in log(power), so this fits a parabola
    to log(power) over +/-half grid points and returns the vertex -- using more of the peak
    shape than 3-point parabolic interpolation, and so less sensitive to one noisy sample.

    Measured against LCDB on 4996 objects already on the correct harmonic, this cuts the median
    fractional period error from 0.00057 to 0.00052 (~9%), best at short periods (+12.6% at
    4-8hr) and negligible above 72hr. It is NOT a large win here because grid quantisation is
    not the limiting factor -- a half grid step is ~0.4% in period while the observed error is
    0.057%, so peak position is already determined far better than the grid spacing. (In
    long-baseline variable-star work, where thousands of cycles make the peak much narrower
    than the grid, interpolation dominates the error budget and the gain is far larger.)

    Falls back to the on-grid frequency if the fit is degenerate or the vertex lands outside
    the fitted window.
    """
    n = len(freq)
    if i <= 0 or i >= n - 1:
        return freq[i]
    lo, hi = max(0, i - half), min(n, i + half + 1)
    x, y = freq[lo:hi], power[lo:hi]
    good = y > 0
    if good.sum() >= 3:
        try:
            c = np.polyfit(x[good], np.log(y[good]), 2)
            if c[0] < 0:                       # must be concave to have a maximum
                v = -c[1] / (2.0 * c[0])
                if x.min() <= v <= x.max():
                    return float(v)
        except Exception:
            pass
    # fall back to 3-point parabolic interpolation, then to the grid point itself
    y0, y1, y2 = power[i - 1], power[i], power[i + 1]
    den = y0 - 2 * y1 + y2
    if den != 0:
        d = 0.5 * (y0 - y2) / den
        if abs(d) <= 1:
            return float(freq[i] + d * (freq[i + 1] - freq[i - 1]) / 2.0)
    return float(freq[i])


def aov_F_statistic(phase, flux, n_bins=15):
    """Analysis-of-Variance (Schwarzenberg-Czerny-style) periodicity statistic: bin the
    phase-folded lightcurve into n_bins and compute the one-way ANOVA F-ratio of between-bin
    variance to within-bin variance. Found (500-object manual QC sample) to be the single
    strongest general discriminator between visually good and bad phase folds -- unlike raw
    Lomb-Scargle power, it's self-consistent per object and not distorted by differing point
    counts/normalisation across objects."""
    bins = np.clip((phase * n_bins).astype(int), 0, n_bins - 1)
    grand_mean = flux.mean()
    ss_between = 0.0
    ss_within = 0.0
    k_used = 0
    for b in range(n_bins):
        m = bins == b
        n_b = m.sum()
        if n_b < 2:
            continue
        k_used += 1
        bin_mean = flux[m].mean()
        ss_between += n_b * (bin_mean - grand_mean) ** 2
        ss_within += ((flux[m] - bin_mean) ** 2).sum()
    if k_used < 2 or ss_within <= 0:
        return np.nan
    dof_between = k_used - 1
    dof_within = len(flux) - k_used
    if dof_within <= 0:
        return np.nan
    return (ss_between / dof_between) / (ss_within / dof_within)


def amp_snr_statistic(phase, flux, ferr, n_harm=4):
    """Fitted-model peak-to-peak amplitude over the median formal per-point error -- catches
    amplitude-below-noise cases FAP alone misses (FAP can reach ~0 from sample size alone even
    for a noise-level "signal" once n is in the thousands)."""
    A = _fourier_design(phase, n_harm)
    w = 1.0 / ferr
    coeffs, *_ = np.linalg.lstsq(A * w[:, None], flux * w, rcond=None)
    phase_grid = np.linspace(0, 1, 300)
    model = _fourier_design(phase_grid, n_harm) @ coeffs
    return float((model.max() - model.min()) / np.median(ferr))


def n_phase_coverings_statistic(mjd, period_days, max_gap=0.1, min_pts=10):
    """Count INDEPENDENT EPOCHS that each, on their own, sample the full rotation phase well.

    This measures the thing that actually matters for trusting a period: how many separate
    times did we independently observe the whole phase curve? One epoch covering the phase is
    a shape; several agreeing epochs is a measurement. It is the scale-aware replacement for
    an earlier "well covered CYCLES" version that counted individual rotation cycles with >=10
    points and <=10% intra-cycle phase gap.

    Why that earlier version had to be replaced: it fixed the unit of analysis at exactly one
    rotation, which is only achievable in a narrow period band. Measured on the real catalogue,
    the fraction of objects that could even HAVE 10 points inside a single cycle was 0.1% below
    2hr, 4.6% at 2-4hr and 15.7% at 4-8hr -- so it returned 0 for essentially every short-period
    object no matter how good the data was (e.g. (40119) 1998 QB23: 2224 points across 660
    cycles, old statistic = 0), while at very long periods the <=10% intra-cycle gap is
    unreachable because the spacecraft does not observe continuously. It carried real
    information only above ~24hr.

    The fix is to let the epoch length adapt: an epoch is max(one period, enough elapsed time
    to plausibly contain min_pts samples). For a short period a single visit spans many
    rotations and densely samples phase, so the visit is the natural epoch; for a long period
    the epoch collapses back to about one cycle, recovering the old (correct, in that regime)
    behaviour. Same quantity, measured in units the data can actually deliver at any period.

    Still not a standalone gate -- it is one feature among several."""
    if len(mjd) < min_pts or not np.isfinite(period_days) or period_days <= 0:
        return 0
    order = np.sort(mjd)
    dt = np.median(np.diff(order)) if len(order) > 1 else 0.0
    # epoch long enough to hold min_pts samples at the typical cadence, but never shorter
    # than one rotation (below that an "epoch" could not cover the phase even in principle)
    window = max(period_days, dt * min_pts * 2) if dt > 0 else period_days
    widx = np.floor((mjd - mjd.min()) / window).astype(int)
    phase = (mjd % period_days) / period_days
    good = 0
    for w in np.unique(widx):
        m = widx == w
        if m.sum() < min_pts:
            continue
        ph = np.sort(phase[m])
        gaps = np.diff(np.concatenate([ph, [ph[0] + 1.0]]))
        if gaps.max() <= max_gap:
            good += 1
    return good


def loo_worst_chi2_statistic(mjd, period_days, flux, ferr, visit, n_harm=4):
    """Leave-one-VISIT-out cross-validation: fit the n_harm-harmonic model from all visits but
    one, predict the held-out visit's own points from that model, and take the worst (max)
    per-visit held-out reduced chi2. Catches "each visit looks fine alone but visits don't
    agree with each other" -- the real signature of a period that's subtly wrong: even a tiny
    period error compounds into large phase drift after many cycles between widely-separated
    visits, confirmed visually on multiple real objects (see project QC notes). Returns NaN
    when there are fewer than 3 visits (leave-one-out needs enough left to fit from, and a
    single held-out visit against only one remaining visit isn't a meaningful cross-check).
    CAUTION (do not use in isolation): the raw/absolute form here has a real blind spot for
    extremely high-SNR objects, where tiny formal errors make ANY model imperfection read as a
    huge chi2 even at the correct period -- always combine with amp_snr/aov_F floors, never
    gate on this alone."""
    phase = (mjd % period_days) / period_days
    uvisits = np.unique(visit)
    if len(uvisits) < 3:
        return np.nan
    worst = 0.0
    for v in uvisits:
        held_out = visit == v
        fit_mask = ~held_out
        if fit_mask.sum() < 2 * (2 * n_harm + 1) or held_out.sum() == 0:
            continue
        A_fit = _fourier_design(phase[fit_mask], n_harm)
        w_fit = 1.0 / ferr[fit_mask]
        coeffs, *_ = np.linalg.lstsq(A_fit * w_fit[:, None], flux[fit_mask] * w_fit, rcond=None)
        A_held = _fourier_design(phase[held_out], n_harm)
        pred = A_held @ coeffs
        resid = flux[held_out] - pred
        chi2 = np.sum((resid / ferr[held_out]) ** 2)
        dof = held_out.sum()
        if dof > 0:
            worst = max(worst, chi2 / dof)
    return worst if worst > 0 else np.nan


# ---- Multi-branch is_reliable, derived from manual QC review of a 500-object random sample
# (kept/deleted by eye, then reverse-engineered into quantitative thresholds) -- see
# project_asteroid_rotation_qc.md. No single axis-aligned threshold on any one statistic
# achieves both high recall and zero false positives: the good/bad classes genuinely overlap
# on every individual axis (confirmed: an unconstrained decision tree only reaches 100% train
# accuracy by memorising ~10-point leaves). The right structure is several distinct branches,
# each a minimal combination targeting one specific failure/success mode, OR'd together; every
# branch was calibrated by grid search requiring ZERO false positives against the full
# accumulated confirmed-bad object list before accepting any recall gain (79.7% recall, 100%
# precision on the labeled sample as of the last validated rule).
def branch_is_reliable(aov_F, amp_snr, max_phase_gap, n_cycles_observed, n_points, period_hr,
                         loo_worst_chi2, n_well_covered_cycles=None):
    loo_ok_loose = np.isnan(loo_worst_chi2) or loo_worst_chi2 < 8
    loo_ok_tight = np.isnan(loo_worst_chi2) or loo_worst_chi2 < 2
    # A_old originally had no cross-visit corroboration check at all -- found to let through
    # (14920) 1994 PE33 (aov_F=178, amp_snr=8.3, max_phase_gap=0.001 -- looks complete only
    # because 5 disjoint few-day visits happen to land at different phases by chance, each
    # visit alone only ever seeing a fragment of one cycle) despite no individual rotation
    # ever being confirmed. A blind loo_ok_loose gate is too blunt: it also broke (851)
    # Zeissia (amp_snr=35, genuinely clean) via loo_worst_chi2=380, the documented high-SNR
    # blind spot (tiny formal errors make ANY model imperfection read as a huge chi2).
    # n_well_covered_cycles is the more direct fix: PE33 has only 1 individually well-sampled
    # cycle, Zeissia has 95 -- so require >=2 directly rather than gating loo by amp_snr (an
    # indirect proxy for the same "is this corroborated by more than one real observation of
    # the shape" question). Falls back to the old loo-based gate when n_well_covered_cycles
    # isn't available (e.g. validation code not yet passing it through).
    # n_well_covered_cycles and the loo/amp_snr escape are NOT interchangeable -- confirmed
    # the hard way in real production data: (4942) Munroe has n_well_covered_cycles=4 (passes
    # easily) but loo_worst_chi2=25.7 and amp_snr=14.6 (just under the 15.0 escape) -- treating
    # n_well_covered_cycles>=2 as a REPLACEMENT for the loo check let a confirmed-bad object
    # straight through on a completely different, unrelated basis. Both checks must hold:
    # n_well_covered_cycles catches PE33-style "aggregate coverage looks complete but no
    # single cycle is corroborated"; loo/amp_snr catches Munroe-style "each cycle looks
    # locally fine but visits don't cross-validate against each other". Neither substitutes
    # for the other.
    loo_or_overwhelming = amp_snr >= 15.0 or loo_ok_loose
    well_covered_ok = n_well_covered_cycles is None or n_well_covered_cycles >= 2
    corroborated = loo_or_overwhelming and well_covered_ok
    branch_A_old = (aov_F >= 150 and amp_snr >= 4 and max_phase_gap <= 0.1 and n_cycles_observed >= 2.0
                      and corroborated)
    branch_A_new = (aov_F >= 90 and amp_snr >= 6 and max_phase_gap <= 0.03 and n_cycles_observed >= 2.0
                     and loo_ok_loose)
    # loo_worst_chi2<50 predates the systematic regression-test discipline and was never
    # actually checked against a real confirmed-bad object -- found the hard way on real
    # production data: (37373) 2001 VM34 (a confirmed wrong-period/tumbler-suspect object)
    # cleared it easily at loo=8.36. Tightened to match the other loose-tier branches (<8).
    branch_B = (aov_F >= 21.9 and max_phase_gap <= 0.03 and n_points >= 700 and period_hr < 80
                 and amp_snr >= 1.3 and n_cycles_observed >= 2.0
                 and loo_ok_loose)
    branch_D = (aov_F >= 30 and max_phase_gap <= 0.01 and n_points >= 1000 and amp_snr >= 1.5
                 and n_cycles_observed >= 2.0 and loo_ok_loose)
    branch_E = (aov_F >= 15 and n_cycles_observed >= 1000 and max_phase_gap <= 0.005 and n_points >= 3000)
    branch_F = (aov_F >= 12 and max_phase_gap <= 0.01 and amp_snr >= 1.4 and n_cycles_observed >= 5
                 and loo_ok_tight)
    return bool(branch_A_old or branch_A_new or branch_B or branch_D or branch_E or branch_F)


def plot_asteroid(designation, track, outpath, h_mag=None, sbdb_entry=None):
    track = track.sort_values("mjd").reset_index(drop=True)
    n = len(track)
    n_visits = track["visit"].nunique()
    n_sectors = track["sector"].nunique()
    median_n_stack = float(track["n_frames"].median())
    frac_stacked = float((track["n_frames"] > 1).mean())

    result = None
    if n >= MIN_POINTS_FOR_PERIODOGRAM:
        result = periodogram_with_harmonic_check(track["mjd"].values, track["rel_flux"].values,
                                                    track["rel_flux_err"].values)

    clean_track = track[result["keep"]] if result is not None else track
    plot_mjd = clean_track["mjd"].values
    plot_flux = clean_track["rel_flux"].values
    plot_ferr = clean_track["rel_flux_err"].values

    # Observation baseline actually covered by the cleaned data this period was fit to -- a
    # period longer than (or comparable to) this baseline is not a measured periodicity, it's
    # an unconstrained fit to less than one full cycle (see RELIABLE_MIN_CYCLES above).
    baseline_hr = float((plot_mjd.max() - plot_mjd.min()) * 24.0) if len(plot_mjd) > 1 else 0.0
    period_hr = result["period2"] * 24 if result else np.nan
    n_cycles_observed = (baseline_hr / period_hr) if (result is not None and period_hr > 0) else np.nan

    # QC statistics -- computed here (before plotting) so the figure title can report the
    # verdict and score directly.
    phase_angle_range = float(clean_track["phase_angle_deg"].max() - clean_track["phase_angle_deg"].min()) \
        if clean_track["phase_angle_deg"].notna().any() else np.nan

    aov_F = amp_snr = loo_worst_chi2 = n_phase_coverings = np.nan
    if result is not None and len(plot_mjd) >= RELIABLE_MIN_POINTS:
        period_days = result["period2"]
        phase_for_qc = (plot_mjd % period_days) / period_days
        aov_F = aov_F_statistic(phase_for_qc, plot_flux)
        amp_snr = amp_snr_statistic(phase_for_qc, plot_flux, plot_ferr)
        loo_worst_chi2 = loo_worst_chi2_statistic(plot_mjd, period_days, plot_flux, plot_ferr,
                                                    clean_track["visit"].values)
        n_phase_coverings = n_phase_coverings_statistic(plot_mjd, period_days)

    # Learned reliability score (see RELIABILITY_* above). Hard physical prerequisites are kept
    # as explicit gates rather than delegated to the model -- a period that was never computed,
    # or was fit to fewer than two observed rotations, is not a measurement regardless of what
    # any statistic says. Everything else is left to the classifier.
    rel_score = np.nan
    if result is not None and np.isfinite(aov_F):
        rel_score = reliability_score(dict(
            aov_F=aov_F, amp_snr=amp_snr, max_phase_gap=result["max_phase_gap"],
            n_cycles_observed=n_cycles_observed, n_points=len(clean_track),
            loo_worst_chi2=loo_worst_chi2, power_ratio=result["power_ratio"],
            fap=result["fap2"],
            pk_over_med=result["pk_over_med"], pk_prom=result["pk_prom"], pk_snr=result["pk_snr"],
            n_visits=n_visits, n_sectors=n_sectors, baseline_hr=baseline_hr,
            n_outliers_rejected=result["n_rejected"], frac_points_stacked=frac_stacked,
            phase_angle_range_deg=phase_angle_range, median_n_stack=median_n_stack,
            n_phase_coverings=n_phase_coverings))

    prereq = bool(
        result is not None
        and len(clean_track) >= RELIABLE_MIN_POINTS
        and np.isfinite(n_cycles_observed)
        and n_cycles_observed >= RELIABLE_MIN_CYCLES
        and np.isfinite(aov_F) and np.isfinite(amp_snr)
        # v4 out-of-distribution floors -- see RELIABILITY_MIN_POINTS_FLOOR above
        and len(clean_track) >= RELIABILITY_MIN_POINTS_FLOOR
        and aov_F >= RELIABILITY_MIN_AOVF_FLOOR
    )
    if np.isfinite(rel_score):
        is_reliable = bool(prereq and rel_score >= RELIABILITY_SCORE_MIN)
    else:
        # no model available -- fall back to the legacy branch rule so the run still completes
        is_reliable = bool(prereq and branch_is_reliable(
            aov_F, amp_snr, result["max_phase_gap"], n_cycles_observed,
            len(clean_track), period_hr, loo_worst_chi2, n_phase_coverings))
    verdict_str = (f"PASS (score={rel_score:.2f})" if is_reliable else
                    f"FAIL (score={rel_score:.2f})") if np.isfinite(rel_score) else (
                    "PASS (is_reliable)" if is_reliable else "FAIL (not reliable)")

    if SAVE_PLOTS:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 12.3))

        ax1.errorbar(plot_mjd, plot_flux, yerr=plot_ferr,
                     fmt="o", ms=2.5, alpha=0.08, elinewidth=0.3, color="C0", ecolor="C0", zorder=2)
        ax1.axhline(1.0, color="0.3", lw=0.8, ls="--")
        ax1.set_xlabel("MJD")
        ax1.set_ylabel("relative flux")
        h_str = f", H={h_mag:.1f}" if h_mag is not None else ""
        type_str = f", type={sbdb_entry['spec_type']}" if sbdb_entry and sbdb_entry.get("spec_type") else ""
        stack_str = f", median {median_n_stack:.0f} frames/pt ({frac_stacked:.0%} stacked)" if median_n_stack > 1 or frac_stacked > 0 else ""
        rej_str = f", {result['n_rejected']} outliers rejected" if result is not None and result["n_rejected"] else ""
        title_color = "darkgreen" if is_reliable else "firebrick"
        ax1.set_title(f"{designation} -- {verdict_str}  ({len(plot_mjd)} points >= {MIN_SNR:.0f}sig, {n_visits} visits, "
                       f"{n_sectors} sector(s){h_str}{type_str}{stack_str}{rej_str})",
                       fontsize=9, color=title_color)

        if result is not None:
            period_hr_2 = result["period2"] * 24
            period_hr_1 = result["period1"] * 24
            ax2.plot(1.0 / result["freq1"] * 24, result["power1"], color="0.6", lw=0.7, label="1-term (naive)")
            ax2.plot(1.0 / result["freq2"] * 24, result["power2"], color="C1", lw=0.9, label="nterms=2 (display only)")
            ax2.axvline(period_hr_2, color="C3", lw=1.1, ls="--",
                        label=f"adopted ({result['best_name']} of naive) P={period_hr_2:.2f} hr")
            if result["double_peaked"]:
                ax2.axvline(period_hr_1, color="C7", lw=0.8, ls=":", label=f"half-period alias={period_hr_1:.2f} hr")
            pub_rot_per = sbdb_entry.get("rot_per_hr") if sbdb_entry else None
            if pub_rot_per and np.isfinite(pub_rot_per):
                for cand in (pub_rot_per, pub_rot_per * 2):
                    if MIN_PERIOD_DAYS * 24 <= cand <= MAX_PERIOD_DAYS * 24:
                        ax2.axvline(cand, color="C2", lw=0.8, ls="-.", alpha=0.7)
                ax2.axvline(np.nan, color="C2", lw=0.8, ls="-.", label=f"published P={pub_rot_per:.2f} hr")
            ax2.set_xscale("log")
            ax2.set_xlabel("period (hours)")
            ax2.set_ylabel("Lomb-Scargle power")
            dp_str = "likely double-peaked" if result["double_peaked"] else "likely single-peaked"
            ax2.set_title(f"P={period_hr_2:.2f} hr (nterms=2 power={result['power2_best']:.3f}, FAP={result['fap2']:.2g}) -- {dp_str}",
                           fontsize=9)
            ax2.legend(loc="upper right", fontsize=6)

            period_days = result["period2"]
            phase = (plot_mjd % period_days) / period_days
            for offset in (0, 1):
                ax3.errorbar(phase + offset, plot_flux, yerr=plot_ferr,
                             fmt="o", ms=3, alpha=0.1, elinewidth=0.5, color="C0", ecolor="C0", zorder=2)
            A = _fourier_design(phase, DISAMBIGUATION_N_HARM)
            w = 1.0 / plot_ferr
            coeffs, *_ = np.linalg.lstsq(A * w[:, None], plot_flux * w, rcond=None)
            phase_grid = np.linspace(0, 1, 300)
            model_flux = _fourier_design(phase_grid, DISAMBIGUATION_N_HARM) @ coeffs
            for offset in (0, 1):
                ax3.plot(phase_grid + offset, model_flux, color="C3", lw=1.3, zorder=3)
            ax3.axvline(1.0, color="0.5", lw=0.6, ls=":")
            ax3.set_xlabel(f"phase (P = {period_hr_2:.2f} hr)")
            ax3.set_ylabel("relative flux")
            ax3.set_xlim(0, 2)
            ax3.set_title(f"{verdict_str} -- aov_F={aov_F:.1f}, amp_snr={amp_snr:.2f}, "
                           f"loo_worst_chi2={loo_worst_chi2:.1f}, n_cycles={n_cycles_observed:.1f}, "
                           f"n_phase_coverings={n_phase_coverings:.0f}",
                           fontsize=8, color=title_color)

            # Phase-binned lightcurve, using the EXACT same bins aov_F_statistic itself computes
            # from -- shows what the F-test actually sees (the between-bin signal aov_F measures)
            # on its own natural scale, since that signal is often invisible against the full
            # raw-point scatter in ax3 (aov_F can be small/large independent of how noisy any
            # single raw point is). y-limits are set from the binned means' own range (with
            # padding), not the raw data's range, so a real but modest between-bin swing is
            # actually visible rather than flattened into a thin line.
            AOV_N_BINS = 15
            bin_idx = np.clip((phase * AOV_N_BINS).astype(int), 0, AOV_N_BINS - 1)
            bin_centers, bin_means, bin_sems = [], [], []
            for b in range(AOV_N_BINS):
                m = bin_idx == b
                if m.sum() < 2:
                    continue
                bin_centers.append((b + 0.5) / AOV_N_BINS)
                bin_means.append(plot_flux[m].mean())
                bin_sems.append(plot_flux[m].std(ddof=1) / np.sqrt(m.sum()))
            bin_centers = np.array(bin_centers)
            bin_means = np.array(bin_means)
            bin_sems = np.array(bin_sems)
            for offset in (0, 1):
                ax4.errorbar(bin_centers + offset, bin_means, yerr=bin_sems,
                              fmt="o-", ms=5, lw=1.2, color="C0", ecolor="C0", capsize=2, zorder=3)
            ax4.axvline(1.0, color="0.5", lw=0.6, ls=":")
            if len(bin_means):
                pad = 0.15 * (bin_means.max() - bin_means.min() + 2 * bin_sems.max())
                pad = max(pad, 1e-6)
                ax4.set_ylim(bin_means.min() - bin_sems.max() - pad, bin_means.max() + bin_sems.max() + pad)
            ax4.set_xlim(0, 2)
            ax4.set_xlabel(f"phase (P = {period_hr_2:.2f} hr) -- {AOV_N_BINS}-bin means")
            ax4.set_ylabel("relative flux (binned)")
            ax4.set_title("phase-binned lightcurve (what aov_F measures) -- y-limits set by the binned data",
                            fontsize=8)
        else:
            ax2.text(0.5, 0.5, f"insufficient data for periodogram (n={n} < {MIN_POINTS_FOR_PERIODOGRAM})",
                      ha="center", va="center", transform=ax2.transAxes, fontsize=9)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3.text(0.5, 0.5, "no period available to fold on",
                      ha="center", va="center", transform=ax3.transAxes, fontsize=9)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax4.set_xticks([])
            ax4.set_yticks([])

        fig.tight_layout()
        fig.savefig(outpath, dpi=130)
        plt.close(fig)

    real_diam = sbdb_entry.get("diameter_km") if sbdb_entry else np.nan
    real_albedo = sbdb_entry.get("albedo") if sbdb_entry else np.nan
    spec_type = sbdb_entry.get("spec_type") if sbdb_entry else None
    pub_rot_per = sbdb_entry.get("rot_per_hr") if sbdb_entry else np.nan

    # period_hr, n_cycles_observed, aov_F, amp_snr, loo_worst_chi2, is_reliable were already
    # computed above (before plotting), so the figure title can show pass/fail directly.

    # write a per-object clean-lightcurve CSV (post-outlier-rejection, geometry included) for
    # any object with a confident period and enough points -- input for shape_model_pipeline.py
    wrote_clean_lc = False
    if is_reliable and len(clean_track) >= SHAPE_MODEL_MIN_POINTS:
        os.makedirs(CLEAN_LC_DIR, exist_ok=True)
        clean_path = f"{CLEAN_LC_DIR}/{safe_name(designation)}_clean_lc.csv"
        clean_track[["mjd", "flux", "e_flux", "ra", "dec", "delta_au", "phase_angle_deg", "sector", "visit"]].to_csv(
            clean_path, index=False)
        wrote_clean_lc = True

    return dict(designation=designation, n_points=len(plot_mjd), n_points_before_rejection=n,
                n_outliers_rejected=(result["n_rejected"] if result else 0),
                n_visits=n_visits, n_sectors=n_sectors, magnitude_H=h_mag,
                period_hr=period_hr,
                period_hr_naive=(result["period1"] * 24 if result else np.nan),
                ls_power=(result["power2_best"] if result else np.nan),
                naive_power=(result["naive_power"] if result else np.nan),
                power_ratio=(result["power_ratio"] if result else np.nan),
                pk_over_med=(result["pk_over_med"] if result else np.nan),
                pk_prom=(result["pk_prom"] if result else np.nan),
                pk_snr=(result["pk_snr"] if result else np.nan),
                max_phase_gap=(result["max_phase_gap"] if result else np.nan),
                fap=(result["fap2"] if result else np.nan),
                baseline_hr=baseline_hr, n_cycles_observed=n_cycles_observed,
                aov_F=aov_F, amp_snr=amp_snr, loo_worst_chi2=loo_worst_chi2,
                n_phase_coverings=n_phase_coverings, reliability_score=rel_score,
                is_reliable=is_reliable,
                double_peaked=(result["double_peaked"] if result else None),
                diameter_km_real=real_diam, albedo_real=real_albedo,
                spec_type=spec_type, published_rot_per_hr=pub_rot_per,
                min_snr=MIN_SNR, median_n_stack=median_n_stack, frac_points_stacked=frac_stacked,
                phase_angle_range_deg=phase_angle_range, wrote_clean_lc=wrote_clean_lc)


def _broad_taxonomic_class(spec_type):
    """Collapse Tholen/SMASS sub-types (Sk, Sq, Sr, Sa, ...) to their broad complex letter for
    a legible categorical colour scheme -- otherwise dozens of near-identical sub-types would
    each need their own colour."""
    if not spec_type or not isinstance(spec_type, str):
        return "unknown"
    return spec_type[0].upper()


def make_spin_size_plot(summary):
    """The classic Pravec & Harris "spin barrier" diagram: rotation rate (rotations/day) vs.
    diameter (km). Uses SBDB's real measured diameter where available, falling back to the
    H-and-assumed-albedo proxy otherwise (see ASSUMED_ALBEDO) -- only objects flagged
    is_reliable (see the canonical definition above plot_asteroid) are included. Coloured by
    broad taxonomic complex where known (unknown types shown in grey); marker shape
    distinguishes single- vs. double-peaked lightcurves."""
    good = summary[summary["is_reliable"] & summary["period_hr"].notna()].copy()
    good["diameter_km_final"] = good["diameter_km_real"].where(
        good["diameter_km_real"].notna(), diameter_km(good["magnitude_H"].values))
    good = good[good["diameter_km_final"].notna()]
    if len(good) == 0:
        print("no objects meet the spin-size plot's confidence threshold -- skipping", flush=True)
        return

    diam = good["diameter_km_final"].values
    spin_rate = 24.0 / good["period_hr"].values  # rotations/day
    broad_class = good["spec_type"].apply(_broad_taxonomic_class)

    # ~18 distinct Tholen/SMASS complex letters occur, but most have <10 objects here --
    # cycling a 10-colour map (cmap(i % 10)) silently ALIASES unrelated classes onto the same
    # colour once you exceed 10 (e.g. 'A' and the 11th class 'M' both got tab10 colour 0),
    # actively misleading the reader rather than just looking busy. Fixed by folding any class
    # with fewer than MIN_CLASS_N objects into an explicit "other" bucket (distinct from
    # "unknown" = no spec_type at all) and giving every remaining class its own fixed-order hue
    # from a >10-colour qualitative map -- no modulo, so no aliasing regardless of count.
    MIN_CLASS_N = 10
    class_counts = broad_class[broad_class != "unknown"].value_counts()
    named_classes = sorted(class_counts[class_counts >= MIN_CLASS_N].index)
    broad_class = broad_class.where(broad_class.isin(named_classes) | (broad_class == "unknown"), "other")

    cmap = plt.get_cmap("tab20")
    class_colors = {c: cmap(i) for i, c in enumerate(named_classes)}
    class_colors["other"] = "#8c6d31"
    class_colors["unknown"] = "0.75"

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    # draw order matters: unknown (97% of the sample, uninformative) goes down first as
    # small/faint background texture so it can't randomly occlude the sparse, informative
    # typed points -- then "other", then each named class on top, largest/most opaque last.
    draw_order = ["unknown", "other"] + named_classes
    for cls in draw_order:
        is_unknown = cls == "unknown"
        for is_double, marker in [(False, "o"), (True, "^")]:
            mask = ((broad_class == cls).values
                     & (good["double_peaked"].fillna(False).values.astype(bool) == is_double))
            if not mask.any():
                continue
            ax.scatter(diam[mask], spin_rate[mask], color=class_colors[cls], marker=marker,
                       s=4 if is_unknown else 26, alpha=0.15 if is_unknown else 0.85,
                       edgecolors="none" if is_unknown else "0.3",
                       linewidths=0 if is_unknown else 0.3,
                       zorder=1 if is_unknown else 3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("diameter (km) -- SBDB-measured where known, else H+assumed albedo=%.2f" % ASSUMED_ALBEDO)
    ax.set_ylabel("spin rate (rotations / day)")
    ax.axhline(24 / 2.2, color="0.4", lw=0.8, ls="--", zorder=2)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=class_colors[c], markersize=7, label=c)
               for c in named_classes]
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=class_colors["other"], markersize=7,
                            label=f"other (<{MIN_CLASS_N} objects/class)"))
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=class_colors["unknown"], markersize=7, label="unknown type"))
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5", markersize=7, label="single-peaked"))
    handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="0.5", markersize=7, label="double-peaked"))
    handles.append(Line2D([0], [0], color="0.4", lw=0.8, ls="--", label="2.2 hr spin barrier"))
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, frameon=False)
    ax.set_title(f"Sectors {SECTORS[0]}-{SECTORS[-1]}: {len(good)} reliable asteroid rotation periods (FAP<{RELIABLE_FAP_MAX})")
    fig.tight_layout()
    fig.savefig(SPIN_SIZE_PATH, dpi=150)
    plt.close(fig)
    print(f"spin-size plot saved to {SPIN_SIZE_PATH} ({len(good)} objects)", flush=True)


# ---- Bucketed (out-of-core) loading path -- an ALTERNATIVE to load_all_photometry() +
# stack_to_min_snr(), not a replacement. Both paths are kept: set ASTEROID_USE_BUCKETED_LOAD=1
# to use this one; the original whole-dataset-in-memory path (load_all_photometry ->
# stack_to_min_snr -> normalise -> one big groupby) remains the default.
#
# Why: load_all_photometry collects every cut's DataFrame into a Python list, then does ONE
# pd.concat at the end -- at full 29-sector scale this held close to the entire ~400GB
# multi-sector photometry table in RAM (confirmed: peak RAM ~360GB/350G requested on 2 separate
# full runs, both essentially AT the ceiling, even after the earlier stack_to_min_snr batching
# fix). There is no real reason to ever hold the FULL survey in memory at once just to group it
# by designation -- this is the standard "doesn't fit in memory" groupby problem, solved the
# standard way: partition by key on write (hash designation into a fixed number of buckets,
# write each cut's rows straight to its bucket's file on disk, never holding more than one
# cut's data in the main process), then stream-process one bucket at a time on the read side
# (stacking/periodogram/plotting), so peak RAM is bounded by ONE BUCKET's share of the survey
# (~1/N_BUCKETS) instead of the whole thing. N_BUCKETS=256 -> ~1.5GB/bucket at this survey's
# size, vs ~400GB unified -- should let peak RAM drop from ~360GB to well under 50GB.
N_BUCKETS = 256
BUCKET_DIR = f"{OUTDIR}/_bucket_tmp"


def _bucket_of(designation_series, n_buckets=N_BUCKETS):
    """Deterministic (stable across runs/processes -- unlike Python's built-in hash(), which is
    randomised per-process by PYTHONHASHSEED) vectorised hash-to-bucket assignment, so every
    row for a given designation always lands in the same bucket regardless of which cut/worker
    processed it."""
    return pd.util.hash_pandas_object(designation_series, index=False).values % n_buckets


def _load_and_bucket_one_cut(path, bucket_dir, n_buckets=N_BUCKETS):
    """Same load/merge/geometric-correction as _load_one_cut, but instead of returning the
    full DataFrame back to the main process (the source of the memory pressure this path
    avoids), writes each non-empty bucket's slice straight to its own small parquet file on
    disk and returns only a lightweight row-count summary."""
    df = _load_one_cut(path)
    if df is None or len(df) == 0:
        return 0
    bucket = _bucket_of(df["designation"], n_buckets)
    cut_tag = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")[-120:]
    n_written = 0
    for b in np.unique(bucket):
        sub = df[bucket == b]
        if len(sub) == 0:
            continue
        bdir = f"{bucket_dir}/bucket_{b:04d}"
        os.makedirs(bdir, exist_ok=True)
        sub.to_parquet(f"{bdir}/{cut_tag}.parquet", index=False)
        n_written += len(sub)
    return n_written


def load_and_bucket_photometry(n_buckets=N_BUCKETS):
    """Parallel first pass: every cut's photometry gets loaded, merged with ephemeris, and
    geometry-corrected exactly as in load_all_photometry -- but written straight to its
    designation-hashed bucket on disk rather than accumulated in the main process. Only a
    per-cut row count flows back through the worker pool, not the data itself."""
    paths = find_photometry_cuts()
    print(f"{len(paths)} cuts with PSF photometry across sectors {SECTORS[0]}-{SECTORS[-1]} "
          f"(bucketed load, {n_buckets} buckets)", flush=True)
    if os.path.isdir(bucket_dir_check := BUCKET_DIR):
        import shutil
        shutil.rmtree(bucket_dir_check)
    os.makedirs(BUCKET_DIR, exist_ok=True)

    n_workers = n_available_workers()
    total_rows = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_and_bucket_one_cut, path, BUCKET_DIR, n_buckets): path for path in paths}
        for i, fut in enumerate(as_completed(futures)):
            try:
                total_rows += fut.result()
            except Exception as e:
                print(f"  skip {futures[fut]}: {e}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"  loaded {i+1}/{len(paths)} cuts, {total_rows} rows bucketed so far", flush=True)
    print(f"bucketing done: {total_rows} total rows across {n_buckets} buckets on disk at {BUCKET_DIR}", flush=True)
    return total_rows


def _process_one_bucket(bucket_idx, h_lookup, sbdb_by_number, sbdb_by_designation):
    """Second pass, one bucket: read back only this bucket's share of the survey (every row
    for every designation hashed into it, across all cuts/sectors -- this IS a complete,
    correct per-designation grouping, just spread across many small files instead of one big
    in-memory table), then run the existing stack_to_min_snr/normalise/plot_asteroid pipeline
    on it exactly as the original path does, just at ~1/N_BUCKETS the scale. Returns this
    bucket's summary rows; the bucket's own files are left on disk for debugging (small enough
    not to matter) rather than deleted -- clean up BUCKET_DIR manually once satisfied."""
    bdir = f"{BUCKET_DIR}/bucket_{bucket_idx:04d}"
    files = glob.glob(f"{bdir}/*.parquet")
    if not files:
        return []
    all_phot = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    stacked = _stack_bucket_serial(all_phot, MIN_SNR)
    del all_phot
    if len(stacked) == 0:
        return []
    normalised = normalise(stacked)
    del stacked

    rows = []
    for d, g in normalised.groupby("designation"):
        if len(g) < MIN_POINTS_FOR_PLOT:
            continue
        outpath = f"{LC_OUTDIR}/{safe_name(d)}_lc.png"
        try:
            sbdb_entry = match_sbdb(d, sbdb_by_number, sbdb_by_designation)
            result = plot_asteroid(d, g, outpath, h_mag=h_lookup.get(d), sbdb_entry=sbdb_entry)
            if result is not None:
                rows.append(result)
        except Exception as e:
            print(f"  FAILED {d} (bucket {bucket_idx}): {e}", flush=True)
    return rows


def _stack_bucket_serial(bucket_phot, min_snr):
    """Stack one bucket's designations serially (no nested ProcessPoolExecutor -- this already
    runs inside a bucket worker process, and a bucket's ~1/N_BUCKETS share of designations is
    small enough that per-designation stacking's own cost is no longer the bottleneck the
    original stack_to_min_snr's batched pool was built to address)."""
    out = []
    for d, g in bucket_phot.groupby("designation"):
        res = _stack_one_designation(d, g, min_snr)
        if res is not None:
            out.append(res)
    if not out:
        return pd.DataFrame(columns=["designation", "visit", "mjd", "mjd_err", "flux", "e_flux", "sig",
                                      "n_frames", "sector", "ra", "dec", "delta_au", "phase_angle_deg"])
    return pd.concat(out, ignore_index=True)


def main_bucketed():
    """Alternative entry point using the out-of-core bucketed load path -- see the module
    comment above _bucket_of for why. Produces the exact same SUMMARY_PATH/SPIN_SIZE_PATH
    outputs as main(); only the internal data-flow (and peak memory) differs. Not called by
    default -- main() picks between this and the original based on
    ASTEROID_USE_BUCKETED_LOAD."""
    if SAVE_PLOTS:
        os.makedirs(LC_OUTDIR, exist_ok=True)
    os.makedirs(CLEAN_LC_DIR, exist_ok=True)
    print("per-object PNG plots: " + ("ON" if SAVE_PLOTS else "OFF"), flush=True)

    print("loading H magnitudes...", flush=True)
    h_lookup = load_h_magnitudes()
    print(f"{len(h_lookup)} designations with H magnitude", flush=True)

    print("loading SBDB physical properties...", flush=True)
    sbdb_by_number, sbdb_by_designation = load_sbdb_physical_properties()

    load_and_bucket_photometry(N_BUCKETS)

    n_workers = n_available_workers()
    print(f"processing {N_BUCKETS} buckets across {n_workers} workers...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one_bucket, b, h_lookup, sbdb_by_number, sbdb_by_designation): b
                   for b in range(N_BUCKETS)}
        for i, fut in enumerate(as_completed(futures)):
            rows.extend(fut.result())
            print(f"  bucket {i+1}/{N_BUCKETS} done, {len(rows)} designations processed so far", flush=True)

    summary = pd.DataFrame(rows).sort_values("n_points", ascending=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    n_with_period = summary["period_hr"].notna().sum()
    n_reliable = summary["is_reliable"].sum() if "is_reliable" in summary else 0
    n_clean_lc = summary["wrote_clean_lc"].sum() if "wrote_clean_lc" in summary else 0
    print(f"done (bucketed path). {len(summary)} lightcurves saved, {n_with_period} with a computed period, "
          f"{n_reliable} flagged is_reliable, {n_clean_lc} clean-lc CSVs written.", flush=True)
    print(f"summary saved to {SUMMARY_PATH}", flush=True)

    make_spin_size_plot(summary)


# ---- Indexed (out-of-core, no data duplication) loading path -- a THIRD alternative,
# alongside the original and the bucketed path above (both kept, unchanged). Set
# ASTEROID_USE_INDEXED_LOAD=1 to use this one.
#
# Why a third path: the bucketed path fixes peak RAM (validated: 58.7GB vs the original's
# 360GB+ on real 13-sector data) but does it by physically rewriting every row to a new,
# hash-partitioned copy on disk -- confirmed on that same real test: 951,500 tiny files
# (~100KB average) for only 97GB of actual content, and Lustre's fixed per-file overhead
# multiplied that into 1.7TB written / 7.8TB read / 151M IOPS. That's solving "group by
# designation without fitting it all in RAM" the hard way.
#
# The standard way (this is a solved out-of-core groupby/join problem): build a lightweight
# INDEX of where each key's data lives, then read back only what's needed per batch --
# no duplicate copy of the data ever gets written. This works especially well here because
# each asteroid only appears in a handful of cuts (roughly one per sector it crossed, out of
# ~7500 total cuts across the survey) -- so the index itself is tiny (a few (designation,
# cut_path) rows per object, not the full per-frame photometry), and a batch of designations
# typically only needs a small fraction of all cut files, not all of them.
INDEX_DB_PATH = f"{OUTDIR}/designation_index.sqlite"
N_INDEX_BATCHES = 256


def _index_one_cut(path):
    """Read ONLY the designation column (cheap, columnar) from one cut -- no photometry, no
    ephemeris merge, none of the actual per-frame work _load_one_cut does -- just "which
    asteroids appear in this file". Returns (path, sorted unique designations) or None."""
    try:
        df = pd.read_parquet(path, columns=["designation"])
    except Exception:
        return None
    if len(df) == 0:
        return None
    return path, sorted(df["designation"].unique().tolist())


def build_designation_index():
    """Figure out which cuts each designation appears in, and record it in a small SQLite
    database on disk -- NOT a copy of the photometry itself, just a lookup table, so this is
    orders of magnitude smaller than the bucketed path's duplicate-data approach.

    PERSISTENT and INCREMENTAL, not rebuilt from scratch every run: a second table
    (indexed_cuts) tracks exactly which cut files have already been indexed, so re-running
    this (e.g. after a later sector's data lands, or just re-running the pipeline again on
    the same sectors) only scans cuts that are actually new -- the database file itself is
    never deleted, and a run that adds no new cuts does no scanning at all. This is what
    makes the index a stable, reusable artifact of the survey rather than disposable
    per-run scratch state (the bucketed path's tmp files were the latter; this must not be).

    Parallelised across NEW cuts only, exactly like load_all_photometry/load_and_bucket_
    photometry, but each task returns only a short list of designation strings, not a
    DataFrame of photometry rows."""
    import sqlite3
    paths = find_photometry_cuts()
    conn = sqlite3.connect(INDEX_DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS idx (designation TEXT, cut_path TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS indexed_cuts (cut_path TEXT PRIMARY KEY, indexed_at TEXT)")
    conn.execute("PRAGMA synchronous=OFF")
    conn.commit()

    already_indexed = {row[0] for row in conn.execute("SELECT cut_path FROM indexed_cuts")}
    new_paths = [p for p in paths if p not in already_indexed]
    print(f"{len(paths)} cuts total across sectors {SECTORS[0]}-{SECTORS[-1]}, "
          f"{len(already_indexed)} already indexed, {len(new_paths)} new to index", flush=True)

    if not new_paths:
        n_designations = conn.execute("SELECT COUNT(DISTINCT designation) FROM idx").fetchone()[0]
        conn.close()
        print(f"index already up to date: {n_designations} unique designations, "
              f"nothing to scan at {INDEX_DB_PATH}", flush=True)
        return n_designations

    n_workers = n_available_workers()
    n_pairs = 0
    now = pd.Timestamp.utcnow().isoformat()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_index_one_cut, path): path for path in new_paths}
        for i, fut in enumerate(as_completed(futures)):
            path = futures[fut]
            res = fut.result()
            if res is not None:
                _, designations = res
                conn.executemany("INSERT INTO idx VALUES (?, ?)", [(d, path) for d in designations])
                n_pairs += len(designations)
            conn.execute("INSERT OR REPLACE INTO indexed_cuts VALUES (?, ?)", (path, now))
            if (i + 1) % 500 == 0:
                conn.commit()
                print(f"  indexed {i+1}/{len(new_paths)} new cuts, {n_pairs} new (designation, cut) pairs so far",
                      flush=True)
    conn.commit()
    # rebuild the lookup index fresh each time rather than incrementally maintaining it --
    # cheap relative to the scan above, and simpler to reason about than incremental B-tree
    # updates on a table that's only ever appended to.
    conn.execute("DROP INDEX IF EXISTS idx_designation")
    conn.execute("CREATE INDEX idx_designation ON idx(designation)")
    conn.commit()
    n_designations = conn.execute("SELECT COUNT(DISTINCT designation) FROM idx").fetchone()[0]
    conn.close()
    print(f"index updated: {n_pairs} new (designation, cut) pairs from {len(new_paths)} new cuts, "
          f"{n_designations} unique designations total, saved to {INDEX_DB_PATH}", flush=True)
    return n_designations


def _process_one_index_batch(batch_designations, h_lookup, sbdb_by_number, sbdb_by_designation):
    """One batch of designations: look up (via the index) exactly which cut files any of them
    appear in, load ONLY those files (full _load_one_cut, unchanged -- reused as-is), then
    restrict to this batch's designations before stacking/normalising/plotting. A cut needed
    by multiple batches gets opened once per batch that needs it (unavoidable without
    reshuffling the data itself, which is exactly what the bucketed path did and paid for in
    I/O amplification) -- but nothing is ever duplicated on disk, and most cuts are irrelevant
    to most batches (each designation only touches a handful of cuts), so the total read
    volume across all batches stays close to a small multiple of the raw data size, not the
    80x amplification the bucketed path's tiny-file overhead produced."""
    import sqlite3
    conn = sqlite3.connect(INDEX_DB_PATH)
    placeholders = ",".join("?" * len(batch_designations))
    cut_paths = [row[0] for row in conn.execute(
        f"SELECT DISTINCT cut_path FROM idx WHERE designation IN ({placeholders})", batch_designations)]
    conn.close()
    if not cut_paths:
        return []

    frames = [df for df in (_load_one_cut(p) for p in cut_paths) if df is not None]
    if not frames:
        return []
    batch_set = set(batch_designations)
    all_phot = pd.concat(frames, ignore_index=True)
    all_phot = all_phot[all_phot["designation"].isin(batch_set)]
    if len(all_phot) == 0:
        return []

    stacked = _stack_bucket_serial(all_phot, MIN_SNR)
    del all_phot
    if len(stacked) == 0:
        return []
    normalised = normalise(stacked)
    del stacked

    rows = []
    for d, g in normalised.groupby("designation"):
        if len(g) < MIN_POINTS_FOR_PLOT:
            continue
        outpath = f"{LC_OUTDIR}/{safe_name(d)}_lc.png"
        try:
            sbdb_entry = match_sbdb(d, sbdb_by_number, sbdb_by_designation)
            result = plot_asteroid(d, g, outpath, h_mag=h_lookup.get(d), sbdb_entry=sbdb_entry)
            if result is not None:
                rows.append(result)
        except Exception as e:
            print(f"  FAILED {d}: {e}", flush=True)
    return rows


def main_indexed():
    """Third alternative entry point -- see the module comment above INDEX_DB_PATH for why.
    Produces the exact same SUMMARY_PATH/SPIN_SIZE_PATH outputs as main()/main_bucketed();
    only the internal data-flow (and I/O footprint) differs. Not called by default -- main()
    picks between the three based on ASTEROID_USE_INDEXED_LOAD / ASTEROID_USE_BUCKETED_LOAD."""
    import sqlite3
    if SAVE_PLOTS:
        os.makedirs(LC_OUTDIR, exist_ok=True)
    os.makedirs(CLEAN_LC_DIR, exist_ok=True)
    print("per-object PNG plots: " + ("ON" if SAVE_PLOTS else "OFF"), flush=True)

    print("loading H magnitudes...", flush=True)
    h_lookup = load_h_magnitudes()
    print(f"{len(h_lookup)} designations with H magnitude", flush=True)

    print("loading SBDB physical properties...", flush=True)
    sbdb_by_number, sbdb_by_designation = load_sbdb_physical_properties()

    build_designation_index()

    conn = sqlite3.connect(INDEX_DB_PATH)
    all_designations = sorted(r[0] for r in conn.execute("SELECT DISTINCT designation FROM idx"))
    conn.close()
    d_series = pd.Series(all_designations)
    batch_idx = _bucket_of(d_series, N_INDEX_BATCHES)
    batches = [d_series[batch_idx == b].tolist() for b in range(N_INDEX_BATCHES)]
    batches = [b for b in batches if b]
    print(f"{len(all_designations)} designations split into {len(batches)} batches", flush=True)

    n_workers = n_available_workers()
    print(f"processing {len(batches)} batches across {n_workers} workers...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one_index_batch, b, h_lookup, sbdb_by_number, sbdb_by_designation): i
                   for i, b in enumerate(batches)}
        for i, fut in enumerate(as_completed(futures)):
            rows.extend(fut.result())
            print(f"  batch {i+1}/{len(batches)} done, {len(rows)} designations processed so far", flush=True)

    summary = pd.DataFrame(rows).sort_values("n_points", ascending=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    n_with_period = summary["period_hr"].notna().sum()
    n_reliable = summary["is_reliable"].sum() if "is_reliable" in summary else 0
    n_clean_lc = summary["wrote_clean_lc"].sum() if "wrote_clean_lc" in summary else 0
    print(f"done (indexed path). {len(summary)} lightcurves saved, {n_with_period} with a computed period, "
          f"{n_reliable} flagged is_reliable, {n_clean_lc} clean-lc CSVs written.", flush=True)
    print(f"summary saved to {SUMMARY_PATH}", flush=True)

    make_spin_size_plot(summary)


_worker_h_lookup = None
_worker_sbdb_by_number = None
_worker_sbdb_by_designation = None


def _pool_worker_init(h_lookup, sbdb_by_number, sbdb_by_designation):
    """Each per-designation task is fully independent (its own track, its own periodogram,
    its own figure) -- this is embarrassingly parallel across the several thousand
    designations, so dispatch via a process pool rather than one core running the whole loop
    sequentially. The small lookup dicts are broadcast once per worker rather than pickled
    into every individual task."""
    global _worker_h_lookup, _worker_sbdb_by_number, _worker_sbdb_by_designation
    _worker_h_lookup = h_lookup
    _worker_sbdb_by_number = sbdb_by_number
    _worker_sbdb_by_designation = sbdb_by_designation


def _plot_worker(designation, track):
    outpath = f"{LC_OUTDIR}/{safe_name(designation)}_lc.png"
    try:
        sbdb_entry = match_sbdb(designation, _worker_sbdb_by_number, _worker_sbdb_by_designation)
        return plot_asteroid(designation, track, outpath, h_mag=_worker_h_lookup.get(designation), sbdb_entry=sbdb_entry)
    except Exception as e:
        print(f"  FAILED {designation}: {e}", flush=True)
        return None


def main():
    if _os.environ.get("ASTEROID_USE_INDEXED_LOAD", "0") == "1":
        print("ASTEROID_USE_INDEXED_LOAD=1 -- using the index-based load path (no data "
              "duplication, lower I/O than the bucketed path). Set to 0/unset otherwise.", flush=True)
        return main_indexed()

    if _os.environ.get("ASTEROID_USE_BUCKETED_LOAD", "0") == "1":
        print("ASTEROID_USE_BUCKETED_LOAD=1 -- using the out-of-core bucketed load path "
              "(lower peak RAM, more disk I/O). Set to 0/unset for the original path.", flush=True)
        return main_bucketed()

    if SAVE_PLOTS:
        os.makedirs(LC_OUTDIR, exist_ok=True)
    os.makedirs(CLEAN_LC_DIR, exist_ok=True)
    print(f"per-object PNG plots: {'ON' if SAVE_PLOTS else 'OFF (set ASTEROID_SAVE_PLOTS=1 to enable)'}", flush=True)

    print("loading H magnitudes...", flush=True)
    h_lookup = load_h_magnitudes()
    print(f"{len(h_lookup)} designations with H magnitude", flush=True)

    print("loading SBDB physical properties...", flush=True)
    sbdb_by_number, sbdb_by_designation = load_sbdb_physical_properties()

    print("loading all cuts' PSF photometry across all sectors...", flush=True)
    all_phot = load_all_photometry()
    print(f"total rows loaded: {len(all_phot)}, unique designations: {all_phot['designation'].nunique()}", flush=True)

    print(f"stacking to >= {MIN_SNR:.0f} sigma per point...", flush=True)
    stacked = stack_to_min_snr(all_phot)
    del all_phot
    print(f"{len(stacked)} points >= {MIN_SNR:.0f} sigma across {stacked['designation'].nunique()} designations "
          f"(median {stacked['n_frames'].median():.0f} raw frames/point)", flush=True)

    print("normalising per visit...", flush=True)
    normalised = normalise(stacked)
    del stacked

    print("splitting into per-designation tracks...", flush=True)
    groups = {d: g for d, g in normalised.groupby("designation") if len(g) >= MIN_POINTS_FOR_PLOT}
    del normalised
    print(f"{len(groups)} designations to process", flush=True)

    n_workers = n_available_workers()
    print(f"dispatching across {n_workers} workers...", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_pool_worker_init,
                              initargs=(h_lookup, sbdb_by_number, sbdb_by_designation)) as pool:
        futures = {pool.submit(_plot_worker, d, g): d for d, g in groups.items()}
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result is not None:
                rows.append(result)
            if (i + 1) % 200 == 0:
                print(f"  processed {i+1}/{len(groups)} designations, {len(rows)} plotted", flush=True)

    summary = pd.DataFrame(rows).sort_values("n_points", ascending=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    n_with_period = summary["period_hr"].notna().sum()
    n_reliable = summary["is_reliable"].sum() if "is_reliable" in summary else 0
    n_clean_lc = summary["wrote_clean_lc"].sum() if "wrote_clean_lc" in summary else 0
    # criteria text kept in step with the actual decision -- it previously still described the
    # FAP-based rule that was replaced by the classifier two model generations ago
    print(f"done. {len(summary)} lightcurves saved, {n_with_period} with a computed period, "
          f"{n_reliable} flagged is_reliable (v4 score>={RELIABILITY_SCORE_MIN}, "
          f">={RELIABILITY_MIN_POINTS_FLOOR} pts, aov_F>={RELIABILITY_MIN_AOVF_FLOOR}, "
          f">={RELIABLE_MIN_CYCLES} cycle(s) observed), {n_clean_lc} clean-lc CSVs written "
          f"for shape modeling.", flush=True)
    print(f"summary saved to {SUMMARY_PATH}", flush=True)

    make_spin_size_plot(summary)


if __name__ == "__main__":
    main()
