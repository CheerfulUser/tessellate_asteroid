"""Independent check of a period, using neither labels nor synthetic data: split each object's points at the median
time, run the pipeline's 1-term Lomb-Scargle search on each half separately, and ask whether BOTH halves' strongest
peaks land within TOL of the full-data peak (the naive period; the adopted one is 2x or 1x it). A real period repeats
in independent halves; a noise peak that won in the full data usually does not. Reported for the four groups of
synthetic_neighbours.csv (user label x synthetic verdict), with the chance rate from shuffling each half's fluxes.
Runs on ozstar."""
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

import sector_report_pipeline as p
from tessellate import asteroid_store as st

S = "/fred/oz335/rridden/asteroids_store"
TOL = 0.02


def peak(t, y, e):
    fr, pw = LombScargle(t, y, e).autopower(minimum_frequency=1 / p.MAX_PERIOD_DAYS, maximum_frequency=1 / p.MIN_PERIOD_DAYS,
                                            samples_per_peak=p.SAMPLES_PER_PEAK)
    return 1 / fr[np.argmax(pw)]


def one(d):
    try:
        tr = p.normalise(p.frames_to_points(p._store_frame(st.load_lightcurve(d, p.DATA_ROOT, sectors=p.SECTORS, columns=p.STORE_COLUMNS))))
        tr = tr.sort_values("mjd").reset_index(drop=True)
        tr = tr[p.preclip_mask(tr)]
        t, y, e = tr.mjd.values, tr.rel_flux.values, tr.rel_flux_err.values
        full = peak(t, y, e)
        mid = np.median(t)
        a, b = t < mid, t >= mid
        if a.sum() < 30 or b.sum() < 30:
            return dict(designation=d, error="too few points per half")
        pa, pb = peak(t[a], y[a], e[a]), peak(t[b], y[b], e[b])
        rng = np.random.default_rng(abs(hash(d)) % 2 ** 32)
        sa, sb = peak(t[a], rng.permutation(y[a]), e[a]), peak(t[b], rng.permutation(y[b]), e[b])
        m = lambda q: abs(q / full - 1) < TOL
        return dict(designation=d, full_hr=full * 24, half1_hr=pa * 24, half2_hr=pb * 24, both=m(pa) and m(pb),
                    either=m(pa) or m(pb), shuffled_both=m(sa) and m(sb))
    except Exception as ex:
        return dict(designation=d, error=f"{type(ex).__name__}: {ex}")


def main():
    g = pd.read_csv(f"{S}/synthetic_neighbours.csv")
    with mp.get_context("fork").Pool(p.n_available_workers(), maxtasksperchild=10) as pool:
        r = pd.DataFrame(pool.map(one, g.designation))
    d = g.merge(r, on="designation")
    d = d[d.error.isna()] if "error" in d else d
    d.to_csv(f"{S}/split_half_test.csv", index=False)
    print("share of objects whose period repeats in BOTH independent halves (and in either half; chance from shuffled fluxes):")
    print(d.groupby("group").agg(n=("both", "size"), both=("both", "mean"), either=("either", "mean"),
                                 chance_both=("shuffled_both", "mean")).round(3).to_string())
    G = d[d.group.str.startswith("G")]
    print(f"\nG objects repeating in both halves, by look-alike correctness: "
          + ", ".join(f"{lo}-{hi}: {G[(G.neighbour_correct >= lo) & (G.neighbour_correct < hi)].both.mean():.0%} "
                      f"(n={int(((G.neighbour_correct >= lo) & (G.neighbour_correct < hi)).sum())})"
                      for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.01))))


if __name__ == "__main__":
    main()
