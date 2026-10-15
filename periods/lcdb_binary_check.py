"""LCDB binary table (lcdb_lc_binary.csv, PDS bundle 4.0) against the TESS periods: for every LCDB binary/suspected
binary in the report, which of its periods does the pipeline period reach (within SCAN of P x {1, 2, 1/2})?
  primary    the summary (primary rotation) period
  orbit      the satellite's orbital period (median over the binary-table entries)
  secondary  a reported secondary period (median over the entries)
The closest reachable match wins. Split by deployed reliable / C + L + event dependence pass.
Runs on ozstar.   python3 -u lcdb_binary_check.py
"""
import numpy as np
import pandas as pd

import explain_score as ex

S = ex.S
SCAN = 0.08829
BIN = f"{S}/lcdb_lc_binary.csv"
OUT = f"{S}/lcdb_binary_check.csv"


def binary_table():
    lines = open(BIN).readlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Number,Name"))
    dat = next(i for i, l in enumerate(lines) if l.startswith("# Data"))
    b = pd.read_csv(BIN, skiprows=dat + 1, header=None, names=lines[hdr].strip().split(","))
    for c in ("SumPeriod", "SecPer", "OrbPer"):
        b[c] = pd.to_numeric(b[c], errors="coerce").where(lambda x: x > 0)
    b = b[b.Number > 0]
    return b.groupby("Number").agg(name=("Name", "first"), sumbin=("SumBin", "first"), bintype=("BinType", lambda s: "".join(sorted(set(s) - {"-"}))),
                                   primary=("SumPeriod", "first"), orbit=("OrbPer", "median"), secondary=("SecPer", "median")).reset_index()


def best_match(pt, row):
    best, err = "none", np.inf
    for kind in ("primary", "orbit", "secondary"):
        ref = row[kind]
        if not np.isfinite(ref):
            continue
        for q, lab in ((1, ""), (2, " x2"), (0.5, " x1/2")):
            e = abs(pt / (ref * q) - 1)
            if e < SCAN and e < err:
                best, err = kind + lab, e
    return best


def main():
    _, sc, _, _ = ex.cl_ev_scores()
    rep = pd.read_csv(f"{S}/all_sector_asteroid_report.csv", low_memory=False,
                      usecols=["designation", "period_hr", "is_reliable", "amp_snr", "n_cycles_observed", "n_phase_coverings"])
    rep = rep[rep.period_hr.notna()].drop_duplicates("designation")
    rep = rep.merge(sc[["designation", "cl", "cl_pass"]], on="designation", how="left")
    rep["cl_pass"] = rep.cl_pass.fillna(False).astype(bool)
    rep["dep_pass"] = rep.is_reliable.astype(str).str.lower().eq("true")
    rep["Number"] = pd.to_numeric(rep.designation.str.extract(r"^\((\d+)\)")[0], errors="coerce")
    d = binary_table().merge(rep, on="Number")
    d["match"] = [best_match(pt, r) for pt, (_, r) in zip(d.period_hr, d.iterrows())]
    d["kind"] = d.match.str.split().str[0]
    d.to_csv(OUT, index=False)
    print(f"LCDB binary-table objects {binary_table().shape[0]}, with a TESS period {len(d)}, with an orbital period {d.orbit.notna().sum()}")
    for lab, m in (("all", np.ones(len(d), bool)), ("deployed reliable", d.dep_pass), ("C + L + ev pass", d.cl_pass),
                   ("C + L + ev pass, orbit known", d.cl_pass & d.orbit.notna())):
        print(f"\n{lab} ({int(m.sum())}):\n" + d[m].match.value_counts().to_string())
    o = d[d.kind.eq("orbit") & (d.cl_pass | d.dep_pass)]
    print("\nreliable (either model) objects whose TESS period is the orbit:")
    print(o[["designation", "bintype", "primary", "orbit", "period_hr", "match", "amp_snr", "n_cycles_observed", "cl", "dep_pass", "cl_pass"]]
          .sort_values("orbit").round(3).to_string(index=False))
    n = d[d.kind.eq("none") & (d.cl_pass | d.dep_pass)]
    print("\nreliable (either model) objects matching none of the LCDB periods:")
    print(n[["designation", "bintype", "primary", "orbit", "secondary", "period_hr", "amp_snr", "n_cycles_observed", "cl", "dep_pass", "cl_pass"]]
          .sort_values("period_hr").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
