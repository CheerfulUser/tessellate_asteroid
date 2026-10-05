"""One-time conversion of Gaia DR3 sso_observation (ESA CDN bulk files, sso_observation_raw/) into
per-TRANSIT parquet partitioned by asteroid number, for the TESS + ATLAS + Gaia shape inversion.
Run on ozstar via sbatch in /fred/oz335/TESSdata/mpc/gaia.

Each raw row is one CCD; the G photometry is per transit (identical across the 8-9 CCDs of a
transit, all within ~40 s), so transits are collapsed to one point: mean epoch, RA/Dec and Gaia
position, the transit's G mag / flux / error, and n_ccd. Raw files are read in chunks; the last
transit of each chunk is carried into the next so no transit is split.

Output (this directory):
  gaia_dr3_sso/bucket_XXX.parquet   transits, bucket = number_mp % 256
  gaia_dr3_sso_objects.parquet      number_mp -> bucket, n_transits, n_ccd
Checks: sum of n_ccd == raw CCD rows read; no transit split across writes; transits whose CCDs
disagree in G magnitude are counted.
Reading one object:
    pd.read_parquet(f"gaia_dr3_sso/bucket_{n % 256:03d}.parquet", filters=[("number_mp", "==", n)])
"""
import glob
import os
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW = "sso_observation_raw"
OUT = "gaia_dr3_sso"
NB = 256
CHUNK = 2_000_000
COLS = ["number_mp", "denomination", "transit_id", "epoch_utc", "ra", "dec", "g_mag", "g_flux",
        "g_flux_error", "x_gaia", "y_gaia", "z_gaia"]
AGG = dict(number_mp=("number_mp", "first"), denomination=("denomination", "first"),
           epoch_utc=("epoch_utc", "mean"), ra=("ra", "mean"), dec=("dec", "mean"),
           g_mag=("g_mag", "first"), g_flux=("g_flux", "first"), g_flux_error=("g_flux_error", "first"),
           g_mag_nunique=("g_mag", "nunique"), x_gaia=("x_gaia", "mean"), y_gaia=("y_gaia", "mean"),
           z_gaia=("z_gaia", "mean"), n_ccd=("transit_id", "size"))


def main():
    os.makedirs(OUT, exist_ok=True)
    writers, stats, t0 = {}, dict(raw=0, transits=0, multi_mag=0), time.time()

    def write(tr):
        tr = tr.reset_index()
        stats["multi_mag"] += int((tr.g_mag_nunique > 1).sum())
        tr = tr.drop(columns="g_mag_nunique")
        tr["number_mp"] = tr.number_mp.astype("int64")
        for b, part in tr.groupby(tr.number_mp % NB, sort=False):
            t = pa.Table.from_pandas(part, preserve_index=False)
            if b not in writers:
                writers[b] = pq.ParquetWriter(f"{OUT}/bucket_{b:03d}.parquet", t.schema, compression="zstd")
            writers[b].write_table(t)
        stats["transits"] += len(tr)

    carry = None
    for f in sorted(glob.glob(f"{RAW}/SsoObservation_*.csv.gz")):
        for ch in pd.read_csv(f, comment="#", usecols=COLS, chunksize=CHUNK,
                              dtype={"denomination": str, "transit_id": "int64"}):
            ch = ch[ch.number_mp.notna()]
            stats["raw"] += len(ch)
            if carry is not None:
                ch = pd.concat([carry, ch], ignore_index=True)
            last = ch.transit_id.iloc[-1]
            carry = ch[ch.transit_id == last]
            ch = ch[ch.transit_id != last]
            write(ch.groupby("transit_id", sort=False).agg(**AGG))
        print(f"{os.path.basename(f)}: {stats['raw']:,} CCD rows, {stats['transits']:,} transits, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if carry is not None and len(carry):
        write(carry.groupby("transit_id", sort=False).agg(**AGG))
    for w in writers.values():
        w.close()

    objs = []
    for b in range(NB):
        p = f"{OUT}/bucket_{b:03d}.parquet"
        if os.path.exists(p):
            d = pd.read_parquet(p, columns=["number_mp", "n_ccd", "transit_id"])
            g = d.groupby("number_mp").agg(n_transits=("transit_id", "nunique"), n_ccd=("n_ccd", "sum"),
                                           n_rows=("transit_id", "size"))
            g["bucket"] = b
            objs.append(g.reset_index())
    objs = pd.concat(objs, ignore_index=True)
    objs.to_parquet("gaia_dr3_sso_objects.parquet", index=False)
    split = int((objs.n_rows != objs.n_transits).sum())
    print(f"\nCCD rows read {stats['raw']:,}; sum n_ccd {int(objs.n_ccd.sum()):,}; transits {stats['transits']:,}; "
          f"objects {len(objs):,}; transits whose CCDs disagree in G {stats['multi_mag']}; "
          f"objects with a split transit {split}", flush=True)
    print("check (165) Loreley:", objs[objs.number_mp == 165].to_dict("records"),
          "(archive query: 60 transits, 496 CCD rows)", flush=True)
    ok = stats["raw"] == int(objs.n_ccd.sum()) and split == 0
    print("validation passed" if ok else "VALIDATION FAILED", flush=True)


if __name__ == "__main__":
    main()
