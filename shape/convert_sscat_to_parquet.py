"""One-time conversion of the ATLAS Solar System Catalog V3 (Denneau et al.; atlas-sscat.v3.0.dat.bz2,
259M rows, 2016 - mid-2025) into parquet partitioned by object, so any asteroid's ATLAS photometry
is a single small read and the catalogue never has to be re-streamed as the TESSELLATE asteroid list
grows. Run on ozstar via sbatch in /fred/oz335/TESSdata/mpc/atlas.

Output (same directory):
  sscat_v3/bucket_XXX.parquet  rows bucketed by crc32(packed designation) % 256
  sscat_v3_objects.parquet     packed designation -> bucket, n_rows (the lookup table)
Validated against atlas-sscat.v3.0.index: total rows and every object's row count must match.

Reading one object:
    b = sscat_bucket(packed)                       # zlib.crc32(packed.encode()) % 256
    pd.read_parquet(f"sscat_v3/bucket_{b:03d}.parquet", filters=[("kast", "==", packed)])
"""
import os
import subprocess
import time
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "atlas-sscat.v3.0.dat.bz2"
INDEX = "atlas-sscat.v3.0.index"
OUT = "sscat_v3"
N_BUCKETS = 256
CHUNK = 5_000_000
COLS = ["MJD_obs", "MJD_lc", "H", "dm", "filt", "m", "V", "m1AU", "delta", "R", "SOE", "phi0", "phi1",
        "x", "y", "obs", "kast", "ra", "dec", "dx", "dy", "tol"]
DTYPES = {"MJD_obs": "float64", "MJD_lc": "float64", "H": "float32", "dm": "float32", "filt": "str",
          "m": "float32", "V": "float32", "m1AU": "float32", "delta": "float32", "R": "float32",
          "SOE": "float32", "phi0": "float32", "phi1": "float32", "x": "int32", "y": "int32",
          "obs": "str", "kast": "str", "ra": "float64", "dec": "float64", "dx": "float32",
          "dy": "float32", "tol": "float32"}


def sscat_bucket(packed):
    return zlib.crc32(packed.encode()) % N_BUCKETS


def main():
    os.makedirs(OUT, exist_ok=True)
    proc = subprocess.Popen(["bzcat", SRC], stdout=subprocess.PIPE, bufsize=1 << 24)
    reader = pd.read_csv(proc.stdout, sep=r"\s+", header=None, names=COLS, dtype=DTYPES,
                         comment="#", chunksize=CHUNK, engine="c")
    writers, counts, n_total, t0 = {}, {}, 0, time.time()
    bucket_of = {}
    for i, chunk in enumerate(reader):
        keys = chunk["kast"].values
        uniq = pd.unique(keys)
        for k in uniq:
            if k not in bucket_of:
                bucket_of[k] = sscat_bucket(k)
        b = np.fromiter((bucket_of[k] for k in keys), dtype=np.int32, count=len(keys))
        vc = chunk["kast"].value_counts()
        for k, n in vc.items():
            counts[k] = counts.get(k, 0) + int(n)
        for bb, part in chunk.groupby(b, sort=False):
            table = pa.Table.from_pandas(part, preserve_index=False)
            if bb not in writers:
                writers[bb] = pq.ParquetWriter(f"{OUT}/bucket_{bb:03d}.parquet", table.schema,
                                               compression="zstd")
            writers[bb].write_table(table)
        n_total += len(chunk)
        print(f"chunk {i}: {n_total:,} rows, {len(counts):,} objects, {(time.time() - t0) / 60:.1f} min",
              flush=True)
    for w in writers.values():
        w.close()
    if proc.wait() != 0:
        raise SystemExit(f"bzcat exited {proc.returncode}")

    objs = pd.DataFrame({"kast": list(counts), "n_rows": list(counts.values())})
    objs["bucket"] = objs.kast.map(sscat_bucket).astype("int16")
    objs.to_parquet("sscat_v3_objects.parquet", index=False)

    idx = pd.read_csv(INDEX, sep=r"\s+", header=None, names=["kast", "start", "end", "n"], dtype={"kast": str})
    idx = idx[idx.kast != "kast"]   # the header line's own entry
    # The catalogue has one row the index does not list (kast Z9999, MJD 60254.249); kept, not an error
    idx = pd.concat([idx, pd.DataFrame({"kast": ["Z9999"], "n": [1]})], ignore_index=True)
    m = idx.merge(objs, on="kast", how="outer", indicator=True)
    bad = m[(m._merge != "both") | (m.n != m.n_rows)]
    print(f"\nrows written {n_total:,}; index total {int(idx.n.sum()):,}; objects {len(objs):,} vs index {len(idx):,}; "
          f"mismatched objects {len(bad)}", flush=True)
    if len(bad):
        print(bad.head(20).to_string(index=False))
        raise SystemExit("validation FAILED")
    print("validation passed", flush=True)


if __name__ == "__main__":
    main()
