"""One-time extraction of survey photometry from the MPC observation archive (MPCAT-OBS NumObs.txt.gz,
80-column format) into per-object parquet, for the TESS + sparse-survey shape inversion.
Run on ozstar via sbatch in /fred/oz335/TESSdata/mpc/mpc_obs.

Kept: CCD observations (column 15 'C') of numbered asteroids from the surveys whose photometry is useful
for inversion -- F51/F52 Pan-STARRS 1/2, I41 ZTF, G96 Mt. Lemmon, 703 Catalina -- and only magnitudes
reported to 0.01 mag. Older reports are rounded to 0.1 mag, too coarse against the ~0.06 mag scatter of
the precise ones. The 80-column format carries no per-point error; each (station, band) block gets its
own error floor in the fit. Every numbered object is kept (not only the current targets), so a larger
target list later does not need the 9 GB archive again.

Output (this directory):
  survey_phot/bucket_XXX.parquet   number, mjd_utc, ra, dec (deg), mag, band, stn; bucket = number % 256
Reading one object:
    pd.read_parquet(f"survey_phot/bucket_{n % 256:03d}.parquet", filters=[("number", "==", n)])

    python3 -u convert_mpc_obs_to_parquet.py NumObs.txt.gz [--max-lines N]
"""
import argparse
import os
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

STATIONS = {"F51", "F52", "I41", "G96", "703"}
OUT = "survey_phot"
NB = 256
FLUSH = 2_000_000
B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_B62 = {c: i for i, c in enumerate(B62)}
MJD0 = np.datetime64("1858-11-17")


def unpack_number(p):
    if p[0] == "~":
        n = 0
        for c in p[1:]:
            n = n * 62 + _B62[c]
        return n + 620000
    return int(p) if p[0].isdigit() else _B62[p[0]] * 10000 + int(p[1:])


def parse(lines):
    """Kept 80-column lines -> DataFrame. Columns (1-based): 1-5 packed number, 16-32 date, 33-44 RA,
    45-56 Dec, 66-70 mag, 71 band, 78-80 station."""
    a = np.array(lines)
    s = pd.Series(a)
    date = pd.to_datetime(s.str[15:25].str.replace(" ", "-"), format="%Y-%m-%d")
    mjd = (date.values - MJD0) / np.timedelta64(1, "D") + s.str[25:32].str.strip().replace("", "0").astype(float).values
    rh, rm, rs = (s.str[32:34].astype(float), s.str[35:37].astype(float), s.str[38:44].str.strip().astype(float))
    sign = np.where(s.str[44] == "-", -1.0, 1.0)
    dd, dm, ds = (s.str[45:47].astype(float), s.str[48:50].astype(float), s.str[51:56].str.strip().astype(float))
    return pd.DataFrame(dict(
        number=np.array([unpack_number(x) for x in s.str[0:5]], dtype=np.int64),
        mjd_utc=mjd, ra=15 * (rh + rm / 60 + rs / 3600).values, dec=sign * (dd + dm / 60 + ds / 3600).values,
        mag=s.str[65:70].astype(float).values.astype(np.float32), band=s.str[70].values, stn=s.str[77:80].values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--max-lines", type=int, default=0, help="stop after this many lines (timing tests)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    writers, keep, t0 = {}, [], time.time()
    stats = dict(lines=0, kept=0, coarse=0, not_ccd=0)

    def flush():
        if not keep:
            return
        df = parse(keep)
        keep.clear()
        for b, part in df.groupby(df.number % NB, sort=False):
            t = pa.Table.from_pandas(part, preserve_index=False)
            if b not in writers:
                writers[b] = pq.ParquetWriter(f"{a.out}/bucket_{b:03d}.parquet", t.schema, compression="zstd")
            writers[b].write_table(t)

    proc = subprocess.Popen(["gzip", "-dc", a.archive], stdout=subprocess.PIPE, bufsize=1 << 24)
    for raw in proc.stdout:
        stats["lines"] += 1
        if raw[77:80].decode() not in STATIONS:
            pass
        elif raw[14:15] != b"C":
            stats["not_ccd"] += 1
        elif not (raw[67:68] == b"." and raw[69:70].isdigit()):
            stats["coarse"] += 1
        elif raw[0:5].strip():                            # numbered (NumObs is all numbered; kept as a guard)
            keep.append(raw.decode())
            stats["kept"] += 1
            if len(keep) >= FLUSH:
                flush()
        if stats["lines"] % 50_000_000 == 0:
            print(f"  {stats['lines']:,} lines, {stats['kept']:,} kept, {(time.time() - t0) / 60:.1f} min", flush=True)
        if a.max_lines and stats["lines"] >= a.max_lines:
            proc.kill()
            break
    flush()
    for w in writers.values():
        w.close()
    print(f"done: {stats['lines']:,} lines; kept {stats['kept']:,} survey CCD rows at 0.01 mag; skipped "
          f"{stats['coarse']:,} rounded to 0.1 mag and {stats['not_ccd']:,} non-CCD survey rows; "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
