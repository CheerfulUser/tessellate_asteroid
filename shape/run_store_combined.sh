#!/bin/bash
# Joint TESS + ATLAS + Gaia + MPC-survey inversion (combined_tess_atlas.py, iterative, equal weights) for the
# reliable objects of the raw-frame store period run. Each array task fits every SHAPE_TOTAL-th target in
# turn on its allocated cores. Targets: asteroids_store/shapes/${SHAPE_TARGETS:-targets.csv}.
#SBATCH --job-name=store_combined
#SBATCH --output=/fred/oz335/rridden/asteroids_store/shapes/logs/combined_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --account=oz335
# sizing: set from a test task before the full array (local 16 workers: Loreley 2.6 min, Elektra 9 min)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DAMIT_CONVEX=/fred/oz335/rridden/code/DAMIT-convex
S=/fred/oz335/rridden/asteroids_store
export LCDIR=$S/clean_lightcurves
OUT=$S/shapes/${SHAPE_OUT:-combined}
mkdir -p $OUT
cd /fred/oz335/rridden/asteroids
/home/rridden/miniconda3/bin/python3 - "$S/shapes/${SHAPE_TARGETS:-targets.csv}" ${SLURM_ARRAY_TASK_ID} ${SHAPE_TOTAL:-$SLURM_ARRAY_TASK_COUNT} <<'PY' |
import sys, pandas as pd
t = pd.read_csv(sys.argv[1]).iloc[int(sys.argv[2])::int(sys.argv[3])]
for d, p in zip(t.designation, t.period_hr):
    print(f"{d}|{p}")
PY
while IFS='|' read -r des per; do
  tag=$(echo "$des" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+|_+$//g')
  [ -s "$OUT/$tag.json" ] && continue
  t0=$(date +%s)
  /home/rridden/miniconda3/bin/python3 -u shape/combined_tess_atlas.py "$des" "$per" --mode iterative --weights equal --out $OUT < /dev/null > $OUT/$tag.log 2>&1
  echo "$des exit $? $(( $(date +%s) - t0 )) s"
  # keep what the page builder needs (build_tess_atlas_variant.py --prefix final / mirror): shape, params,
  # model lightcurve and uncertainties of the pole-free and mirror fits; the rest of the work dir goes
  mkdir -p "$OUT/fits/$tag"
  mv "$OUT/work_$tag"/{final,mirror}_{s,p,f,err}.txt "$OUT/fits/$tag/" 2>/dev/null
  rm -rf "$OUT/work_$tag"
done
