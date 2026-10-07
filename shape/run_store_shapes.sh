#!/bin/bash
# TESS-only fixed-pole shape inversion (batch_shapes.py) for the objects the raw-frame store period run
# rates reliable, from that run's cleaned lightcurves. Targets: asteroids_store/shapes/targets.csv.
#SBATCH --job-name=store_shapes
#SBATCH --output=/fred/oz335/rridden/asteroids_store/shapes/logs/shape_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:30:00
# test task (18207466, 20 objects): 9-28 s per fit, one mesh failure; 1000 tasks x ~40 objects ~12 min
#SBATCH --account=oz335
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DAMIT_CONVEX=/fred/oz335/rridden/code/DAMIT-convex
S=/fred/oz335/rridden/asteroids_store
cd /fred/oz335/rridden/asteroids
/home/rridden/miniconda3/bin/python3 shape/batch_shapes.py \
  --targets $S/shapes/${SHAPE_TARGETS:-targets.csv} \
  --lcdir $S/clean_lightcurves --lcsuffix _clean_lc.csv \
  --index ${SLURM_ARRAY_TASK_ID} --total ${SHAPE_TOTAL:-$SLURM_ARRAY_TASK_COUNT} \
  --out $S/shapes/${SHAPE_OUT:-out} --work $S/shapes/work_${SHAPE_OUT:-out}_${SLURM_ARRAY_TASK_ID}
