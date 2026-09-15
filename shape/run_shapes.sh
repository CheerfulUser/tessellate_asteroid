#!/bin/bash
#SBATCH --job-name=shapes
#SBATCH --output=/fred/oz335/rridden/asteroids/shapes/logs/shape_%A_%a.out
#SBATCH --error=/fred/oz335/rridden/asteroids/shapes/logs/shape_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=oz335
export PYTHONUNBUFFERED=1
cd /fred/oz335/rridden/asteroids
mkdir -p shapes/logs
/home/rridden/miniconda3/bin/python3 shape/batch_shapes.py \
  --targets shapes/targets.csv \
  --lcdir clean_lightcurves --lcsuffix _clean_lc.csv \
  --index ${SLURM_ARRAY_TASK_ID} --total ${SLURM_ARRAY_TASK_COUNT} \
  --out shapes/out --work shapes/work_${SLURM_ARRAY_TASK_ID}
