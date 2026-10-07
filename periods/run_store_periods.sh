#!/bin/bash
# Period determination over the asteroid store: one array task per designation range (see
# sector_report_pipeline.py, store mode), then gather with
#   ASTEROID_USE_STORE=1 ASTEROID_OUTDIR_SUFFIX=_store python3 sector_report_pipeline.py --gather
#SBATCH --job-name=store_periods
#SBATCH --output=/fred/oz335/rridden/asteroids_store/logs/periods_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:10:00
#SBATCH --account=oz335
# tasks measured on the full Sectors 27-55 run: pooled median 69 s, slowest 187 s; per-sector median 33 s, slowest 71 s (snr5); raw test tasks 128-160 s pooled, 44-53 s per sector
export ASTEROID_USE_STORE=1
export ASTEROID_OUTDIR_SUFFIX=${ASTEROID_OUTDIR_SUFFIX:-_store}
export ASTEROID_STORE_NTASKS=${ASTEROID_STORE_NTASKS:-1000}
export ASTEROID_SAVE_PLOTS=0
export ASTEROID_POINT_MODE=${ASTEROID_POINT_MODE:-raw}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /fred/oz335/rridden/asteroids
/home/rridden/miniconda3/bin/python3 sector_report_pipeline.py
