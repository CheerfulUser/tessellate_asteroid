#!/bin/bash
# Random good/bad sample of the raw-frame pooled report, plotted for checking by eye (render_label_sample.py).
#SBATCH --job-name=render_label_sample
#SBATCH --output=/fred/oz335/rridden/asteroids_store/logs/render_label_sample_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:05:00
#SBATCH --account=oz335
export ASTEROID_USE_STORE=1 ASTEROID_OUTDIR_SUFFIX=_store ASTEROID_SAVE_PLOTS=1 ASTEROID_WRITE_CLEAN_LC=0
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
cd /fred/oz335/rridden/asteroids
/home/rridden/miniconda3/bin/python3 render_label_sample.py "$@"
