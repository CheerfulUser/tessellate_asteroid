#!/bin/bash
# Strong agreement labels on the raw-frame per-sector report, then retrain the reliability classifier on it.
#SBATCH --job-name=retrain_raw
#SBATCH --output=/fred/oz335/rridden/asteroids_store/logs/retrain_raw_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --account=oz335
# not yet measured: strong_labels is one pass over ~0.9M report rows, training fits ~12 small models on ~25k rows
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PY=/home/rridden/miniconda3/bin/python3
cd /fred/oz335/rridden/asteroids
$PY /fred/oz335/rridden/asteroids_store_sector/strong_labels.py && $PY train_raw_ozstar.py
