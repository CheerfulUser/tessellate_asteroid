#!/bin/bash
# After the pooled (run_store_periods.sh) and per-sector (run_store_sector_periods.sh) arrays: gather
# both into their reports, then derive the cross-sector agreement labels.
#SBATCH --job-name=gather_and_label
#SBATCH --output=/fred/oz335/rridden/asteroids_store_sector/logs/gather_and_label_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:05:00
# gather + labels measured 61 s (job 18173243)
#SBATCH --account=oz335
export PYTHONUNBUFFERED=1 ASTEROID_USE_STORE=1 ASTEROID_SAVE_PLOTS=0
PY=/home/rridden/miniconda3/bin/python3
cd /fred/oz335/rridden/asteroids
ASTEROID_OUTDIR_SUFFIX=_store $PY sector_report_pipeline.py --gather
ASTEROID_OUTDIR_SUFFIX=_store_sector ASTEROID_STORE_SPLIT=sector $PY sector_report_pipeline.py --gather
$PY agreement_labels.py
