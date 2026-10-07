#!/bin/bash
#SBATCH --job-name=power_scan_test
#SBATCH --output=/fred/oz335/rridden/asteroids_store_sector/logs/power_scan_test_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:45:00
#SBATCH --account=oz335
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /fred/oz335/rridden/asteroids
/home/rridden/miniconda3/bin/python3 power_scan_test.py
