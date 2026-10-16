#!/usr/bin/env bash
# Runs every experiment and the aggregation. Usage:
#   N_JOBS=16 DATASETS=main bash run_all.sh        # nine core datasets of the paper
#   N_JOBS=16 DATASETS=all  bash run_all.sh        # all 19 regression tasks
#   QUICK=1 bash run_all.sh                         # smoke test on small splits
set -euo pipefail
N_JOBS=${N_JOBS:-$(nproc)}
DATASETS=${DATASETS:-main}
OUT=${OUT:-results}
EXTRA=""
if [[ "${QUICK:-0}" == "1" ]]; then
  EXTRA="--quick"; DATASETS=${DATASETS_QUICK:-cpu_act,sulfur}; OUT=${OUT}_quick
fi
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C="--datasets $DATASETS --n_jobs $N_JOBS --out $OUT $EXTRA"

python -m experiments.e10_coverage_check --out "$OUT"          # validity check of level-adjusted WLACP
python -m experiments.e01_ranking   $C                          # ranking, rejection curves, baselines, accuracy
python -m experiments.e02_conformal $C                          # conformal comparison without refitting
python -m experiments.e05_shift     $C                          # shift magnitude and direction
python -m experiments.e06_calib_size $C                         # calibration-size sweep, 5 seeds
python -m experiments.e03_extended  $C                          # NGBoost, CV+, CQR, three libraries
python -m experiments.e07_noise --n_jobs "$N_JOBS" --out "$OUT" $EXTRA   # noise structure
python -m experiments.e08_ring  --out "$OUT"                    # ring benchmark
python -m experiments.e04_sensitivity $C --with_conformal       # partition sensitivity and tuned models
python -m experiments.e09_timing --out "$OUT"                   # timing, best run on an idle machine

python -m analysis.a01_tables --results "$OUT"
python -m analysis.a02_stats  --results "$OUT"
