#!/usr/bin/env bash
# Runs every experiment and the aggregation. Usage:
#   N_JOBS=16 bash run_all.sh                       # nine core datasets, output in rerun/
#   N_JOBS=16 DATASETS=all OUT=rerun_all bash run_all.sh   # all 19 tasks (the paper runs
#                                                   # only e01 to e03 on them)
#   QUICK=1 bash run_all.sh                         # smoke test on small splits
# The stored records of the paper are in results/ and results_all/; write new runs
# to a different folder, since jobs whose output already exists are skipped.
set -euo pipefail
N_JOBS=${N_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)}
DATASETS=${DATASETS:-main}
OUT=${OUT:-rerun}
EXTRA=""; E04_EXTRA=""; E10_EXTRA=""; QUICK_RUN=0
if [[ "${QUICK:-0}" == "1" ]]; then
  QUICK_RUN=1
  EXTRA="--quick"; DATASETS=${DATASETS_QUICK:-cpu_act,sulfur}; OUT=${OUT}_quick
  E04_EXTRA="--trials_lgbm 3 --trials_xgb 3 --trials_sklearn 2"; E10_EXTRA="--reps 200"
fi
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C="--datasets $DATASETS --n_jobs $N_JOBS --out $OUT $EXTRA"

python -m experiments.e10_coverage_check --out "$OUT" $E10_EXTRA   # Monte Carlo coverage check of WLACP
python -m experiments.e01_ranking   $C                          # ranking, rejection curves, baselines, accuracy
python -m experiments.e02_conformal $C                          # conformal comparison without refitting
python -m experiments.e05_shift     $C                          # shift magnitude and direction
python -m experiments.e06_calib_size $C                         # calibration-size sweep, 5 seeds
python -m experiments.e03_extended  $C                          # NGBoost, CV+, CQR, three libraries
python -m experiments.e07_noise --n_jobs "$N_JOBS" --out "$OUT" $EXTRA   # noise structure
python -m experiments.e08_ring  --out "$OUT"                    # ring benchmark
python -m experiments.e04_sensitivity $C $E04_EXTRA             # partition sensitivity and tuned models
if [[ "$QUICK_RUN" == "0" ]]; then
  python -m experiments.e09_timing --out "$OUT"                 # timing, best run on an idle machine
fi

python -m analysis.a01_tables --results "$OUT"
python -m analysis.a02_stats  --results "$OUT"
