#!/usr/bin/env bash
# Full ~100 ka glacial cycle at native 78 m (experiment/glacial_cycle_rescaled_lgm_v6_native.yaml); about 30 h on one GPU.
# Runs are not bit-reproducible and can NaN early, so up to 3 attempts are made; resume a late failure with make_restart.py.
# Usage: ./run_glacial_cycle.sh [OUTDIR]   (default outputs/glacial_cycle_final)
set -u
cd "$(dirname "$(readlink -f "$0")")"
eval "$(conda shell.bash hook)"
conda activate igm-erosion
BASE=${1:-outputs/glacial_cycle_final}
mkdir -p "$BASE"
for ATT in 1 2 3; do
  O="$BASE/attempt${ATT}"; LOG="$BASE/attempt${ATT}.log"; mkdir -p "$O"
  echo "=== attempt $ATT start $(date) ===" | tee -a "$BASE/driver.log"
  igm_run +experiment=glacial_cycle_rescaled_lgm_v6_native hydra.run.dir="$O" > "$LOG" 2>&1
  EC=$?
  OK=1
  [ $EC -eq 0 ] || OK=0
  [ -f "$O/output.nc" ] || OK=0
  grep -qi "nan" "$LOG" && OK=0
  if [ $OK -eq 1 ]; then
    echo "attempt $ATT SUCCESS: $O/output.nc" | tee -a "$BASE/driver.log"; exit 0
  fi
  echo "attempt $ATT FAILED (exit=$EC)" | tee -a "$BASE/driver.log"
done
exit 1
