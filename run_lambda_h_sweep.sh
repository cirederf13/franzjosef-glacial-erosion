#!/usr/bin/env bash
# Thickness-regularisation L-curve: lambda_h 1e4..1e8, 3 replicates each -> outputs/lambda_h_sweep/. Analyse with analyze_lambda_h.py.
set -u
cd "$(dirname "$(readlink -f "$0")")"
eval "$(conda shell.bash hook)"
conda activate igm-erosion
BASE=outputs/lambda_h_sweep
mkdir -p "$BASE"; D="$BASE/driver.log"
echo "lambda_h sweep started $(date)" | tee -a "$D"
for LAM in 1e4 1e5 1e6 1e7 1e8; do
  for REP in 1 2 3; do
    O="$BASE/lam${LAM}_rep${REP}"; mkdir -p "$O"
    echo "=== lam=$LAM rep=$REP start $(date) ===" | tee -a "$D"
    igm_run +experiment=inversion_2013_uniform_tau \
      assimilations.field_inversion.objective.regularization.0.lam=$LAM \
      hydra.run.dir="$O" > "$O/run.log" 2>&1
    EC=$?
    OK=1; [ $EC -eq 0 ] || OK=0; [ -f "$O/optimize.nc" ] || OK=0
    grep -qi "nan" "$O/run.log" && OK=0
    [ $OK -eq 1 ] && echo "lam=$LAM rep=$REP SUCCESS $(date)" | tee -a "$D" \
                  || echo "lam=$LAM rep=$REP FAILED exit=$EC $(date)" | tee -a "$D"
  done
done
echo "lambda_h sweep finished $(date)" | tee -a "$D"
