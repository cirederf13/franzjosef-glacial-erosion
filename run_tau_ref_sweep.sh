#!/usr/bin/env bash
# tau_ref calibration against the 2013 velocities: 5 values x 3 replicates -> outputs/tau_ref_sweep/. Analyse with analyze_tau_ref.py.
set -u
cd "$(dirname "$(readlink -f "$0")")"
eval "$(conda shell.bash hook)"
conda activate igm-erosion
BASE=outputs/tau_ref_sweep
mkdir -p "$BASE"; D="$BASE/driver.log"
echo "tau_ref sweep started $(date)" | tee -a "$D"
for TAU in 0.10 0.13 0.17 0.2153697218796329 0.28; do
  for REP in 1 2 3; do
    O="$BASE/tau${TAU}_rep${REP}"; mkdir -p "$O"
    echo "=== tau=$TAU rep=$REP start $(date) ===" | tee -a "$D"
    igm_run +experiment=inversion_2013_uniform_tau \
      processes.iceflow.physics.sliding.tau_ref=$TAU \
      hydra.run.dir="$O" > "$O/run.log" 2>&1
    EC=$?
    OK=1; [ $EC -eq 0 ] || OK=0; [ -f "$O/optimize.nc" ] || OK=0
    grep -qi "nan" "$O/run.log" && OK=0
    [ $OK -eq 1 ] && echo "tau=$TAU rep=$REP SUCCESS $(date)" | tee -a "$D" \
                  || echo "tau=$TAU rep=$REP FAILED exit=$EC $(date)" | tee -a "$D"
  done
done
echo "tau_ref sweep finished $(date)" | tee -a "$D"
