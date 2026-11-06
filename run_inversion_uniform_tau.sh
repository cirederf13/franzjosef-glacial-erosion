#!/usr/bin/env bash
# 2013 thickness inversion with uniform tau_ref (experiment/inversion_2013_uniform_tau.yaml) -> outputs/inversion_2013_uniform_tau/.
set -u
cd "$(dirname "$(readlink -f "$0")")"
eval "$(conda shell.bash hook)"
conda activate igm-erosion

OUT=outputs/inversion_2013_uniform_tau
DRIVER=$OUT/driver.log
mkdir -p "$OUT"
echo "uniform-tau inversion started $(date)" | tee -a "$DRIVER"

igm_run +experiment=inversion_2013_uniform_tau \
        hydra.run.dir="$OUT" > "$OUT/run.log" 2>&1
EC=$?

OK=1
[ $EC -eq 0 ] || OK=0
[ -f "$OUT/optimize.nc" ] || OK=0
grep -qi "nan" "$OUT/run.log" && OK=0

if [ $OK -eq 1 ]; then
  echo "SUCCESS exit=$EC $(date)" | tee -a "$DRIVER"
else
  echo "FAILED exit=$EC optimize.nc=$([ -f "$OUT/optimize.nc" ] && echo yes || echo no) $(date)" | tee -a "$DRIVER"
fi
echo "uniform-tau inversion finished $(date)" | tee -a "$DRIVER"
