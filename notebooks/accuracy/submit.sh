#!/bin/bash
RUN_ID="eig-mcmc"

DRAW_STEP=10
DRAW_LO=0
DRAW_HI=1000
PY_SCRIPT="core_job.py"

DO_RUN=0
if [[ "${1:-}" == "--run" ]]; then
  DO_RUN=1
fi

for ((draw=DRAW_LO; draw < DRAW_HI; draw+=DRAW_STEP)); do
  draw_hi=$((draw + DRAW_STEP))

  if [[ "$DO_RUN" -eq 1 ]]; then
    sbatch \
      --reservation=vt2353_jbc28_gp_res \
      --job-name="${RUN_ID}_d${draw}_${draw_hi}" \
      job.sh "$PY_SCRIPT" "$RUN_ID" "$draw" "$draw_hi"
  else
    echo "[DRYRUN submit] sbatch \
    --job-name=${RUN_ID}_d${draw}_${draw_hi} \
    job.sh $PY_SCRIPT $RUN_ID $draw $draw_hi"
  fi

done
