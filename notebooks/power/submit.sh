#!/bin/bash
RUN_ID="eig-mcmc"

# FP
DRAW_STEP=1
DRAW_LO=0
DRAW_HI=2000
PY_SCRIPT="core_job.py"

# # TP
# DRAW_STEP=1
# DRAW_LO=0
# DRAW_HI=1000
# PY_SCRIPT="core_job.py"

P_LO=0
P_HI=16
P_STEP=8

DO_RUN=0
if [[ "${1:-}" == "--run" ]]; then
  DO_RUN=1
fi

for ((draw=DRAW_LO; draw < DRAW_HI; draw+=DRAW_STEP)); do
  draw_hi=$((draw + DRAW_STEP))

  for ((p=P_LO; p < P_HI; p+=P_STEP)); do
    p_hi=$((p + P_STEP))

    if [[ "$DO_RUN" -eq 1 ]]; then
      sbatch \
        --reservation=vt2353_jbc28_gp_res \
        --job-name="${RUN_ID}_d${draw}_${draw_hi}_p${p}_${p_hi}" \
        job.sh "$PY_SCRIPT" "$RUN_ID" "$draw" "$draw_hi" "$p" "$p_hi"
    else
      echo "[DRYRUN submit] sbatch \
      --job-name=${RUN_ID}_d${draw}_${draw_hi}_p${p}_${p_hi} \
      job.sh $PY_SCRIPT $RUN_ID $draw $draw_hi $p $p_hi"
    fi

  done
done
