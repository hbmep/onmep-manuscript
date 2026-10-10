#!/bin/bash

PARAMS=(
    "a"
    "mu_int"
)

DRAW_LO=0
DRAW_HI=200
DRAW_STEP=1

DO_RUN=0
if [[ "${1:-}" == "--run" ]]; then
    DO_RUN=1
fi

USE_SLURM=0
if command -v sbatch >/dev/null 2>&1; then
  USE_SLURM=1
fi

for PARAM in "${PARAMS[@]}"; do
    for ((draw_lo=DRAW_LO; draw_lo<DRAW_HI; draw_lo+=DRAW_STEP)); do
        draw_hi=$((draw_lo + DRAW_STEP))

        if (( draw_hi > DRAW_HI )); then
            draw_hi="$DRAW_HI"
        fi

        JOB_NAME="${PARAM}_d${draw_lo}_${draw_hi}_s${DRAW_STEP}"

        COMMAND=(
            sbatch
            --reservation=vt2353_jbc28_gp_res
            --job-name="$JOB_NAME"
            job_samples.sh
            "$PARAM"
            "$draw_lo"
            "$draw_hi"
        )

        if [[ "$DO_RUN" -eq 1 ]]; then
            "${COMMAND[@]}"
        else
            printf '[DRY RUN] '
            printf '%q ' "${COMMAND[@]}"
            printf '\n'
        fi
    done
done
