#!/bin/bash

JOB="job_trajectory.sh"
PY_SCRIPT="core_trajectory.py"

RUN_IDS=(
    "a"
    "h"
    "b"
)

DRAW_LO=0
DRAW_HI=200
DRAW_STEP=1

NLO=3
NHI=17

DO_RUN=0
if [[ "${1:-}" == "--run" ]]; then
    DO_RUN=1
fi

USE_SLURM=0
if command -v sbatch >/dev/null 2>&1; then
    USE_SLURM=1
fi

if [[ "$USE_SLURM" -eq 1 && -z "${SLURM_RESERVATION_NAME:-}" ]]; then
    echo "Error: SLURM_RESERVATION_NAME is not set." >&2
    exit 1
fi

mkdir -p logs

for RUN_ID in "${RUN_IDS[@]}"; do
    for ((draw_lo=DRAW_LO; draw_lo<DRAW_HI; draw_lo+=DRAW_STEP)); do
        draw_hi=$((draw_lo + DRAW_STEP))

        if (( draw_hi > DRAW_HI )); then
            draw_hi="$DRAW_HI"
        fi

        JOB_NAME="${RUN_ID}_d${draw_lo}_${draw_hi}_n${NLO}_${NHI}"

        if [[ "$USE_SLURM" -eq 1 ]]; then
            COMMAND=(
                sbatch
                "--reservation=$SLURM_RESERVATION_NAME"
                --job-name="$JOB_NAME"
            )
        else
            COMMAND=(bash)
        fi

        COMMAND+=(
            "$JOB"
            "$PY_SCRIPT"
            "$RUN_ID"
            "$draw_lo"
            "$draw_hi"
            "$NLO"
            "$NHI"
        )

        if [[ "$DO_RUN" -eq 1 ]]; then
            "${COMMAND[@]}"
        else
            printf '[DRYRUN] '
            printf '%q ' "${COMMAND[@]}"
            printf '\n'
        fi

    done
done
