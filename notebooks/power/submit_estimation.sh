#!/bin/bash

JOB="job_estimation.sh"
PY_SCRIPT="core_estimation.py"

RUN_IDS=(
    "uniform-mcmc"
    "pest-mle1"
    "eig-mcmc"
)

USE_FP=0
DRAW_STEP=1
DRAW_LO=0
DRAW_HI=2000

NUM_SUBJECTS="2,4,8,12,16"
N0=8
N=16
STEP=$((N0 + N))

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

        JOB_NAME="${RUN_ID}_d${draw_lo}_${draw_hi}_s${STEP}"

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
            "$NUM_SUBJECTS"
            "$STEP"
            "$USE_FP"
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
