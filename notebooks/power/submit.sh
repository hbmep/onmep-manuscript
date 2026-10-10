#!/bin/bash

JOB="job.sh"
PY_SCRIPT="core_job.py"

RUNS=(
    "uniform-mcmc 1"
    "pest-mle1 1"
    "eig-mcmc 1"
)

USE_FP=0
DRAW_STEP=50
DRAW_LO=0
DRAW_HI=2000

S_LO=0
S_HI=16
S_STEP=4

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

for RUN in "${RUNS[@]}"; do
    read -r RUN_ID K <<< "$RUN"

    for ((draw=DRAW_LO; draw < DRAW_HI; draw+=DRAW_STEP)); do
        draw_hi=$((draw + DRAW_STEP))

        for ((s=S_LO; s < S_HI; s+=S_STEP)); do
            s_hi=$((s + S_STEP))

            JOB_NAME="${RUN_ID}_K${K}_d${draw}_${draw_hi}_s${s}_${s_hi}"

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
                "$K"
                "$draw"
                "$draw_hi"
                "$s"
                "$s_hi"
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
done
