#!/bin/bash

JOB="job_make_accuracy.sh"
PY_SCRIPT="make_accuracy.py"

JOB_NAME="make_accuracy"

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
)

if [[ "$DO_RUN" -eq 1 ]]; then
    "${COMMAND[@]}"
else
    printf '[DRYRUN] '
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
fi
