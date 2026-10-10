#!/bin/bash

JOB="job_make_accuracy.sh"
PY_SCRIPT="make_accuracy.py"

PARAMS=(
    "a"
    "mu_int"
)

RESPONSES=(
    1
    3
)

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

for RESPONSE_LEN in "${RESPONSES[@]}"; do
    for PARAM in "${PARAMS[@]}"; do
        JOB_NAME="${RESPONSE_LEN}${PARAM}"

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
            "$RESPONSE_LEN"
            "$PARAM"
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
