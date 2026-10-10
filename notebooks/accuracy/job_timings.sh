#!/bin/bash
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00

cd "$HOME/repos/hbmep-adaptive/notebooks/accuracy" || exit 1
mkdir -p logs

PY_SCRIPT="$1"
RUN_ID="$2"
LO="$3"
HI="$4"
RESPONSE_LEN="$5"

"$HOME/repos/hbmep-adaptive/.venv/bin/python" \
    "$PY_SCRIPT" \
    "$RUN_ID" \
    "$LO" \
    "$HI" \
    "$RESPONSE_LEN"
