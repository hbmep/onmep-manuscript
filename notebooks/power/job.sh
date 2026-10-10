#!/bin/bash
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00

cd "$HOME/repos/hbmep-adaptive/notebooks/power" || exit 1
mkdir -p logs

PY_SCRIPT="$1"
RUN_ID="$2"
K="$3"
DRAW_LO="$4"
DRAW_HI="$5"
S_LO="$6"
S_HI="$7"
USE_FP="$8"

"$HOME/repos/hbmep-adaptive/.venv/bin/python" "$PY_SCRIPT" \
    "$RUN_ID" \
    "$K" \
    "$DRAW_LO" \
    "$DRAW_HI" \
    "$S_LO" \
    "$S_HI" \
    "$USE_FP"
