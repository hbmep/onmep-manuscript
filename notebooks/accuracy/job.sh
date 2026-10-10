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
DRAW_LO="$3"
DRAW_HI="$4"

"$HOME/repos/hbmep-adaptive/.venv/bin/python" "$PY_SCRIPT" \
"$RUN_ID" "$DRAW_LO" "$DRAW_HI"
