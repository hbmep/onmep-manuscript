#!/bin/bash
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --mem=700G
#SBATCH --cpus-per-task=180
#SBATCH --time=7-00:00:00

cd "$HOME/repos/hbmep-adaptive/notebooks/accuracy" || exit 1
mkdir -p logs

PY_SCRIPT="$1"
RESPONSE_LEN="$2"
PARAM="$3"

"$HOME/repos/hbmep-adaptive/.venv/bin/python" \
    "$PY_SCRIPT" \
    "$RESPONSE_LEN" \
    "$PARAM" \
