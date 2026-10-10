#!/bin/bash
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --mem=700G
#SBATCH --cpus-per-task=180
#SBATCH --time=7-00:00:00

cd "$HOME/repos/hbmep-adaptive/notebooks/figure_validation" || exit 1
mkdir -p logs

PY_SCRIPT="$1"

"$HOME/repos/hbmep-adaptive/.venv/bin/python" \
    "$PY_SCRIPT" \
