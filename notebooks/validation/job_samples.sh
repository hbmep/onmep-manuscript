#!/bin/bash
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --reservation=vt2353_jbc28_gp_res
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00

cd "$HOME/repos/hbmep-adaptive/notebooks/figure_validation" || exit 1
mkdir -p logs

PY_SCRIPT="core_samples.py"

PARAM="$1"
LO="$2"
HI="$3"

"$HOME/repos/hbmep-adaptive/.venv/bin/python" \
    "$PY_SCRIPT" \
    "$PARAM" \
    "$LO" \
    "$HI" \
