#!/bin/bash
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00

cd "$HOME/repos/hbmep-adaptive/notebooks/eig_decomposition" || exit 1
mkdir -p logs

PY_SCRIPT="$1"
RUN_ID="$2"
LO="$3"
HI="$4"
NLO="$5"
NHI="$6"

"$HOME/repos/hbmep-adaptive/.venv/bin/python" \
    "$PY_SCRIPT" \
    "$RUN_ID" \
    "$LO" \
    "$HI" \
    "$NLO" \
    "$NHI" \
