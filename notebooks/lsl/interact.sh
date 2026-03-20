#!/usr/bin/env bash
# usage:
#   ./interact.sh 0 99 16
#   ./interact.sh   # defaults: 0 99 16

start=${1:-0}
end=${2:-200}
nobs=${3:-72}

for d in $(seq "$start" "$end"); do
  echo "=== draw $d (nobs=$nobs) ==="
  python node_simulator.py "$d" "$nobs"
done
