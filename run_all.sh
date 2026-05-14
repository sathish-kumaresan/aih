#!/usr/bin/env bash
# Run the model tiers end to end, logging each to results/logs/.
# Intended for an unattended tmux session:
#   tmux new -s aih
#   ./run_all.sh         # runs tiers 1 2 3
#   ./run_all.sh 2 3     # runs only the given tiers
set -u
cd "$(dirname "$0")"

TIERS=("$@")
[ ${#TIERS[@]} -eq 0 ] && TIERS=(1 2 3)

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
declare -A STATUS

run_tier () {
    local tier=$1
    local log="${LOG_DIR}/tier${tier}_${STAMP}.log"
    echo "=== tier ${tier} starting $(date -Is) -> ${log} ==="
    python train.py --tier "$tier" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        STATUS[$tier]="OK"
    else
        STATUS[$tier]="FAILED (exit ${rc})"
    fi
    echo "=== tier ${tier} ${STATUS[$tier]} $(date -Is) ==="
}

for t in "${TIERS[@]}"; do
    run_tier "$t"
done

echo
echo "=== summary (${STAMP}) ==="
for t in "${TIERS[@]}"; do
    echo "  tier ${t}: ${STATUS[$t]}"
done
