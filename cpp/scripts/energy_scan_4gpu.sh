#!/bin/bash
# 4-GPU energy scan: 5000–5010 eV, dense near 5000, 2M electrons/point.
# Dispatches 4 energy points in parallel across GPUs 0-3, waits, repeats.
#
# Usage: ./energy_scan_4gpu.sh [electrons_per_point] [spin_axis]
#   Default: 1000000 electrons per energy point, spin +z
#   spin_axis: +x, -x, +y, -y, +z, -z
#
# Output:
#   - Individual .dat files in ../../results/
#   - Summary CSV: energy_scan_TIMESTAMP.csv
#   - Log: energy_scan_TIMESTAMP.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/../build/elektron2_rocm_fp64"
ELECTRONS="${1:-1000000}"
SPIN="${2:-+z}"
NUM_GPUS=$(rocminfo 2>/dev/null | grep -c "Name:.*gfx" || echo 4)
if [ "$NUM_GPUS" -lt 1 ]; then NUM_GPUS=4; fi

# Energy points: dense around 5005 anomaly (0.25 eV steps), sparser at edges (1.0 eV)
ENERGIES=(
    5000.0  5001.0  5002.0  5003.0
    5003.5  5004.0  5004.25 5004.5
    5004.75 5005.0  5005.25 5005.5
    5005.75 5006.0  5006.25 5006.5
    5007.0  5007.5  5008.0  5009.0
    5010.0
)

TOTAL_POINTS=${#ENERGIES[@]}

RESULTS_DIR="${SCRIPT_DIR}/../../results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${RESULTS_DIR}/energy_scan_${TIMESTAMP}.csv"
LOGFILE="${RESULTS_DIR}/energy_scan_${TIMESTAMP}.log"

echo "4-GPU energy scan: ${TOTAL_POINTS} points, ${ELECTRONS} electrons/point, spin ${SPIN}"
echo "GPUs: ${NUM_GPUS}"
echo "Summary: ${SUMMARY}"
echo "Log: ${LOGFILE}"
echo ""

echo "energy_eV,forward_exit,total,pct,kernel_time_s,throughput_e_per_s" > "$SUMMARY"

exec > >(tee -a "$LOGFILE") 2>&1

SCAN_START=$(date +%s)
COMPLETED=0
NEXT_IDX=0

# Per-GPU tracking: PID, temp file, energy
declare -A GPU_PID GPU_TMP GPU_ENERGY

# Launch a job on the given GPU from the work queue
launch_next() {
    local gpu=$1
    if [ $NEXT_IDX -ge $TOTAL_POINTS ]; then
        return 1
    fi
    local energy=${ENERGIES[$NEXT_IDX]}
    local tmpout=$(mktemp)
    NEXT_IDX=$((NEXT_IDX + 1))

    echo "[GPU $gpu] Launching ${ELECTRONS} electrons at ${energy} eV — $(date)"
    GPU=$gpu stdbuf -oL "$BINARY" "$ELECTRONS" "$energy" "$SPIN" > "$tmpout" 2>&1 &
    GPU_PID[$gpu]=$!
    GPU_TMP[$gpu]="$tmpout"
    GPU_ENERGY[$gpu]="$energy"
    return 0
}

# Harvest results from a finished GPU
harvest() {
    local gpu=$1
    local tmpout=${GPU_TMP[$gpu]}
    local energy=${GPU_ENERGY[$gpu]}
    COMPLETED=$((COMPLETED + 1))

    # Extract forward-exit count (isPos) and total from summary line
    FORWARD=$(grep "^isNaN:" "$tmpout" | sed 's/.*isPos: \([0-9]*\).*/\1/' || echo "0")
    TOTAL_SIMS=$(grep "^TOTAL TIME FOR" "$tmpout" | sed 's/.*FOR \([0-9]*\) SIMULATIONS.*/\1/' || echo "0")
    KERNEL_MS=$(grep "^KERNEL TIME:" "$tmpout" | sed 's/.*: \([0-9]*\) ms.*/\1/' || echo "0")
    THROUGHPUT=$(grep "^KERNEL TIME:" "$tmpout" | sed 's/.*(// ; s/ electrons.*//' || echo "0")
    rm -f "$tmpout"

    if [ "$TOTAL_SIMS" -gt 0 ] 2>/dev/null; then
        PCT=$(echo "scale=4; 100 * ${FORWARD} / ${TOTAL_SIMS}" | bc)
    else
        PCT="0"
    fi
    KERNEL_S=$(echo "scale=1; ${KERNEL_MS:-0} / 1000" | bc)
    echo "${energy},${FORWARD},${TOTAL_SIMS},${PCT},${KERNEL_S},${THROUGHPUT}" >> "$SUMMARY"

    ELAPSED=$(( $(date +%s) - SCAN_START ))
    REMAINING=$(( (TOTAL_POINTS - COMPLETED) * ELAPSED / COMPLETED ))
    echo ">>> ${energy} eV: ${FORWARD}/${TOTAL_SIMS} forward exit (${PCT}%) in ${KERNEL_S}s [${COMPLETED}/${TOTAL_POINTS}] ETA: $((REMAINING/3600))h$((REMAINING%3600/60))m"
}

# Seed all GPUs with initial work
for ((g=0; g<NUM_GPUS; g++)); do
    launch_next $g || true
done

# Work-queue loop: when any GPU finishes, harvest results and feed it the next point
while [ $COMPLETED -lt $TOTAL_POINTS ]; do
    # Poll for the first GPU that has finished
    for ((g=0; g<NUM_GPUS; g++)); do
        if [ -n "${GPU_PID[$g]:-}" ] && ! kill -0 "${GPU_PID[$g]}" 2>/dev/null; then
            wait "${GPU_PID[$g]}" || true
            harvest $g
            unset GPU_PID[$g]
            launch_next $g || true
        fi
    done
    sleep 1
done

# Sort CSV by energy (header first, then data sorted)
TMPCSV=$(mktemp)
head -1 "$SUMMARY" > "$TMPCSV"
tail -n +2 "$SUMMARY" | sort -t, -k1 -n >> "$TMPCSV"
mv "$TMPCSV" "$SUMMARY"

echo "============================================================"
echo "SCAN COMPLETE — ${TOTAL_POINTS} points, ${ELECTRONS} electrons each"
echo "Results: ${SUMMARY}"
echo "============================================================"
cat "$SUMMARY"
