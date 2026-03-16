#!/bin/bash
# Energy scan: two-row channeling, spin +y (4995–5005 eV, 1 eV steps)
# then no-zitter control at 4995, 5000, 5005 eV.
# Uses 4-GPU work queue.
#
# Usage: ./scan_tworows_spiny_and_nozitter_4gpu.sh [electrons_per_point]
#   Default: 1000000 electrons per energy point

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/../build/elektron2_rocm_fp64"
ELECTRONS="${1:-1000000}"
NUM_GPUS=4  # MI300X has 4 usable VFs (rocminfo reports phantom devices)

# Phase 1: spin +y, 4995–5005 eV in 1 eV steps
SPINY_ENERGIES=(4995 4996 4997 4998 4999 5000 5001 5002 5003 5004 5005)
# Phase 2: no-zitter control at 3 points
NOZITTER_ENERGIES=(4995 5000 5005)

RESULTS_DIR="${SCRIPT_DIR}/../../results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${RESULTS_DIR}/scan_spiny_nozitter_${TIMESTAMP}.csv"
LOGFILE="${RESULTS_DIR}/scan_spiny_nozitter_${TIMESTAMP}.log"

# Build work queue: (energy, spin, extra_flag, label)
declare -a WORK_ENERGY WORK_SPIN WORK_EXTRA WORK_LABEL
for e in "${SPINY_ENERGIES[@]}"; do
    WORK_ENERGY+=("$e")
    WORK_SPIN+=("+y")
    WORK_EXTRA+=("")
    WORK_LABEL+=("spin+y")
done
for e in "${NOZITTER_ENERGIES[@]}"; do
    WORK_ENERGY+=("$e")
    WORK_SPIN+=("+z")
    WORK_EXTRA+=("--no-zitter")
    WORK_LABEL+=("no-zitter")
done

TOTAL_POINTS=${#WORK_ENERGY[@]}

echo "============================================================"
echo "SCAN: spin +y (4995–5005, 1eV steps) + no-zitter control"
echo "${TOTAL_POINTS} points, ${ELECTRONS} electrons/point"
echo "GPUs: ${NUM_GPUS}"
echo "Summary: ${SUMMARY}"
echo "============================================================"
echo ""

echo "label,energy_eV,forward_exit,total,pct,kernel_time_s,throughput_e_per_s" > "$SUMMARY"

exec > >(tee -a "$LOGFILE") 2>&1

SCAN_START=$(date +%s)
COMPLETED=0
NEXT_IDX=0

declare -A GPU_PID GPU_TMP GPU_ENERGY GPU_LABEL

launch_next() {
    local gpu=$1
    if [ $NEXT_IDX -ge $TOTAL_POINTS ]; then
        return 1
    fi
    local energy=${WORK_ENERGY[$NEXT_IDX]}
    local spin=${WORK_SPIN[$NEXT_IDX]}
    local extra=${WORK_EXTRA[$NEXT_IDX]}
    local label=${WORK_LABEL[$NEXT_IDX]}
    local tmpout=$(mktemp)
    NEXT_IDX=$((NEXT_IDX + 1))

    echo "[GPU $gpu] Launching ${ELECTRONS} electrons at ${energy} eV (${label}) — $(date)"
    GPU=$gpu stdbuf -oL "$BINARY" "$ELECTRONS" "$energy" "$spin" $extra > "$tmpout" 2>&1 &
    GPU_PID[$gpu]=$!
    GPU_TMP[$gpu]="$tmpout"
    GPU_ENERGY[$gpu]="$energy"
    GPU_LABEL[$gpu]="$label"
    return 0
}

harvest() {
    local gpu=$1
    local tmpout=${GPU_TMP[$gpu]}
    local energy=${GPU_ENERGY[$gpu]}
    local label=${GPU_LABEL[$gpu]}
    COMPLETED=$((COMPLETED + 1))

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
    echo "${label},${energy},${FORWARD},${TOTAL_SIMS},${PCT},${KERNEL_S},${THROUGHPUT}" >> "$SUMMARY"

    ELAPSED=$(( $(date +%s) - SCAN_START ))
    REMAINING=$(( (TOTAL_POINTS - COMPLETED) * ELAPSED / COMPLETED ))
    echo ">>> [${label}] ${energy} eV: ${FORWARD}/${TOTAL_SIMS} forward exit (${PCT}%) in ${KERNEL_S}s [${COMPLETED}/${TOTAL_POINTS}] ETA: $((REMAINING/3600))h$((REMAINING%3600/60))m"
}

# Seed all GPUs
for ((g=0; g<NUM_GPUS; g++)); do
    launch_next $g || true
done

# Work-queue loop
while [ $COMPLETED -lt $TOTAL_POINTS ]; do
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

# Sort CSV by label then energy (header first)
TMPCSV=$(mktemp)
head -1 "$SUMMARY" > "$TMPCSV"
tail -n +2 "$SUMMARY" | sort -t, -k1,1 -k2,2n >> "$TMPCSV"
mv "$TMPCSV" "$SUMMARY"

echo ""
echo "============================================================"
echo "SCAN COMPLETE — ${TOTAL_POINTS} points, ${ELECTRONS} electrons each"
echo "Results: ${SUMMARY}"
echo "============================================================"
cat "$SUMMARY"
