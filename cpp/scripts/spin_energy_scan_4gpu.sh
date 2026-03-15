#!/bin/bash
# 4-GPU spin × energy scan.
# Runs 4 spin orientations (+x, +y, +z, random) at 3 energies (4995, 5000, 5005 eV).
# 12 jobs total, dispatched across 4 GPUs via work queue.
#
# Usage: ./spin_energy_scan_4gpu.sh [electrons_per_point]
#   Default: 1000000 electrons per point
#
# Output:
#   - Individual .dat files in ../../results/
#   - Summary CSV: spin_energy_scan_TIMESTAMP.csv
#   - Log: spin_energy_scan_TIMESTAMP.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/../build/elektron2_rocm_fp64"
ELECTRONS="${1:-1000000}"
NUM_GPUS=$(rocminfo 2>/dev/null | grep -c "Name:.*gfx" || echo 4)
if [ "$NUM_GPUS" -lt 1 ]; then NUM_GPUS=4; fi

# 4 spins × 3 energies = 12 jobs
SPINS=("+x" "+y" "+z" "random")
ENERGIES=(4995 5000 5005)

# Build flat work queue: (energy, spin) pairs
declare -a WORK_ENERGY WORK_SPIN
for spin in "${SPINS[@]}"; do
    for energy in "${ENERGIES[@]}"; do
        WORK_ENERGY+=("$energy")
        WORK_SPIN+=("$spin")
    done
done
TOTAL_POINTS=${#WORK_ENERGY[@]}

RESULTS_DIR="${SCRIPT_DIR}/../../results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${RESULTS_DIR}/spin_energy_scan_${TIMESTAMP}.csv"
LOGFILE="${RESULTS_DIR}/spin_energy_scan_${TIMESTAMP}.log"

echo "Spin × Energy scan: ${#SPINS[@]} spins × ${#ENERGIES[@]} energies = ${TOTAL_POINTS} jobs"
echo "Spins: ${SPINS[*]}"
echo "Energies: ${ENERGIES[*]} eV"
echo "Electrons/point: ${ELECTRONS}"
echo "GPUs: ${NUM_GPUS}"
echo "Summary: ${SUMMARY}"
echo "Log: ${LOGFILE}"
echo ""

echo "spin,energy_eV,forward_exit,total,pct,kernel_time_s,throughput_e_per_s" > "$SUMMARY"

exec > >(tee -a "$LOGFILE") 2>&1

SCAN_START=$(date +%s)
COMPLETED=0
NEXT_IDX=0

declare -A GPU_PID GPU_TMP GPU_ENERGY GPU_SPIN

launch_next() {
    local gpu=$1
    if [ $NEXT_IDX -ge $TOTAL_POINTS ]; then
        return 1
    fi
    local energy=${WORK_ENERGY[$NEXT_IDX]}
    local spin=${WORK_SPIN[$NEXT_IDX]}
    local tmpout=$(mktemp)
    NEXT_IDX=$((NEXT_IDX + 1))

    echo "[GPU $gpu] Launching ${ELECTRONS} electrons at ${energy} eV, spin ${spin} — $(date)"
    GPU=$gpu stdbuf -oL "$BINARY" "$ELECTRONS" "$energy" "$spin" > "$tmpout" 2>&1 &
    GPU_PID[$gpu]=$!
    GPU_TMP[$gpu]="$tmpout"
    GPU_ENERGY[$gpu]="$energy"
    GPU_SPIN[$gpu]="$spin"
    return 0
}

harvest() {
    local gpu=$1
    local tmpout=${GPU_TMP[$gpu]}
    local energy=${GPU_ENERGY[$gpu]}
    local spin=${GPU_SPIN[$gpu]}
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
    echo "${spin},${energy},${FORWARD},${TOTAL_SIMS},${PCT},${KERNEL_S},${THROUGHPUT}" >> "$SUMMARY"

    ELAPSED=$(( $(date +%s) - SCAN_START ))
    REMAINING=$(( (TOTAL_POINTS - COMPLETED) * ELAPSED / COMPLETED ))
    echo ">>> spin=${spin} ${energy} eV: ${FORWARD}/${TOTAL_SIMS} forward exit (${PCT}%) in ${KERNEL_S}s [${COMPLETED}/${TOTAL_POINTS}] ETA: $((REMAINING/3600))h$((REMAINING%3600/60))m"
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

# Sort CSV by spin then energy (header first, then data sorted)
TMPCSV=$(mktemp)
head -1 "$SUMMARY" > "$TMPCSV"
tail -n +2 "$SUMMARY" | sort -t, -k1,1 -k2,2n >> "$TMPCSV"
mv "$TMPCSV" "$SUMMARY"

echo "============================================================"
echo "SCAN COMPLETE — ${TOTAL_POINTS} jobs (${#SPINS[@]} spins × ${#ENERGIES[@]} energies), ${ELECTRONS} electrons each"
echo "Results: ${SUMMARY}"
echo "============================================================"
cat "$SUMMARY"
