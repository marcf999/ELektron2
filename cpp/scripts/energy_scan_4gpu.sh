#!/bin/bash
# 4-GPU energy scan: 5000–5010 eV, dense near 5000, 2M electrons/point.
# Dispatches 4 energy points in parallel across GPUs 0-3, waits, repeats.
#
# Usage: ./energy_scan_4gpu.sh [electrons_per_point]
#   Default: 2000000 electrons per energy point
#
# Output:
#   - Individual .dat files in ../../results/
#   - Summary CSV: energy_scan_TIMESTAMP.csv
#   - Log: energy_scan_TIMESTAMP.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/../build/elektron2_rocm_fp64"
ELECTRONS="${1:-1000000}"
NUM_GPUS=4

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

echo "4-GPU energy scan: ${TOTAL_POINTS} points, ${ELECTRONS} electrons/point"
echo "GPUs: ${NUM_GPUS}"
echo "Summary: ${SUMMARY}"
echo "Log: ${LOGFILE}"
echo ""

echo "energy_eV,detected,total,pct,kernel_time_s,throughput_e_per_s" > "$SUMMARY"

exec > >(tee -a "$LOGFILE") 2>&1

SCAN_START=$(date +%s)
POINT=0

# Process in batches of NUM_GPUS
for ((i=0; i<TOTAL_POINTS; i+=NUM_GPUS)); do
    PIDS=()
    TMPS=()
    BATCH_ENERGIES=()

    # Launch up to NUM_GPUS jobs in parallel
    for ((g=0; g<NUM_GPUS; g++)); do
        IDX=$((i + g))
        if [ $IDX -ge $TOTAL_POINTS ]; then
            break
        fi
        ENERGY=${ENERGIES[$IDX]}
        BATCH_ENERGIES+=("$ENERGY")
        TMPOUT=$(mktemp)
        TMPS+=("$TMPOUT")

        echo "[GPU $g] Launching ${ELECTRONS} electrons at ${ENERGY} eV — $(date)"
        GPU=$g stdbuf -oL "$BINARY" "$ELECTRONS" "$ENERGY" > "$TMPOUT" 2>&1 &
        PIDS+=($!)
    done

    # Wait for all GPUs in this batch
    for ((j=0; j<${#PIDS[@]}; j++)); do
        wait ${PIDS[$j]}
        POINT=$((POINT + 1))
        ENERGY=${BATCH_ENERGIES[$j]}
        TMPOUT=${TMPS[$j]}

        DETECTED=$(grep "^DETECTED:" "$TMPOUT" | sed 's/.*: \([0-9]*\)\/.*/\1/' || echo "0")
        TOTAL=$(grep "^DETECTED:" "$TMPOUT" | sed 's/.*\/\([0-9]*\) .*/\1/' || echo "0")
        PCT=$(grep "^DETECTED:" "$TMPOUT" | sed 's/.*(\(.*\)%)/\1/' || echo "0")
        KERNEL_MS=$(grep "^KERNEL TIME:" "$TMPOUT" | sed 's/.*: \([0-9]*\) ms.*/\1/' || echo "0")
        THROUGHPUT=$(grep "^KERNEL TIME:" "$TMPOUT" | sed 's/.*(// ; s/ electrons.*//' || echo "0")
        rm -f "$TMPOUT"

        KERNEL_S=$(echo "scale=1; ${KERNEL_MS:-0} / 1000" | bc)
        echo "${ENERGY},${DETECTED},${TOTAL},${PCT},${KERNEL_S},${THROUGHPUT}" >> "$SUMMARY"

        ELAPSED=$(( $(date +%s) - SCAN_START ))
        REMAINING=$(( (TOTAL_POINTS - POINT) * ELAPSED / POINT ))
        echo ">>> ${ENERGY} eV: ${DETECTED}/${TOTAL} detected (${PCT}%) in ${KERNEL_S}s [${POINT}/${TOTAL_POINTS}] ETA: $((REMAINING/3600))h$((REMAINING%3600/60))m"
    done
    echo ""
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
