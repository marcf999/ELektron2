#!/bin/bash
# Fine energy scan around 5005 eV feature.
# Usage: ./energy_scan_fine.sh [electrons_per_point]
#   Default: 750000 electrons per energy point
#
# Output:
#   - Individual .dat files in ../results/ (one per energy)
#   - Summary CSV: energy_scan_TIMESTAMP.csv with energy, detected, total, pct, time_s
#   - Console log: energy_scan_TIMESTAMP.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/../build/elektron2_rocm_fp64"
ELECTRONS="${1:-750000}"

# Fine scan: 5003–5007 eV in 0.5 eV steps (9 points)
E_START=5003
E_END=5007
E_STEP=0.5

RESULTS_DIR="${SCRIPT_DIR}/../../results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${RESULTS_DIR}/energy_scan_${TIMESTAMP}.csv"
LOGFILE="${RESULTS_DIR}/energy_scan_${TIMESTAMP}.log"

echo "Fine energy scan: ${E_START}–${E_END} eV, step ${E_STEP}, ${ELECTRONS} electrons/point"
echo "Summary: ${SUMMARY}"
echo "Log: ${LOGFILE}"
echo ""

# CSV header
echo "energy_eV,detected,total,pct,kernel_time_s,throughput_e_per_s" > "$SUMMARY"

# Redirect all output to both console and log
exec > >(tee -a "$LOGFILE") 2>&1

TOTAL_POINTS=$(echo "($E_END - $E_START) / $E_STEP + 1" | bc)
POINT=0
SCAN_START=$(date +%s)

for ENERGY in $(seq $E_START $E_STEP $E_END); do
    POINT=$((POINT + 1))
    echo "============================================================"
    echo "[$POINT/$TOTAL_POINTS] Running ${ELECTRONS} electrons at ${ENERGY} eV"
    echo "Started: $(date)"
    echo "============================================================"

    # Run simulation with live output (tee to temp file for post-parsing)
    TMPOUT=$(mktemp)
    stdbuf -oL "$BINARY" "$ELECTRONS" "$ENERGY" 2>&1 | tee "$TMPOUT"

    # Extract results from captured output
    DETECTED=$(grep "^DETECTED:" "$TMPOUT" | sed 's/.*: \([0-9]*\)\/.*/\1/')
    TOTAL=$(grep "^DETECTED:" "$TMPOUT" | sed 's/.*\/\([0-9]*\) .*/\1/')
    PCT=$(grep "^DETECTED:" "$TMPOUT" | sed 's/.*(\(.*\)%)/\1/')
    KERNEL_MS=$(grep "^KERNEL TIME:" "$TMPOUT" | sed 's/.*: \([0-9]*\) ms.*/\1/')
    THROUGHPUT=$(grep "^KERNEL TIME:" "$TMPOUT" | sed 's/.*(// ; s/ electrons.*//')
    rm -f "$TMPOUT"

    KERNEL_S=$(echo "scale=1; ${KERNEL_MS:-0} / 1000" | bc)

    echo "${ENERGY},${DETECTED},${TOTAL},${PCT},${KERNEL_S},${THROUGHPUT}" >> "$SUMMARY"

    ELAPSED=$(( $(date +%s) - SCAN_START ))
    REMAINING=$(( (TOTAL_POINTS - POINT) * ELAPSED / POINT ))
    echo ""
    echo ">>> ${ENERGY} eV: ${DETECTED}/${TOTAL} detected (${PCT}%) in ${KERNEL_S}s"
    echo ">>> Scan progress: ${POINT}/${TOTAL_POINTS} | Elapsed: $((ELAPSED/3600))h$((ELAPSED%3600/60))m | ETA: $((REMAINING/3600))h$((REMAINING%3600/60))m"
    echo ""
done

echo "============================================================"
echo "SCAN COMPLETE"
echo "Results: ${SUMMARY}"
echo "============================================================"
cat "$SUMMARY"
