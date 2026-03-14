#!/bin/bash
# Energy scan: run 1M electrons at each energy point and log detection rates.
# Usage: ./energy_scan.sh [electrons_per_point] [spin_axis]
#   Default: 750000 electrons per energy point, spin +z
#   spin_axis: +x, -x, +y, -y, +z, -z
#
# Output:
#   - Individual .dat files in ../results/ (one per energy)
#   - Summary CSV: energy_scan_TIMESTAMP.csv with energy, forward_exit, total, pct, time_s
#   - Console log: energy_scan_TIMESTAMP.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/../build/elektron2_rocm_fp64"
ELECTRONS="${1:-750000}"
SPIN="${2:-+z}"

# Energy scan range: 10 points centered on 5000 eV
E_START=4991
E_END=5009
E_STEP=2

RESULTS_DIR="${SCRIPT_DIR}/../../results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${RESULTS_DIR}/energy_scan_${TIMESTAMP}.csv"
LOGFILE="${RESULTS_DIR}/energy_scan_${TIMESTAMP}.log"

echo "Energy scan: ${E_START}–${E_END} eV, step ${E_STEP}, ${ELECTRONS} electrons/point"
echo "Summary: ${SUMMARY}"
echo "Log: ${LOGFILE}"
echo ""

# CSV header
echo "energy_eV,forward_exit,total,pct,kernel_time_s,throughput_e_per_s" > "$SUMMARY"

# Redirect all output to both console and log
exec > >(tee -a "$LOGFILE") 2>&1

TOTAL_POINTS=$(( (E_END - E_START) / E_STEP + 1 ))
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
    stdbuf -oL "$BINARY" "$ELECTRONS" "$ENERGY" "$SPIN" 2>&1 | tee "$TMPOUT"

    # Extract forward-exit count (isPos) and timing from stdout
    FORWARD=$(grep "^isNaN:" "$TMPOUT" | sed 's/.*isPos: \([0-9]*\).*/\1/' || echo "0")
    TOTAL_SIMS=$(grep "^TOTAL TIME FOR" "$TMPOUT" | sed 's/.*FOR \([0-9]*\) SIMULATIONS.*/\1/' || echo "0")
    KERNEL_MS=$(grep "^KERNEL TIME:" "$TMPOUT" | sed 's/.*: \([0-9]*\) ms.*/\1/' || echo "0")
    THROUGHPUT=$(grep "^KERNEL TIME:" "$TMPOUT" | sed 's/.*(// ; s/ electrons.*//' || echo "0")
    rm -f "$TMPOUT"

    if [ "$TOTAL_SIMS" -gt 0 ] 2>/dev/null; then
        PCT=$(echo "scale=4; 100 * ${FORWARD} / ${TOTAL_SIMS}" | bc)
    else
        PCT="0"
    fi
    KERNEL_S=$(echo "scale=1; ${KERNEL_MS:-0} / 1000" | bc)

    echo "${ENERGY},${FORWARD},${TOTAL_SIMS},${PCT},${KERNEL_S},${THROUGHPUT}" >> "$SUMMARY"

    ELAPSED=$(( $(date +%s) - SCAN_START ))
    REMAINING=$(( (TOTAL_POINTS - POINT) * ELAPSED / POINT ))
    echo ""
    echo ">>> ${ENERGY} eV: ${FORWARD}/${TOTAL_SIMS} forward exit (${PCT}%) in ${KERNEL_S}s"
    echo ">>> Scan progress: ${POINT}/${TOTAL_POINTS} | Elapsed: $((ELAPSED/3600))h$((ELAPSED%3600/60))m | ETA: $((REMAINING/3600))h$((REMAINING%3600/60))m"
    echo ""
done

echo "============================================================"
echo "SCAN COMPLETE"
echo "Results: ${SUMMARY}"
echo "============================================================"
cat "$SUMMARY"
