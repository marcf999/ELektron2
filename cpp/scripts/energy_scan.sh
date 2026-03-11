#!/bin/bash
# Energy scan: run 1M electrons at each energy point and log detection rates.
# Usage: ./energy_scan.sh [electrons_per_point]
#   Default: 1000000 electrons per energy point
#
# Output:
#   - Individual .dat files in ../results/ (one per energy)
#   - Summary CSV: energy_scan_TIMESTAMP.csv with energy, detected, total, pct, time_s
#   - Console log: energy_scan_TIMESTAMP.log

set -euo pipefail

BINARY="./elektron2_rocm_fp64"
ELECTRONS="${1:-1000000}"

# Energy scan range
E_START=4980
E_END=5020
E_STEP=2

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="energy_scan_${TIMESTAMP}.csv"
LOGFILE="energy_scan_${TIMESTAMP}.log"

echo "Energy scan: ${E_START}–${E_END} eV, step ${E_STEP}, ${ELECTRONS} electrons/point"
echo "Summary: ${SUMMARY}"
echo "Log: ${LOGFILE}"
echo ""

# CSV header
echo "energy_eV,detected,total,pct,kernel_time_s,throughput_e_per_s" > "$SUMMARY"

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

    # Run simulation and capture output
    OUTPUT=$("$BINARY" "$ELECTRONS" "$ENERGY" 2>&1)
    echo "$OUTPUT"

    # Extract results from output
    DETECTED=$(echo "$OUTPUT" | grep "^DETECTED:" | sed 's/.*: \([0-9]*\)\/.*/\1/')
    TOTAL=$(echo "$OUTPUT" | grep "^DETECTED:" | sed 's/.*\/\([0-9]*\) .*/\1/')
    PCT=$(echo "$OUTPUT" | grep "^DETECTED:" | sed 's/.*(\(.*\)%)/\1/')
    KERNEL_MS=$(echo "$OUTPUT" | grep "^KERNEL TIME:" | sed 's/.*: \([0-9]*\) ms.*/\1/')
    THROUGHPUT=$(echo "$OUTPUT" | grep "^KERNEL TIME:" | sed 's/.*(// ; s/ electrons.*//')

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
