#!/bin/bash
# Run spin -y at 5000 eV, 2M electrons split across 4 GPUs (500k each),
# then merge the 4 .dat files into one combined file.
#
# Usage: ./run_spin_minus_y_4gpu.sh
#   Runs from the remote server (ssh hotaisle@23.183.40.68)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/../build/elektron2_rocm_fp64"
RESULTS_DIR="${SCRIPT_DIR}/../../results"
mkdir -p "$RESULTS_DIR"

ENERGY=5000
SPIN="-y"
TOTAL=2000000
NUM_GPUS=4
PER_GPU=$((TOTAL / NUM_GPUS))   # 500000 each

echo "============================================================"
echo "Spin -y run: ${TOTAL} electrons (${PER_GPU} × ${NUM_GPUS} GPUs)"
echo "Energy: ${ENERGY} eV | Spin: ${SPIN}"
echo "Binary: ${BINARY}"
echo "============================================================"
echo ""

# Launch one job per GPU
declare -A PIDS TMPFILES
for ((g=0; g<NUM_GPUS; g++)); do
    tmpout=$(mktemp)
    TMPFILES[$g]="$tmpout"
    echo "[GPU $g] Launching ${PER_GPU} electrons — $(date)"
    GPU=$g stdbuf -oL "$BINARY" "$PER_GPU" "$ENERGY" "$SPIN" > "$tmpout" 2>&1 &
    PIDS[$g]=$!
done

echo ""
echo "All 4 GPUs running. Waiting for completion..."
echo ""

# Wait for all to finish
FAILED=0
for ((g=0; g<NUM_GPUS; g++)); do
    if wait "${PIDS[$g]}"; then
        echo "[GPU $g] Done — $(date)"
    else
        echo "[GPU $g] FAILED — $(date)"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "ERROR: $FAILED GPU(s) failed. Aborting merge."
    exit 1
fi

# Find the 4 .dat files just produced (most recent spin-y files)
echo ""
echo "Collecting partial .dat files..."
PART_FILES=()
for ((g=0; g<NUM_GPUS; g++)); do
    # Extract the output filename from the program's stdout
    DATFILE=$(grep "^Wrote.*to " "${TMPFILES[$g]}" | sed 's/.*to //')
    if [ -z "$DATFILE" ] || [ ! -f "$DATFILE" ]; then
        # Fallback: find most recent spin-y file in results
        DATFILE=$(ls -t "${RESULTS_DIR}"/*spin-y*"${PER_GPU}.dat" 2>/dev/null | head -1)
    fi
    echo "  GPU $g: $DATFILE"
    PART_FILES+=("$DATFILE")
    rm -f "${TMPFILES[$g]}"
done

# Merge: take header from first file, concatenate data lines, re-index
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
MERGED="${RESULTS_DIR}/${TIMESTAMP}_rocm-dp853_${ENERGY}eV_spin${SPIN}_${TOTAL}.dat"

echo ""
echo "Merging into: ${MERGED}"

{
    # Header from first file, but fix "Total simulations" to reflect merged total
    # and fix Summary line by summing counts
    TOTAL_ISNAN=0; TOTAL_ISPOS=0; TOTAL_ISNEG=0; TOTAL_XYESC=0
    for f in "${PART_FILES[@]}"; do
        vals=$(grep "^# Summary:" "$f" | sed 's/.*isNaN=// ; s/ isPos=/ / ; s/ isNeg=/ / ; s/ xyEscape=/ /')
        read nan pos neg esc <<< "$vals"
        TOTAL_ISNAN=$((TOTAL_ISNAN + nan))
        TOTAL_ISPOS=$((TOTAL_ISPOS + pos))
        TOTAL_ISNEG=$((TOTAL_ISNEG + neg))
        TOTAL_XYESC=$((TOTAL_XYESC + esc))
    done

    # Write header from first partial, patching totals
    while IFS= read -r line; do
        if [[ "$line" == "# Total simulations:"* ]]; then
            echo "# Total simulations: ${TOTAL}"
        elif [[ "$line" == "# Summary:"* ]]; then
            echo "# Summary: isNaN=${TOTAL_ISNAN} isPos=${TOTAL_ISPOS} isNeg=${TOTAL_ISNEG} xyEscape=${TOTAL_XYESC}"
        elif [[ "$line" == "# Kernel time:"* ]]; then
            # Sum kernel times
            TOTAL_KT=0
            for f in "${PART_FILES[@]}"; do
                kt=$(grep "^# Kernel time:" "$f" | sed 's/[^0-9]//g')
                TOTAL_KT=$((TOTAL_KT + kt))
            done
            AVG_KT=$((TOTAL_KT / NUM_GPUS))
            echo "# Kernel time: ${AVG_KT} ms (avg of ${NUM_GPUS} GPUs)"
        elif [[ "$line" == "# Date:"* ]]; then
            echo "# Date: $(date '+%Y-%m-%d %H:%M:%S')"
        elif [[ "$line" == "#"* ]]; then
            echo "$line"
        else
            break
        fi
    done < "${PART_FILES[0]}"

    # Concatenate all data lines, re-index from 0
    IDX=0
    for f in "${PART_FILES[@]}"; do
        while IFS= read -r line; do
            [[ "$line" == "#"* ]] && continue
            [ -z "$line" ] && continue
            # Replace the first field (old index) with new sequential index
            echo "$line" | sed "s/^[0-9]*/${IDX}/"
            IDX=$((IDX + 1))
        done < "$f"
    done

} > "$MERGED"

# Count data lines in merged file
DATA_LINES=$(grep -cv '^#' "$MERGED" | tr -d ' ')
echo ""
echo "============================================================"
echo "DONE — Merged ${DATA_LINES} detected electrons from ${NUM_GPUS} GPUs"
echo "Total simulated: ${TOTAL}"
echo "Detection rate: $(echo "scale=4; 100 * ${DATA_LINES} / ${TOTAL}" | bc)%"
echo "Output: ${MERGED}"
echo "============================================================"
