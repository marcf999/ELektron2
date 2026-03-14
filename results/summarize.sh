#!/bin/bash
# Summarize all .dat result files into a CSV
# Extracts: energy, detected, total, percentage

OUTFILE="results/summary.csv"
echo "file,energy_eV,detected,total,percent" > "$OUTFILE"

for f in results/*.dat; do
    [ -f "$f" ] || continue
    basename=$(basename "$f")

    energy=$(grep -m1 "^# startEnergy:" "$f" | sed 's/.*: *//; s/ *eV.*//')
    total=$(grep -m1 "^# Total simulations:" "$f" | sed 's/.*: *//')
    detected=$(grep -m1 "^# Detected:" "$f" | sed 's/.*: *//')

    if [ -z "$detected" ]; then
        # Older format: no detector, skip or mark N/A
        echo "$basename,$energy,N/A,$total,N/A" >> "$OUTFILE"
    else
        if [ "$total" -gt 0 ] 2>/dev/null; then
            pct=$(awk "BEGIN {printf \"%.4f\", 100.0 * $detected / $total}")
        else
            pct="0"
        fi
        echo "$basename,$energy,$detected,$total,$pct" >> "$OUTFILE"
    fi
done

echo "Written to $OUTFILE"
cat "$OUTFILE"
