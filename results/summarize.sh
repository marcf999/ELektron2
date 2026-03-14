#!/bin/bash
# Summarize all .dat result files into a CSV
# Extracts: energy, forward exits (data lines), total simulations, percentage

OUTFILE="results/summary.csv"
echo "file,energy_eV,forward_exit,total,percent" > "$OUTFILE"

for f in results/*.dat; do
    [ -f "$f" ] || continue
    basename=$(basename "$f")

    energy=$(grep -m1 "^# startEnergy:" "$f" | sed 's/.*: *//; s/ *eV.*//')
    total=$(grep -m1 "^# Total simulations:" "$f" | sed 's/.*: *//')
    forward_exit=$(grep -vc "^#" "$f" || echo "0")

    if [ "$total" -gt 0 ] 2>/dev/null; then
        pct=$(awk "BEGIN {printf \"%.4f\", 100.0 * $forward_exit / $total}")
    else
        pct="0"
    fi
    echo "$basename,$energy,$forward_exit,$total,$pct" >> "$OUTFILE"
done

echo "Written to $OUTFILE"
cat "$OUTFILE"
