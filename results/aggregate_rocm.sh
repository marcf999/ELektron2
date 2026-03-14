#!/bin/bash

CSVOUT="results/rocm_summary.csv"
LOGOUT="results/rocm_summary.log"

# Pass 1: extract per-file data, aggregate in awk
# Forward exits = data lines (non-comment lines) in each .dat file
for f in results/*rocm*.dat; do
    energy=$(grep -m1 "^# startEnergy:" "$f" | sed 's/.*: *//; s/ *eV.*//')
    total=$(grep -m1 "^# Total simulations:" "$f" | sed 's/.*: *//')
    forward_exit=$(grep -vc "^#" "$f" || echo "0")
    [ -z "$total" ] && continue
    echo "$energy $forward_exit $total $(basename $f)"
done | awk '
BEGIN {
    print "energy_eV,forward_exit,total,percent,stderr_pct,files" > "'"$CSVOUT"'"
}
{
    e=$1; d=$2; n=$3; fn=$4
    det[e] += d
    tot[e] += n
    nfiles[e]++
    files[e] = files[e] " " fn
    if (!(e in energies)) {
        energies[e] = 1
        eList[++eCount] = e
    }
}
END {
    # Sort energies
    for (i=1; i<=eCount; i++)
        for (j=i+1; j<=eCount; j++)
            if (eList[i]+0 > eList[j]+0) { tmp=eList[i]; eList[i]=eList[j]; eList[j]=tmp }

    grand_d = 0; grand_n = 0

    # CSV output
    for (i=1; i<=eCount; i++) {
        e = eList[i]
        p = det[e] / tot[e]
        se = sqrt(p * (1-p) / tot[e])
        printf "%s,%d,%d,%.4f,%.4f,%d\n", e, det[e], tot[e], 100*p, 100*se, nfiles[e] >> "'"$CSVOUT"'"
        # store for log
        pct[e] = 100*p
        err[e] = 100*se
        grand_d += det[e]
        grand_n += tot[e]
    }

    # Log output
    LOG = "'"$LOGOUT"'"
    printf "%s\n", "================================================================================" > LOG
    printf "ELektron2 ROCm Energy Scan - Aggregated Summary\n" >> LOG
    printf "Generated: 2026-03-13\n" >> LOG
    printf "%s\n\n", "================================================================================" >> LOG
    printf "%10s %10s %10s %10s %10s %10s %6s\n", "Energy", "FwdExit", "Total", "Rate %", "+/-1s %", "+/-2s %", "Files" >> LOG
    printf "%s\n", "--------------------------------------------------------------------" >> LOG

    peak_e = ""; peak_p = 0; trough_e = ""; trough_p = 100
    for (i=1; i<=eCount; i++) {
        e = eList[i]
        printf "%10s %10d %10d %10.4f %10.4f %10.4f %6d\n", e, det[e], tot[e], pct[e], err[e], 2*err[e], nfiles[e] >> LOG
        if (pct[e] > peak_p) { peak_p = pct[e]; peak_e = e; peak_se = err[e] }
        if (pct[e] < trough_p) { trough_p = pct[e]; trough_e = e; trough_se = err[e] }
    }
    printf "%s\n\n", "--------------------------------------------------------------------" >> LOG

    overall_p = grand_d / grand_n
    overall_se = sqrt(overall_p * (1-overall_p) / grand_n)
    printf "Overall: %d/%d = %.4f%% +/- %.4f%% (1s)\n\n", grand_d, grand_n, 100*overall_p, 100*overall_se >> LOG

    printf "Peak forward exit: %s eV at %.4f%% +/- %.4f%%\n", peak_e, peak_p, peak_se >> LOG
    printf "Min  forward exit: %s eV at %.4f%% +/- %.4f%%\n\n", trough_e, trough_p, trough_se >> LOG

    # Peak vs trough significance
    z = (peak_p - trough_p) / sqrt(peak_se^2 + trough_se^2)
    printf "Peak vs min: z = %.2f sigma", z >> LOG
    if (z > 3) printf " (>3s, statistically significant)\n" >> LOG
    else if (z > 2) printf " (>2s, suggestive)\n" >> LOG
    else printf " (<2s, not significant)\n" >> LOG

    # Chi-squared
    printf "\nChi-squared test (H0: uniform forward-exit rate across all energies):\n" >> LOG
    chi2 = 0; dof = eCount - 1
    for (i=1; i<=eCount; i++) {
        e = eList[i]
        expected = overall_p * tot[e]
        chi2 += (det[e] - expected)^2 / expected
    }
    crit05 = dof + 1.645*sqrt(2*dof)
    crit01 = dof + 2.326*sqrt(2*dof)
    printf "  chi2 = %.2f, dof = %d\n", chi2, dof >> LOG
    printf "  (Critical values: chi2(%d,0.05) ~ %.1f, chi2(%d,0.01) ~ %.1f)\n", dof, crit05, dof, crit01 >> LOG
    if (chi2 > crit01) printf "  Result: REJECT uniformity at p<0.01\n" >> LOG
    else if (chi2 > crit05) printf "  Result: REJECT uniformity at p<0.05\n" >> LOG
    else printf "  Result: Cannot reject uniformity\n" >> LOG

    printf "\n%s\n", "================================================================================" >> LOG
    printf "Per-energy file breakdown:\n" >> LOG
    printf "%s\n", "================================================================================" >> LOG
    for (i=1; i<=eCount; i++) {
        e = eList[i]
        printf "\n%s eV:%s\n", e, files[e] >> LOG
    }
}
'

echo "Done."
cat "$LOGOUT"
echo ""
echo "--- CSV ---"
cat "$CSVOUT"
