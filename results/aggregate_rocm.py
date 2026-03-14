import os, re, math
from collections import defaultdict

data = defaultdict(lambda: {"detected": 0, "total": 0, "files": []})

for fname in sorted(os.listdir("results")):
    if "rocm" not in fname or not fname.endswith(".dat"):
        continue
    path = os.path.join("results", fname)
    energy = detected = total = None
    with open(path) as f:
        for line in f:
            if line.startswith("# startEnergy:"):
                energy = float(line.split(":")[1].strip().replace(" eV",""))
            elif line.startswith("# Total simulations:"):
                total = int(line.split(":")[1].strip())
            elif line.startswith("# Detected:"):
                detected = int(line.split(":")[1].strip())
            if energy and total and detected is not None:
                break
    if energy is not None and total and detected is not None:
        data[energy]["detected"] += detected
        data[energy]["total"] += total
        data[energy]["files"].append(fname)

# Write CSV
with open("results/rocm_summary.csv", "w") as csv:
    csv.write("energy_eV,detected,total,percent,stderr_pct,files\n")
    for e in sorted(data):
        d = data[e]["detected"]
        n = data[e]["total"]
        p = d / n
        se = math.sqrt(p * (1 - p) / n) if n > 0 else 0
        csv.write(f"{e},{d},{n},{100*p:.4f},{100*se:.4f},{len(data[e]['files'])}\n")

# Write log with full analysis
with open("results/rocm_summary.log", "w") as log:
    log.write("=" * 80 + "\n")
    log.write("ELektron2 ROCm Energy Scan — Aggregated Summary\n")
    log.write(f"Generated: 2026-03-13\n")
    log.write("=" * 80 + "\n\n")
    log.write(f"{'Energy':>10} {'Detected':>10} {'Total':>10} {'Rate %':>10} {'±1σ %':>10} {'±2σ %':>10} {'Files':>6}\n")
    log.write("-" * 68 + "\n")

    results = []
    for e in sorted(data):
        d = data[e]["detected"]
        n = data[e]["total"]
        p = d / n
        se = math.sqrt(p * (1 - p) / n) if n > 0 else 0
        results.append((e, d, n, p, se))
        log.write(f"{e:>10.2f} {d:>10d} {n:>10d} {100*p:>10.4f} {100*se:>10.4f} {200*se:>10.4f} {len(data[e]['files']):>6d}\n")

    log.write("-" * 68 + "\n\n")

    # Overall stats
    total_d = sum(r[1] for r in results)
    total_n = sum(r[2] for r in results)
    overall_p = total_d / total_n
    overall_se = math.sqrt(overall_p * (1 - overall_p) / total_n)
    log.write(f"Overall: {total_d}/{total_n} = {100*overall_p:.4f}% ± {100*overall_se:.4f}% (1σ)\n\n")

    # Find peak
    peak = max(results, key=lambda r: r[3])
    log.write(f"Peak detection: {peak[0]:.2f} eV at {100*peak[3]:.4f}% ± {100*peak[4]:.4f}%\n")

    # Find minimum
    trough = min(results, key=lambda r: r[3])
    log.write(f"Min  detection: {trough[0]:.2f} eV at {100*trough[3]:.4f}% ± {100*trough[4]:.4f}%\n\n")

    # Statistical significance: peak vs trough
    z = (peak[3] - trough[3]) / math.sqrt(peak[4]**2 + trough[4]**2)
    log.write(f"Peak vs min: z = {z:.2f}σ")
    if z > 3:
        log.write(" (>3σ, statistically significant)\n")
    elif z > 2:
        log.write(" (>2σ, suggestive)\n")
    else:
        log.write(" (<2σ, not significant)\n")

    # Chi-squared test for uniformity
    log.write("\nChi-squared test (H0: uniform detection rate across all energies):\n")
    expected_p = overall_p
    chi2 = 0
    dof = len(results) - 1
    for e, d, n, p, se in results:
        expected = expected_p * n
        chi2 += (d - expected)**2 / expected
    log.write(f"  χ² = {chi2:.2f}, dof = {dof}\n")
    # Rough p-value approximation
    log.write(f"  (Critical values: χ²({dof},0.05) ≈ {dof + 1.645*math.sqrt(2*dof):.1f}, ")
    log.write(f"χ²({dof},0.01) ≈ {dof + 2.326*math.sqrt(2*dof):.1f})\n")
    if chi2 > dof + 2.326*math.sqrt(2*dof):
        log.write("  Result: REJECT uniformity at p<0.01\n")
    elif chi2 > dof + 1.645*math.sqrt(2*dof):
        log.write("  Result: REJECT uniformity at p<0.05\n")
    else:
        log.write("  Result: Cannot reject uniformity\n")

    log.write("\n" + "=" * 80 + "\n")
    log.write("Per-file breakdown:\n")
    log.write("=" * 80 + "\n")
    for e in sorted(data):
        log.write(f"\n{e:.2f} eV:\n")
        for fn in data[e]["files"]:
            log.write(f"  {fn}\n")

print("Done. Written rocm_summary.csv and rocm_summary.log")

# Print the log to stdout
with open("results/rocm_summary.log") as f:
    print(f.read())
