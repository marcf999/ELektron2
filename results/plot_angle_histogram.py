#!/usr/bin/env python3
"""Overlaid exit-angle histograms comparing different beam energies."""

import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Energies to compare (eV) — spread across the scan range
ENERGIES_TO_PLOT = [4991, 4999, 5000, 5003, 5005, 5007, 5010]

def parse_dat(filepath):
    """Return (energy_eV, angles_deg) from a .dat file."""
    energy = None
    angles = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("# startEnergy:"):
                energy = float(line.split(":")[1].strip().split()[0])
            elif not line.startswith("#") and line.strip():
                cols = line.split()
                if len(cols) >= 16:
                    try:
                        angles.append(float(cols[15]))
                    except ValueError:
                        pass
    return energy, np.array(angles)


def main():
    # Collect all .dat files (skip machine83 duplicates)
    pattern = os.path.join(RESULTS_DIR, "*_rocm-dp853_*eV_*.dat")
    files = sorted(glob.glob(pattern))

    # Group angles by energy
    data = defaultdict(list)
    for f in files:
        energy, angles = parse_dat(f)
        if energy is not None and len(angles) > 0:
            rounded = round(energy)
            data[rounded].append(angles)

    # Merge arrays per energy
    merged = {e: np.concatenate(arrs) for e, arrs in data.items() if round(e) in ENERGIES_TO_PLOT}

    # Determine common bin edges from all plotted data
    all_angles = np.concatenate(list(merged.values()))
    lo, hi = np.percentile(all_angles, [0.5, 99.5])
    bins = np.linspace(lo, hi, 200)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(ENERGIES_TO_PLOT)))

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for color, e in zip(colors, sorted(merged.keys())):
        counts, _ = np.histogram(merged[e], bins=bins, density=True)
        ax.plot(bin_centers, counts, '-', color=color,
                linewidth=1.4, alpha=0.85, label=f"{e} eV (n={len(merged[e])})")

    ax.set_xlabel("Exit angle (deg)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Exit-angle distribution by beam energy", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(82, 98)
    ax.tick_params(labelsize=11)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "angle_histogram.png")
    fig.savefig(out, dpi=180)
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
