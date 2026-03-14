#!/usr/bin/env python3
"""
Exit-angle density histogram comparing spin-Z vs spin-Y runs, overlaid by energy.
Points (not bars), density-normalized, with fine binning near 87° and 93° peaks.
"""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# --- File paths ---
Z_DIR = os.path.expanduser("~/results/resultZ")
Y_DIR = "/mnt/c/Users/marcf/IdeaProjects/ELektron2/results"

# --- Collect files by (spin_axis, energy) ---
def parse_energy(fname):
    m = re.search(r'(\d+)eV', fname)
    return int(m.group(1)) if m else None

def load_angles(filepath):
    """Load exit angle (column 16, 0-indexed) from .dat text file."""
    angles = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.split()
            if len(cols) >= 16:
                angles.append(float(cols[15]))  # angle_deg (0-indexed col 15)
    return np.array(angles)

# Gather Z-axis files (1M electrons, Mar 14)
z_files = defaultdict(list)
for f in sorted(glob.glob(os.path.join(Z_DIR, "*1000000.dat"))):
    e = parse_energy(os.path.basename(f))
    if e is not None:
        z_files[e].append(f)

# Gather Y-axis files (750K electrons, Mar 12)
y_files = defaultdict(list)
for f in sorted(glob.glob(os.path.join(Y_DIR, "2026-03-12_*rocm*750000.dat"))):
    e = parse_energy(os.path.basename(f))
    if e is not None:
        y_files[e].append(f)

# Overlapping energies
common_energies = sorted(set(z_files.keys()) & set(y_files.keys()))
print(f"Common energies: {common_energies}")

# --- Non-uniform bins: fine near 87° and 93°, coarser elsewhere ---
bins_coarse_lo = np.arange(80, 85, 1.0)
bins_med_1     = np.arange(85, 86, 0.5)
bins_fine_1    = np.arange(86, 88.5, 0.1)   # fine around 87°
bins_med_2     = np.arange(88.5, 91.5, 0.5)
bins_fine_2    = np.arange(91.5, 94.5, 0.1) # fine around 93°
bins_med_3     = np.arange(94.5, 96, 0.5)
bins_coarse_hi = np.arange(96, 101, 1.0)
bins = np.unique(np.concatenate([
    bins_coarse_lo, bins_med_1, bins_fine_1, bins_med_2,
    bins_fine_2, bins_med_3, bins_coarse_hi
]))

# --- Plot ---
fig, axes = plt.subplots(len(common_energies), 1,
                          figsize=(12, 3.5 * len(common_energies)),
                          sharex=True)
if len(common_energies) == 1:
    axes = [axes]

colors_z = '#2166ac'  # blue
colors_y = '#b2182b'  # red

for ax, energy in zip(axes, common_energies):
    # Merge all runs at this energy
    z_angles = np.concatenate([load_angles(f) for f in z_files[energy]])
    y_angles = np.concatenate([load_angles(f) for f in y_files[energy]])

    # Compute density histograms
    z_counts, _ = np.histogram(z_angles, bins=bins)
    y_counts, _ = np.histogram(y_angles, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths  = np.diff(bins)

    z_density = z_counts / (len(z_angles) * bin_widths) if len(z_angles) > 0 else z_counts * 0
    y_density = y_counts / (len(y_angles) * bin_widths) if len(y_angles) > 0 else y_counts * 0

    # Forward-exit percentage
    z_total = sum(int(open(f).readline().split('simulations:')[1]) if 'simulations:' in open(f).readline() else 0 for f in z_files[energy])
    y_total = sum(int(open(f).readline().split('simulations:')[1]) if 'simulations:' in open(f).readline() else 0 for f in y_files[energy])

    # Read total simulations from headers
    z_total_sims = 0
    for f in z_files[energy]:
        with open(f) as fh:
            for line in fh:
                if 'Total simulations:' in line:
                    z_total_sims += int(line.split(':')[1].strip())
                    break
    y_total_sims = 0
    for f in y_files[energy]:
        with open(f) as fh:
            for line in fh:
                if 'Total simulations:' in line:
                    y_total_sims += int(line.split(':')[1].strip())
                    break

    z_pct = 100.0 * len(z_angles) / z_total_sims if z_total_sims > 0 else 0
    y_pct = 100.0 * len(y_angles) / y_total_sims if y_total_sims > 0 else 0

    ax.plot(bin_centers, z_density, 'o-', color=colors_z, ms=3, lw=1.0, alpha=0.85,
            label=f'spin-Z ({len(z_angles):,} det / {z_total_sims:,} = {z_pct:.2f}%)')
    ax.plot(bin_centers, y_density, 's-', color=colors_y, ms=3, lw=1.0, alpha=0.85,
            label=f'spin-Y ({len(y_angles):,} det / {y_total_sims:,} = {y_pct:.2f}%)')

    ax.set_ylabel('Density')
    ax.set_title(f'{energy} eV', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(84, 97)
    ax.grid(True, alpha=0.3)

    # Mark the peaks
    for peak in [87, 93]:
        ax.axvline(peak, color='gray', ls='--', lw=0.7, alpha=0.5)

axes[-1].set_xlabel('Exit angle (deg)')
fig.suptitle('Exit Angle Distribution: Spin-Z (1M e⁻, Mar 14) vs Spin-Y (750K e⁻, Mar 12)',
             fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()

outpath = os.path.join(Z_DIR, 'exit_angle_Zvs Y_comparison.png')
fig.savefig(outpath, dpi=180, bbox_inches='tight')
print(f"\nSaved: {outpath}")

# Also print summary table
print(f"\n{'Energy':>8}  {'Z det%':>8}  {'Y det%':>8}  {'Δ':>8}")
for energy in common_energies:
    z_a = np.concatenate([load_angles(f) for f in z_files[energy]])
    y_a = np.concatenate([load_angles(f) for f in y_files[energy]])
    zt, yt = 0, 0
    for f in z_files[energy]:
        with open(f) as fh:
            for line in fh:
                if 'Total simulations:' in line:
                    zt += int(line.split(':')[1].strip()); break
    for f in y_files[energy]:
        with open(f) as fh:
            for line in fh:
                if 'Total simulations:' in line:
                    yt += int(line.split(':')[1].strip()); break
    zp = 100*len(z_a)/zt if zt else 0
    yp = 100*len(y_a)/yt if yt else 0
    print(f'{energy:>8}  {zp:>8.3f}  {yp:>8.3f}  {zp-yp:>+8.3f}')
