#!/usr/bin/env python3
"""
Density vs exit-angle for all spin x energy .dat files (2M electrons each).
Layout: 3 rows (one per energy) x 1 col, 5 spin curves per panel.
"""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = r"\\wsl$\Ubuntu\home\marcf\resultsSpin"

def parse_energy(fname):
    m = re.search(r'(\d+)eV', fname)
    return int(m.group(1)) if m else None

def parse_spin(fname):
    m = re.search(r'spin([+\-]?\w+)', fname)
    return m.group(1) if m else None

def load_angles(filepath):
    angles = []
    total_sims = 0
    with open(filepath) as f:
        for line in f:
            if 'Total simulations:' in line:
                total_sims = int(line.split(':')[1].strip())
            if line.startswith('#'):
                continue
            cols = line.split()
            if len(cols) >= 16:
                try:
                    angles.append(float(cols[15]))
                except ValueError:
                    pass
    return np.array(angles), total_sims

# Gather files
files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*2000000.dat")))
print(f"Found {len(files)} files")

# Organize by (energy, spin)
data = {}
for f in files:
    energy = parse_energy(os.path.basename(f))
    spin = parse_spin(os.path.basename(f))
    if energy and spin:
        angles, total = load_angles(f)
        data[(energy, spin)] = (angles, total)
        pct = 100.0 * len(angles) / total if total else 0
        print(f"  {energy} eV  spin {spin:>7s}: {len(angles):>7,} detected / {total:,} = {pct:.2f}%")

energies = sorted(set(e for e, s in data.keys()))
spins = ['+x', '+y', '-y', '+z', 'random']
spin_colors = {'+x': '#e41a1c', '+y': '#377eb8', '-y': '#ff7f00', '+z': '#4daf4a', 'random': '#984ea3'}
spin_markers = {'+x': 'o', '+y': 's', '-y': 'v', '+z': '^', 'random': 'D'}

# Non-uniform bins: fine near peaks at ~87 deg and ~93 deg
bins = np.unique(np.concatenate([
    np.arange(80, 85, 1.0),
    np.arange(85, 86, 0.5),
    np.arange(86, 88.5, 0.1),
    np.arange(88.5, 91.5, 0.5),
    np.arange(91.5, 94.5, 0.1),
    np.arange(94.5, 96, 0.5),
    np.arange(96, 101, 1.0),
]))
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_widths = np.diff(bins)

fig, axes = plt.subplots(len(energies), 1, figsize=(13, 4 * len(energies)), sharex=True)
if len(energies) == 1:
    axes = [axes]

for ax, energy in zip(axes, energies):
    for spin in spins:
        key = (energy, spin)
        if key not in data:
            continue
        angles, total = data[key]
        counts, _ = np.histogram(angles, bins=bins)
        density = counts / (len(angles) * bin_widths) if len(angles) > 0 else counts * 0.0
        pct = 100.0 * len(angles) / total if total else 0

        label = f"spin {spin} ({len(angles):,} det, {pct:.2f}%)"
        ax.plot(bin_centers, density,
                marker=spin_markers[spin], ms=3, lw=1.2, alpha=0.85,
                color=spin_colors[spin], label=label)

    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{energy} eV', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(84, 97)
    ax.grid(True, alpha=0.3)
    for peak in [87, 93]:
        ax.axvline(peak, color='gray', ls='--', lw=0.7, alpha=0.5)

axes[-1].set_xlabel('Exit angle (deg)', fontsize=12)
fig.suptitle('Exit-Angle Density: Spin Orientation x Energy (2M electrons each, 5 orientations)',
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()

outpath = os.path.join(RESULTS_DIR, 'spin_angle_density_all15.png')
fig.savefig(outpath, dpi=180, bbox_inches='tight')
print(f"\nSaved: {outpath}")
