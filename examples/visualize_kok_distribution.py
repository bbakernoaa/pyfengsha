"""
Example script to visualize the Kok aerosol distribution.

This script generates a bar plot of the normalized volume distribution
for a set of aerosol size bins, as calculated by the
kok_aerosol_distribution function. The resulting plot is saved to
'kok_aerosol_distribution.png'.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyfengsha.fengsha import kok_aerosol_distribution

def plot_kok_distribution():
    """
    Generates and saves a bar plot of the Kok aerosol distribution.
    """
    # Input data for the distribution function (same as in the unit test)
    radius = np.array([0.1, 0.5, 1.0])
    r_low = np.array([0.05, 0.45, 0.95])
    r_up = np.array([0.15, 0.55, 1.05])
    bin_labels = [f'{low}-{up}μm' for low, up in zip(r_low, r_up)]

    # Calculate the distribution
    distribution = kok_aerosol_distribution(radius, r_low, r_up)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bin_labels, distribution, color='skyblue', edgecolor='black')

    # Add labels and title
    ax.set_xlabel('Aerosol Size Bins (Radius)', fontsize=12)
    ax.set_ylabel('Normalized Volume Distribution', fontsize=12)
    ax.set_title("Kok's Aerosol Distribution", fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(distribution) * 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data labels on top of bars
    for i, v in enumerate(distribution):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')

    # Save the figure
    output_filename = 'kok_aerosol_distribution.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    plot_kok_distribution()
