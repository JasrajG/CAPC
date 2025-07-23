# /Users/MAC/Projects/CAPC-IQ-project/inspect_iq_data.py (FINAL CORRECTED VERSION)

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def inspect_file(file_path):
    """
    Loads a single preprocessed I/Q data file and provides
    numerical and graphical analysis with clear, subsampled plots.
    """
    # --- 1. Load the data ---
    try:
        iq_data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error loading the numpy file: {e}")
        return

    # --- 2. Validate the data shape ---
    if iq_data.ndim != 2 or iq_data.shape[0] != 2:
        print(f"Error: Expected data shape to be (2, N), but got {iq_data.shape}.")
        return

    i_samples = iq_data[0, :]
    q_samples = iq_data[1, :]

    # --- 3. Print Numerical Output (Unchanged) ---
    print("\n" + "="*50)
    print(f"  ANALYSIS FOR: {os.path.basename(file_path)}")
    print("="*50)
    print(f"Data Shape: {iq_data.shape}")
    print(f"Number of I/Q Samples: {len(i_samples)}")
    print(f"Data Type: {iq_data.dtype}")
    print("\n--- First 15 I/Q Samples ---")
    for i in range(15):
        print(f"Sample {i+1:>2}:   I = {i_samples[i]: 9.6f},   Q = {q_samples[i]: 9.6f}")
    print("="*50 + "\n")

    # --- 4. Define Subsampling Parameters ---
    SAMPLES_FOR_TIME_PLOT = 1000      # How many points to show in the time-domain plot
    SAMPLES_FOR_CONSTELLATION = 5000  # How many points for the constellation scatter plot

    # --- 5. Create Graphical Output ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- THIS IS THE KEY CHANGE for the TIME PLOT ---
    # Left Panel: Plot only the FIRST part of the signal to see the waveform
    time_plot_len = min(len(i_samples), SAMPLES_FOR_TIME_PLOT)
    ax1.plot(i_samples[:time_plot_len], label='I Component', color='royalblue')
    ax1.plot(q_samples[:time_plot_len], label='Q Component', color='orangered', alpha=0.8)
    ax1.set_title(f'I/Q Data vs. Time (First {time_plot_len} Samples)')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Constellation plot uses RANDOM samples from the WHOLE signal ---
    if len(i_samples) > SAMPLES_FOR_CONSTELLATION:
        rand_indices = np.random.choice(len(i_samples), SAMPLES_FOR_CONSTELLATION, replace=False)
        i_const = i_samples[rand_indices]
        q_const = q_samples[rand_indices]
        constellation_title = f'I/Q Constellation ({SAMPLES_FOR_CONSTELLATION:,} Random Samples)'
    else:
        i_const, q_const = i_samples, q_samples
        constellation_title = f'I/Q Constellation (All {len(i_samples):,} Samples)'

    # Right Panel: Constellation Diagram
    ax2.scatter(i_const, q_const, s=5, alpha=0.4)
    ax2.set_title(constellation_title)
    ax2.set_xlabel('I (In-phase)')
    ax2.set_ylabel('Q (Quadrature)')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axis('equal')

    plt.suptitle(f"Inspection of: {os.path.basename(file_path)}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inspect a preprocessed I/Q data file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('file_path', type=str, help="Path to the preprocessed .npy I/Q file to inspect.")
    args = parser.parse_args()

    inspect_file(args.file_path)