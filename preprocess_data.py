# preprocess_data.py (FINAL, with "take from the middle" logic)

import os
import numpy as np
from scipy import signal
from tqdm import tqdm

def preprocess_dataset():
    """
    MODIFIED: Reads raw .sigmf-data files, converts them to I/Q data,
    and saves a short, 4096-sample snippet from the middle of the signal.
    """
    # --- Configuration ---
    SOURCE_DIR = '/Users/MAC/Downloads/5 GHz Bandwidth' # Or your raw data path
    DEST_DIR = '/Users/MAC/Downloads/5 GHz Bandwidth_IQ_Preprocessed_Short' # New destination for short samples
    
    # Signal Processing Parameters
    SAMPLING_RATE = 160e9; IF_FREQ = 10e9
    FILTER_CUTOFF = 12e9; FILTER_ORDER = 8

    # --- THIS IS THE KEY CHANGE ---
    TARGET_IQ_LENGTH = 4096 # Short, fast, and a power of 2

    print(f"Source data directory: {SOURCE_DIR}")
    print(f"Destination for preprocessed I/Q data: {DEST_DIR}")
    
    file_paths = []
    for subdir, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith('.sigmf-data'):
                file_paths.append(os.path.join(subdir, file))
    
    print(f"\nFound {len(file_paths)} files to preprocess.")

    # Design the Low-Pass Filter once
    nyq = 0.5 * SAMPLING_RATE; normal_cutoff = FILTER_CUTOFF / nyq
    b, a = signal.butter(FILTER_ORDER, normal_cutoff, btype='low', analog=False)

    for path in tqdm(file_paths, desc="Converting to short I/Q"):
        try:
            # 1. Load and convert the full waveform to I/Q
            if_waveform = np.fromfile(path, dtype=np.float32)
            t = np.arange(len(if_waveform)) / SAMPLING_RATE
            local_osc_cos = np.cos(2 * np.pi * IF_FREQ * t)
            local_osc_sin = np.sain(2 * np.pi * IF_FREQ * t)
            i_mixed = if_waveform * local_osc_cos
            q_mixed = if_waveform * local_osc_sin
            i_baseband = signal.filtfilt(b, a, i_mixed)
            q_baseband = signal.filtfilt(b, a, q_mixed)
            iq_data = np.stack([i_baseband, q_baseband], axis=0)

            # --- THIS IS THE NEW "TAKE FROM THE MIDDLE" LOGIC ---
            current_length = iq_data.shape[1]
            
            if current_length < TARGET_IQ_LENGTH:
                # If the signal is too short, pad it with zeros to reach the target length
                padding_needed = TARGET_IQ_LENGTH - current_length
                # Center the padding
                pad_left = padding_needed // 2
                pad_right = padding_needed - pad_left
                final_iq = np.pad(iq_data, ((0, 0), (pad_left, pad_right)), mode='constant')
            else:
                # If the signal is long enough, find the middle and take a snippet
                start_index = (current_length - TARGET_IQ_LENGTH) // 2
                end_index = start_index + TARGET_IQ_LENGTH
                final_iq = iq_data[:, start_index:end_index]
            # --- END OF NEW LOGIC ---

            # Create destination path and save
            relative_path = os.path.relpath(path, SOURCE_DIR)
            dest_path = os.path.join(DEST_DIR, relative_path)
            dest_path = os.path.splitext(dest_path)[0] + '.npy'
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            np.save(dest_path, final_iq.astype(np.float32))

        except Exception as e:
            print(f"Failed to process {path}: {e}")

    print("\nPreprocessing to short I/Q complete!")
    print(f"All processed data is saved in: {DEST_DIR}")

if __name__ == '__main__':
    preprocess_dataset()