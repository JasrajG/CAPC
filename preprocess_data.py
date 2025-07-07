# /Users/MAC/Projects/CAPC-replica/preprocess_data.py

import os
import numpy as np
from scipy import signal
from tqdm import tqdm # A library for beautiful progress bars

def preprocess_dataset():
    """
    This script reads all raw .sigmf-data files, converts them to spectrograms,
    and saves them as .npy files in a new directory.
    """
    # --- Configuration ---
    SOURCE_DIR = '/Users/MAC/Downloads/10 GHz Bandwidth'
    DEST_DIR = '/Users/MAC/Downloads/10 GHz Bandwidth_Preprocessed'
    
    # Spectrogram parameters (must match what the model expects)
    NPERSEG = 256
    NOVERLAP = 128
    TARGET_TIME_BINS = 184

    print(f"Source data directory: {SOURCE_DIR}")
    print(f"Destination for preprocessed data: {DEST_DIR}")
    
    # Find all the raw data files
    file_paths = []
    for subdir, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith('.sigmf-data'):
                file_paths.append(os.path.join(subdir, file))
    
    print(f"\nFound {len(file_paths)} files to preprocess.")

    # --- Start Pre-processing Loop ---
    for path in tqdm(file_paths, desc="Preprocessing files"):
        try:
            # --- 1. Load the raw 1D waveform ---
            waveform = np.fromfile(path, dtype=np.float32)

            # --- 2. Perform STFT ---
            _, _, Zxx = signal.stft(waveform, fs=1, nperseg=NPERSEG, noverlap=NOVERLAP)
            spectrogram = np.expand_dims(np.abs(Zxx), axis=0)

            # --- 3. Pad or Truncate to a fixed size ---
            _, _, current_time_bins = spectrogram.shape
            if current_time_bins > TARGET_TIME_BINS:
                spectrogram = spectrogram[:, :, :TARGET_TIME_BINS]
            elif current_time_bins < TARGET_TIME_BINS:
                padding_needed = TARGET_TIME_BINS - current_time_bins
                spectrogram = np.pad(spectrogram, ((0, 0), (0, 0), (0, padding_needed)), mode='constant')

            # --- 4. Create the destination path ---
            # This mirrors the source directory structure inside the new destination
            relative_path = os.path.relpath(path, SOURCE_DIR)
            dest_path = os.path.join(DEST_DIR, relative_path)
            # Change the file extension from .sigmf-data to .npy
            dest_path = os.path.splitext(dest_path)[0] + '.npy'
            
            # Create the destination folder if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # --- 5. Save the processed spectrogram ---
            np.save(dest_path, spectrogram.astype(np.float32))

        except Exception as e:
            print(f"Failed to process {path}: {e}")

    print("\nPreprocessing complete!")
    print(f"All processed data is saved in: {DEST_DIR}")

if __name__ == '__main__':
    # Before running, make sure you have tqdm installed: pip install tqdm
    preprocess_dataset()