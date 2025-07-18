# /Users/MAC/Projects/CAPC-replica/create_main_splits.py

import os
import numpy as np
from sklearn.model_selection import train_test_split # We use scikit-learn for a robust split
from dataset import TeraNovaDataset # We use this to get the file list and labels

def create_main_splits():
    """
    This script performs the "Great Divide". It splits the entire dataset into
    a pre-training pool and a downstream evaluation pool, ensuring no leakage.
    It saves the lists of file paths for each pool.
    """
    print("--- Starting the Great Divide: Splitting data into Pre-training and Downstream pools ---")
    
    # --- Configuration ---
    # We will use an 80/20 split. 80% for pre-training, 20% for downstream tasks.
    PRETRAIN_RATIO = 0.8
    DATA_DIR = '/Users/MAC/Downloads/5 GHz Bandwidth_Preprocessed'
    OUTPUT_DIR = '/Users/MAC/Projects/CAPC-replica/data_splits' # Where we'll save our file lists

    # --- Load the full dataset to get file paths and labels ---
    # We only do this to get the list of files, not to load the data itself.
    full_dataset = TeraNovaDataset(root_dir=DATA_DIR)
    
    # Use scikit-learn's train_test_split to create a stratified split.
    # Stratifying ensures both pools have the same percentage of each class.
    indices = np.arange(len(full_dataset))
    
    pretrain_indices, downstream_indices = train_test_split(
        indices,
        train_size=PRETRAIN_RATIO,
        random_state=42, # Use a fixed random state for reproducibility
        stratify=full_dataset.labels # This is the key to a balanced split
    )

    # --- Get the file paths corresponding to the chosen indices ---
    all_files = np.array(full_dataset.file_paths)
    pretrain_files = all_files[pretrain_indices]
    downstream_files = all_files[downstream_indices]

    # --- Save the file lists to disk ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pretrain_list_path = os.path.join(OUTPUT_DIR, 'pretrain_files.txt')
    downstream_list_path = os.path.join(OUTPUT_DIR, 'downstream_files.txt')
    
    with open(pretrain_list_path, 'w') as f:
        for file_path in pretrain_files:
            f.write(f"{file_path}\n")
            
    with open(downstream_list_path, 'w') as f:
        for file_path in downstream_files:
            f.write(f"{file_path}\n")

    print(f"\nSplit complete!")
    print(f"Total files: {len(full_dataset)}")
    print(f"Pre-training pool size: {len(pretrain_files)} files (saved to {pretrain_list_path})")
    print(f"Downstream pool size: {len(downstream_files)} files (saved to {downstream_list_path})")
    
if __name__ == '__main__':
    # We need scikit-learn for this, make sure it's installed
    # pip install scikit-learn
    create_main_splits()