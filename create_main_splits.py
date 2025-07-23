# create_main_splits.py (FINAL AND CORRECTED TO USE SHORT SAMPLES)

import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_all_npy_files(root_dir):
    """
    Scans a directory to get all .npy file paths and their corresponding labels
    for stratification.
    """
    file_paths, labels = [], []
    mod_types = ['4PSK', '8PSK', '16QAM', '64QAM']
    class_to_idx = {cls_name: i for i, cls_name in enumerate(mod_types)}
    
    print(f"Scanning directory: {root_dir}")
    for subdir, _, files in os.walk(root_dir):
        mod_name = os.path.basename(subdir).split('_')[0]
        if mod_name in class_to_idx:
            for file in files:
                if file.endswith('.npy'):
                    file_paths.append(os.path.join(subdir, file))
                    labels.append(class_to_idx[mod_name])
    print(f"--> Found {len(file_paths)} files.")
    return file_paths, labels

def create_experiment_splits():
    """
    Performs the definitive data split using the SHORT (4096-sample) data.
    """
    print("\n--- Creating Final Experiment Data Splits from SHORT Samples ---")
    
    # --- THIS IS THE CRITICAL FIX ---
    # The paths now correctly point to the directories containing the short samples.
    FIVE_GHZ_DIR = '/Users/MAC/Downloads/5 GHz Bandwidth_IQ_Preprocessed_Short'
    TEN_GHZ_DIR = '/Users/MAC/Downloads/10 GHz Bandwidth_IQ_Preprocessed_Short'
    # --- END OF FIX ---

    OUTPUT_DIR = 'final_run_splits'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Handle the 5 GHz Dataset ---
    all_5ghz_files, all_5ghz_labels = get_all_npy_files(FIVE_GHZ_DIR)
    
    pretrain_files, holding_pool_files, _, holding_pool_labels = train_test_split(
        all_5ghz_files, all_5ghz_labels, train_size=0.8, random_state=42, stratify=all_5ghz_labels
    )
    
    # From the 20% holding pool, create the final 4000-sample downstream pool.
    downstream_5ghz_pool, _ = train_test_split(
        holding_pool_files, train_size=4000, random_state=42, stratify=holding_pool_labels
    )

    # --- Step 2: Handle the 10 GHz Dataset ---
    all_10ghz_files, all_10ghz_labels = get_all_npy_files(TEN_GHZ_DIR)
    
    # Create the 4000-sample downstream pool directly from the full 10GHz dataset.
    downstream_10ghz_pool, _ = train_test_split(
        all_10ghz_files, train_size=4000, random_state=42, stratify=all_10ghz_labels
    )

    # --- Step 3: Write the final file lists ---
    pretrain_path = os.path.join(OUTPUT_DIR, 'pretrain_5ghz.txt')
    downstream_5ghz_path = os.path.join(OUTPUT_DIR, 'downstream_5ghz_4k_pool.txt')
    downstream_10ghz_path = os.path.join(OUTPUT_DIR, 'downstream_10ghz_4k_pool.txt')

    with open(pretrain_path, 'w') as f:
        f.write('\n'.join(pretrain_files))
    with open(downstream_5ghz_path, 'w') as f:
        f.write('\n'.join(downstream_5ghz_pool))
    with open(downstream_10ghz_path, 'w') as f:
        f.write('\n'.join(downstream_10ghz_pool))
        
    # --- Final Summary ---
    print("\n" + "="*60)
    print("      DATA SPLITTING COMPLETE      ")
    print("="*60)
    print(f"Pre-training Pool (from 5GHz): {len(pretrain_files)} files.")
    print(f"Downstream 5GHz Pool (from 5GHz): {len(downstream_5ghz_pool)} files.")
    print(f"Downstream 10GHz Pool (from 10GHz): {len(downstream_10ghz_pool)} files.")
    print(f"\nFile lists saved in '{OUTPUT_DIR}' directory.")
    print("="*60)

if __name__ == "__main__":
    create_experiment_splits()