# full_experiment.py (FINAL, CORRECTED LOGIC)

import os
import subprocess
import json
import glob
from sklearn.model_selection import train_test_split
import sys
import pty
import select

# Helper functions are unchanged
def get_all_npy_files(root_dir):
    file_paths, labels, class_to_idx = [], [], {}; mod_types = ['4PSK', '8PSK', '16QAM', '64QAM']
    for i, cls_name in enumerate(mod_types): class_to_idx[cls_name] = i
    for subdir, _, files in os.walk(root_dir):
        mod_name = os.path.basename(subdir).split('_')[0]
        if mod_name in class_to_idx:
            for file in files:
                if file.endswith('.npy'): file_paths.append(os.path.join(subdir, file)); labels.append(class_to_idx[mod_name])
    return file_paths, labels

def create_final_experiment_splits():
    print("--- Step 1: Creating Final Experiment Data Splits ---")
    FIVE_GHZ_DIR = '/Users/MAC/Downloads/5 GHz Bandwidth_IQ_Preprocessed'; TEN_GHZ_DIR = '/Users/MAC/Downloads/10 GHz Bandwidth_IQ_Preprocessed'
    OUTPUT_DIR = 'final_run_splits'; os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_5ghz_files, all_5ghz_labels = get_all_npy_files(FIVE_GHZ_DIR)
    pretrain_files, holding_pool_5ghz_files, _, holding_pool_5ghz_labels = train_test_split(all_5ghz_files, all_5ghz_labels, train_size=0.8, random_state=42, stratify=all_5ghz_labels)
    downstream_5ghz_4k_pool, _ = train_test_split(holding_pool_5ghz_files, train_size=4000, random_state=42, stratify=holding_pool_5ghz_labels)
    all_10ghz_files, all_10ghz_labels = get_all_npy_files(TEN_GHZ_DIR)
    downstream_10ghz_4k_pool, _ = train_test_split(all_10ghz_files, train_size=4000, random_state=42, stratify=all_10ghz_labels)
    split_paths = {"pretrain": os.path.join(OUTPUT_DIR, 'pretrain_5ghz.txt'), "downstream_5ghz": os.path.join(OUTPUT_DIR, 'downstream_5ghz_4k_pool.txt'), "downstream_10ghz": os.path.join(OUTPUT_DIR, 'downstream_10ghz_4k_pool.txt')}
    with open(split_paths["pretrain"], 'w') as f: f.write('\n'.join(pretrain_files))
    with open(split_paths["downstream_5ghz"], 'w') as f: f.write('\n'.join(downstream_5ghz_4k_pool))
    with open(split_paths["downstream_10ghz"], 'w') as f: f.write('\n'.join(downstream_10ghz_4k_pool))
    print(f"--> Splits created: Pre-train ({len(pretrain_files)}), 5GHz Downstream ({len(downstream_5ghz_4k_pool)}), 10GHz Downstream ({len(downstream_10ghz_4k_pool)})")
    return split_paths

# run_command function is correct and does not need changes
def run_command(command):
    print(f"\nRUNNING: {' '.join(command)}"); master_fd, slave_fd = pty.openpty(); process = subprocess.Popen(command, stdout=slave_fd, stderr=slave_fd, text=True, bufsize=1); os.close(slave_fd); all_output = []
    while True:
        try:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if r:
                output = os.read(master_fd, 1024).decode('utf-8', errors='ignore')
                if not output: break
                sys.stdout.write(output); sys.stdout.flush(); all_output.append(output)
            if process.poll() is not None: break
        except OSError: break
    process.wait(); os.close(master_fd)
    if process.returncode != 0:
        full_log = "".join(all_output)
        print("\n" + "="*80); print("!!! SCRIPT FAILED !!!"); print(f"COMMAND: {' '.join(command)}"); print(f"EXIT CODE: {process.returncode}")
        if "Traceback" in full_log: print("\n--- CAPTURED TRACEBACK ---"); print(full_log)
        print("="*80); exit(1)
    full_log = "".join(all_output); last_line = ""
    for line in full_log.strip().split('\n'):
        if line.strip(): last_line = line.strip()
    return last_line

def main():
    # --- THIS SCRIPT ASSUMES THE RECIPE FILE ALREADY EXISTS ---
    RECIPE_FILE = "best_end_to_end_params_CAPC_iq.json"; MODEL_NAME = "CAPC"
    
    # Step 1: Create the final data splits
    split_paths = create_final_experiment_splits()
    
    # Step 2: Pre-train the Champion Model using the recipe
    print("\n--- Step 2: Pre-training the Champion Model ---")
    for f in glob.glob(f"**/{MODEL_NAME}-champion*.ckpt", recursive=True): os.remove(f)
    pretrain_cmd = ["python", "self_supervised.py", "--params_file", RECIPE_FILE, "--pretrain_file_list", split_paths["pretrain"], "--epochs", "1"]
    run_command(pretrain_cmd)
    champion_ckpt_list = glob.glob(f"**/{MODEL_NAME}-champion*.ckpt", recursive=True)
    if not champion_ckpt_list: print("!!! ERROR: Champion checkpoint was not found. Exiting. !!!"); exit(1)
    champion_ckpt = champion_ckpt_list[0]
    print(f"\n--> Champion model found at: {champion_ckpt}")

    # Step 3: Run the final, rigorous evaluations
    print("\n--- Step 3: Running Downstream Evaluations ---")
    results, header = [], "Evaluation Method,Downstream Dataset,Shots,Accuracy"
    shots_to_test = [2, 4, 6, 8, 10]
    datasets_to_test = [("5GHz_Downstream", split_paths["downstream_5ghz"]), ("10GHz_Downstream", split_paths["downstream_10ghz"])]
    for eval_script, eval_method in [("supervised.py", "Linear_Eval"), ("finetune.py", "Fine_Tuning")]:
        for dataset_name, dataset_file in datasets_to_test:
            for shots in shots_to_test:
                print(f"\n-- Testing: {eval_method} on {dataset_name} with {shots} shots --")
                eval_cmd = ["python", eval_script, "--ssl_model_path", champion_ckpt, "--downstream_file_list", dataset_file, "--shots", str(shots), "--epochs", "1"]
                last_line = run_command(eval_cmd)
                accuracy = 0.0
                if "FINAL_ACCURACY:" in last_line:
                    accuracy_str = last_line.split(":")[-1].strip(); accuracy = float(accuracy_str)
                results.append(f"{eval_method},{dataset_name},{shots},{accuracy}")
    
    # Step 4: Print the final results table
    print("\n\n" + "="*80); print("               FINAL EXPERIMENT RESULTS               "); print("="*80); print("This table is in CSV format."); print("-" * 80); print(header)
    for row in results: print(row)
    print("="*80)

if __name__ == "__main__":
    main()