# tune.py (MODIFIED FOR 1D I/Q DATA)

import argparse
import json
import os
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split, Subset

# --- CHANGE 1: Import the correct dataset and model ---
from dataset import SignalDataset
from models import SSLModel

# In tune.py, replace the existing 'objective' function with this one.

def objective(trial, args, train_loader, val_loader):
    """
    The objective function for Optuna, now with WIDER hyperparameter search
    ranges for our initial exploratory run on I/Q data.
    """
    pl.seed_everything(42)

    # --- Step 1: Define a WIDER Hyperparameter Search Space ---
    model_name = args.model_name
    embedding_size = trial.suggest_categorical("embedding_size", [64, 128])
    hidden_nodes = trial.suggest_categorical("hidden_nodes", [256, 512])
    
    # --- HERE ARE THE KEY CHANGES: WIDER RANGES ---
    if model_name == 'CAPC':
        # Original cpc_coeff was (10.0, 100.0). Let's explore a much wider range.
        cpc_coeff = trial.suggest_float("cpc_coeff", 1.0, 200.0, log=True)
        # Original lambd was (1e-4, 1e-2). Let's expand by an order of magnitude on both ends.
        lambd = trial.suggest_float("lambd", 1e-5, 1e-1, log=True)
        
    elif model_name in ['BarlowTwins', 'AutoFi']:
        # Original lambd was (1e-4, 1e-2).
        lambd = trial.suggest_float("lambd", 1e-5, 1e-1, log=True)
        
    elif model_name == 'SimCLR':
        # Original temperature was (0.05, 1.0). Let's explore more extreme values.
        temperature = trial.suggest_float("temperature", 0.01, 2.0)

    # --- The rest of the function remains the same ---

    # Step 2: Build the 1D Model Config
    IQ_LENGTH = 40960
    WINDOW_WIDTH = 256
    
    model_cfg = {
        'name': model_name, 'embedding_size': embedding_size,
        'n_hidden_states_nodes': hidden_nodes, 'n_hidden_states_nodes_last_layer': hidden_nodes,
        'iq_shape': (2, IQ_LENGTH), 'window_width': WINDOW_WIDTH, 'weight_decay': 1.5e-6,
    }

    # Add model-specific params to the config
    if model_name == 'CAPC':
        model_cfg.update({'cpc_coeff': cpc_coeff, 'lambd': lambd, 'recurrent_block': True, 'shared_weights': False, 'timestep': 4})
    elif model_name in ['BarlowTwins', 'AutoFi']:
        model_cfg.update({'lambd': lambd, 'recurrent_block': False, 'shared_weights': True})
    elif model_name == 'SimCLR':
        model_cfg.update({'temperature': temperature, 'recurrent_block': False, 'shared_weights': True})

    hparams = {'dataset': {'name': 'TeraNova_IQ'}, 'model': model_cfg}
    
    # Step 3: Run Training
    model = SSLModel(hparams)
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator='auto',
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")],
        enable_progress_bar=True, logger=False
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Step 4: Return the Score
    best_score = trainer.callback_metrics.get("val_loss", float("inf"))
    return best_score if best_score is not None else float("inf")
    """
    The objective function for Optuna. Unchanged in its logic, but the model_cfg
    it builds is now for the 1D I/Q pipeline.
    """
    pl.seed_everything(42)

    # --- Step 1: Define the Hyperparameter Search Space (Unchanged) ---
    model_name = args.model_name
    embedding_size = trial.suggest_categorical("embedding_size", [64, 128])
    hidden_nodes = trial.suggest_categorical("hidden_nodes", [256, 512])
    
    if model_name == 'CAPC':
        cpc_coeff = trial.suggest_float("cpc_coeff", 10.0, 100.0, log=True)
        lambd = trial.suggest_float("lambd", 1e-4, 1e-2, log=True)
    elif model_name in ['BarlowTwins', 'AutoFi']:
        lambd = trial.suggest_float("lambd", 1e-4, 1e-2, log=True)
    elif model_name == 'SimCLR':
        temperature = trial.suggest_float("temperature", 0.05, 1.0)

    # --- Step 2: Build the 1D Model Config ---
    # --- CHANGE 2: Define 1D I/Q parameters, replacing 2D spectrogram params ---
    IQ_LENGTH = 40960      # As defined in our preprocess_data.py
    WINDOW_WIDTH = 256     # This is a new hyperparameter we might tune later
    
    model_cfg = {
        'name': model_name,
        'embedding_size': embedding_size,
        'n_hidden_states_nodes': hidden_nodes,
        'n_hidden_states_nodes_last_layer': hidden_nodes,
        'iq_shape': (2, IQ_LENGTH),  # Pass the 1D shape
        'window_width': WINDOW_WIDTH,
        'weight_decay': 1.5e-6,
    }

    # Add model-specific params to the config (Unchanged logic)
    if model_name == 'CAPC':
        model_cfg.update({'cpc_coeff': cpc_coeff, 'lambd': lambd, 'recurrent_block': True, 'shared_weights': False, 'timestep': 4})
    elif model_name in ['BarlowTwins', 'AutoFi']:
        model_cfg.update({'lambd': lambd, 'recurrent_block': False, 'shared_weights': True})
    elif model_name == 'SimCLR':
        model_cfg.update({'temperature': temperature, 'recurrent_block': False, 'shared_weights': True})

    hparams = {'dataset': {'name': 'TeraNova_IQ'}, 'model': model_cfg}
    
    # --- Step 3: Run Training (Unchanged logic) ---
    model = SSLModel(hparams)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")],
        enable_progress_bar=True,
        logger=False
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # --- Step 4: Return the Score (Unchanged logic) ---
    best_score = trainer.callback_metrics.get("val_loss", float("inf"))
    return best_score if best_score is not None else float("inf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for SSL Models on I/Q Data")
    parser.add_argument('--model_name', type=str, default='CAPC', choices=['CAPC', 'SimCLR', 'BarlowTwins', 'AutoFi'])
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--subset_fraction', type=float, default=0.1)
    args = parser.parse_args()

    # --- Data Loading Logic (Unchanged philosophy, updated class name) ---
    model_name = args.model_name
    if model_name == 'CAPC':
        aug_config = {'gaussian_noise': True, 'time_flip': True, 'time_mask': True}
    elif model_name == 'BarlowTwins':
        aug_config = {'gaussian_noise': True, 'time_mask': True}
    elif model_name == 'SimCLR':
        aug_config = {'time_mask': True}
    elif model_name == 'AutoFi':
        aug_config = {'gaussian_noise': True, 'time_flip': True}
    else:
        aug_config = {}

    pretrain_file_list = os.path.join('data_splits', 'pretrain_files.txt')
    # --- CHANGE 3: Use the new SignalDataset class ---
    dataset_for_train = SignalDataset(root_dir=None, file_list_path=pretrain_file_list, augmentations=aug_config)
    dataset_for_val = SignalDataset(root_dir=None, file_list_path=pretrain_file_list, augmentations=None)
    
    # --- The rest of the script is UNCHANGED ---
    # The logic for splitting data, creating subsets, and running the Optuna study
    # is perfectly preserved from the original workflow.
    val_split_ratio = 0.1
    train_size = int(len(dataset_for_train) * (1 - val_split_ratio))
    val_size = len(dataset_for_train) - train_size
    full_train_indices, val_indices = random_split(range(len(dataset_for_train)), [train_size, val_size])

    num_train_subset = int(len(full_train_indices) * args.subset_fraction)
    train_subset_indices = np.random.choice(full_train_indices.indices, num_train_subset, replace=False)
    
    train_ssl_dataset = Subset(dataset_for_train, train_subset_indices)
    val_ssl_dataset = Subset(dataset_for_val, val_indices)
    
    print(f"\n--- Hyperparameter Tuning Setup for I/Q Data ---")
    print(f"Model: {args.model_name}")
    print(f"Using a {args.subset_fraction*100:.0f}% subset for each trial: {len(train_ssl_dataset)} samples.")
    print(f"Validation set size: {len(val_ssl_dataset)} samples.")

    train_loader = DataLoader(train_ssl_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ssl_dataset, batch_size=32, num_workers=0)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args, train_loader, val_loader), n_trials=args.n_trials)

    best_params = study.best_trial.params
    best_params['model_name'] = args.model_name 
    output_filename = f"best_params_{args.model_name}_iq.json" # Append '_iq' to avoid overwriting old results
    with open(output_filename, 'w') as f:
        json.dump(best_params, f, indent=4)

    print("\n" + "="*50)
    print("      TUNING COMPLETE      ")
    print("="*50)
    print(f"The winning I/Q recipe has been saved to: {output_filename}")