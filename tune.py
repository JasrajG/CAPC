# /Users/MAC/Projects/CAPC-replica/tune.py

import argparse
import json
import os
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split, Subset

from dataset import TeraNovaDataset
from models import SSLModel

def objective(trial, args, train_loader, val_loader):
    """
    The objective function for Optuna. Takes a trial, trains a model with the
    suggested hyperparameters, and returns the validation loss.
    """
    pl.seed_everything(42)

    # --- Step 1: Define the Hyperparameter Search Space ---
    model_name = args.model_name
    
    # These are common to all models in this tuning run
    embedding_size = trial.suggest_categorical("embedding_size", [64, 128])
    hidden_nodes = trial.suggest_categorical("hidden_nodes", [256, 512])
    
    # --- THIS IS THE KEY CHANGE: Model-specific hyperparameters ---
    if model_name == 'CAPC':
        cpc_coeff = trial.suggest_float("cpc_coeff", 10.0, 100.0, log=True)
        lambd = trial.suggest_float("lambd", 1e-4, 1e-2, log=True)
    elif model_name in ['BarlowTwins', 'AutoFi']:
        lambd = trial.suggest_float("lambd", 1e-4, 1e-2, log=True)
    elif model_name == 'SimCLR':
        # We now correctly define the search space for temperature
        temperature = trial.suggest_float("temperature", 0.05, 1.0)

    # --- Step 2: Build the Model Config ---
    SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH, WINDOW_WIDTH = 129, 184, 23
    model_cfg = {
        'name': model_name,
        'embedding_size': embedding_size, 'n_hidden_states_nodes': hidden_nodes,
        'n_hidden_states_nodes_last_layer': hidden_nodes,
        'spec_shape': (1, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH), 'window_width': WINDOW_WIDTH,
        'weight_decay': 1.5e-6,
    }

    # Add model-specific params to the config
    if model_name == 'CAPC':
        model_cfg.update({'cpc_coeff': cpc_coeff, 'lambd': lambd, 'recurrent_block': True, 'shared_weights': False, 'timestep': 4})
    elif model_name in ['BarlowTwins', 'AutoFi']:
        model_cfg.update({'lambd': lambd, 'recurrent_block': False, 'shared_weights': True})
    elif model_name == 'SimCLR':
        model_cfg.update({'temperature': temperature, 'recurrent_block': False, 'shared_weights': True})

    hparams = {'dataset': {'name': 'TeraNova'}, 'model': model_cfg}
    
    # --- Step 3: Run Training ---
    model = SSLModel(hparams)
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator='auto',
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")],
        enable_progress_bar=True, logger=False
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # --- Step 4: Return the Score ---
    best_score = trainer.callback_metrics.get("val_loss", float("inf"))
    return best_score if best_score is not None else float("inf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for SSL Models")
    parser.add_argument('--model_name', type=str, default='CAPC', choices=['CAPC', 'SimCLR', 'BarlowTwins', 'AutoFi'])
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--subset_fraction', type=float, default=0.1)
    args = parser.parse_args()

    # --- THIS IS THE CORRECTED DATA LOADING LOGIC ---
    
    # First, determine the correct augmentation config based on the model name
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

    # Pre-load data, passing the CORRECT augmentation dictionary
    pretrain_file_list = os.path.join('data_splits', 'pretrain_files.txt')
    dataset_for_train = TeraNovaDataset(root_dir=None, file_list_path=pretrain_file_list, augmentations=aug_config)
    dataset_for_val = TeraNovaDataset(root_dir=None, file_list_path=pretrain_file_list, augmentations=None)
    
    # --- The rest of the script is unchanged ---
    val_split_ratio = 0.1
    train_size = int(len(dataset_for_train) * (1 - val_split_ratio))
    val_size = len(dataset_for_train) - train_size
    full_train_indices, val_indices = random_split(range(len(dataset_for_train)), [train_size, val_size])

    num_train_subset = int(len(full_train_indices) * args.subset_fraction)
    train_subset_indices = np.random.choice(full_train_indices.indices, num_train_subset, replace=False)
    
    train_ssl_dataset = Subset(dataset_for_train, train_subset_indices)
    val_ssl_dataset = Subset(dataset_for_val, val_indices)
    
    print(f"\n--- Hyperparameter Tuning Setup ---")
    print(f"Using a {args.subset_fraction*100:.0f}% subset for each trial: {len(train_ssl_dataset)} samples.")
    print(f"Validation set size: {len(val_ssl_dataset)} samples.")

    train_loader = DataLoader(train_ssl_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ssl_dataset, batch_size=32, num_workers=0)

    # --- Start the Optuna Study ---
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args, train_loader, val_loader), n_trials=args.n_trials)

    # --- Save the best parameters ---
    best_params = study.best_trial.params
    best_params['model_name'] = args.model_name 
    output_filename = f"best_params_{args.model_name}.json"
    with open(output_filename, 'w') as f:
        json.dump(best_params, f, indent=4)

    print("\n" + "="*50)
    print("      TUNING COMPLETE      ")
    print("="*50)
    print(f"The winning recipe has been saved to: {output_filename}")