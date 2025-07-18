# /Users/MAC/Projects/CAPC-replica/self_supervised.py

import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split, Subset
from models import SSLModel
from dataset import TeraNovaDataset

def main():
    import json # We need this to load the parameters

    parser = argparse.ArgumentParser(description="Unified Self-Supervised Pre-training Framework")
    parser.add_argument('--params_file', type=str, default=None, help="[Optional] Path to a JSON file with model hyperparameters.")
    parser.add_argument('--model_name', type=str, default='CAPC', choices=['CAPC', 'SimCLR', 'BarlowTwins', 'AutoFi'])
    parser.add_argument('--split_dir', type=str, default='data_splits')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    args = parser.parse_args()
    pl.seed_everything(42)

    # --- Step 1: Define base defaults ---
    SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH, WINDOW_WIDTH = 129, 184, 23
    model_cfg = {
        'embedding_size': 64, 'n_hidden_states_nodes': 256, 'n_hidden_states_nodes_last_layer': 256,
        'spec_shape': (1, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH), 'window_width': WINDOW_WIDTH,
        'weight_decay': 1.5e-6,
    }

    # --- Step 2: Load parameters from file, if provided ---
    if args.params_file:
        print(f"\n--- Loading hyperparameters from file: {args.params_file} ---")
        with open(args.params_file, 'r') as f:
            tuned_params = json.load(f)
        # Update the defaults with the tuned values
        model_cfg.update(tuned_params)
        model_name = model_cfg.get('model_name', args.model_name)
    else:
        model_name = args.model_name

    # --- Step 3: Apply final model-specific architecture and augmentations ---
    print(f"\n--- Final configuration for model: {model_name} ---")
    model_cfg['name'] = model_name

    if model_name == 'CAPC':
        model_cfg['recurrent_block'] = True
        model_cfg['shared_weights'] = False
        # Set default CAPC loss params if not in the tuned file
        model_cfg.setdefault('lambd', 0.002)
        model_cfg.setdefault('cpc_coeff', 50.0)
        model_cfg.setdefault('timestep', 4)
        # CAPC uses all augmentations
        model_cfg['augmentations'] = {'gaussian_noise': True, 'time_flip': True, 'time_mask': True}
    else:
        model_cfg['recurrent_block'] = False
        model_cfg['shared_weights'] = True
        
        # Set augmentations based on the paper's ablation study
        if model_name == 'BarlowTwins':
            model_cfg['augmentations'] = {'gaussian_noise': True, 'time_mask': True}
        elif model_name == 'SimCLR':
            model_cfg['augmentations'] = {'time_mask': True}
        elif model_name == 'AutoFi':
            model_cfg['augmentations'] = {'gaussian_noise': True, 'time_flip': True}

    hparams = {'dataset': {'name': 'TeraNova'}, 'model': model_cfg}
    
    # --- This printout is your guarantee that the correct parameters are being used ---
    print("\n--- Final Model Hyperparameters Being Used ---")
    for key, value in model_cfg.items():
        print(f"  {key}: {value}")
    
    # --- Data loading and trainer setup (unchanged) ---
    pretrain_file_list = os.path.join(args.split_dir, 'pretrain_files.txt')
    dataset_for_train = TeraNovaDataset(root_dir=None, file_list_path=pretrain_file_list, augmentations=model_cfg['augmentations'])
    dataset_for_val = TeraNovaDataset(root_dir=None, file_list_path=pretrain_file_list, augmentations=None)

    val_split_ratio = 0.1
    train_size = int(len(dataset_for_train) * (1 - val_split_ratio))
    val_size = len(dataset_for_train) - train_size
    train_indices, val_indices = random_split(range(len(dataset_for_train)), [train_size, val_size])

    train_ssl_dataset = Subset(dataset_for_train, train_indices)
    val_ssl_dataset = Subset(dataset_for_val, val_indices)
    
    train_loader = DataLoader(train_ssl_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ssl_dataset, batch_size=args.batch_size, num_workers=0)

    model = SSLModel(hparams)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename=f'{model_name}-{{epoch}}', verbose=True)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', callbacks=[checkpoint_callback])
    
    print(f"\n--- STARTING PRE-TRAINING FOR {model_name} ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_from_checkpoint)
    print(f"\n--- PRE-TRAINING COMPLETE. Best model saved at: {checkpoint_callback.best_model_path} ---")

if __name__ == '__main__':
    main()