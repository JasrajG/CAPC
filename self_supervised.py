# self_supervised.py (FINAL AND DEFINITIVELY CORRECTED)

import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader, random_split, Subset
import json

from models import SSLModel
from dataset import SignalDataset

def main():
    parser = argparse.ArgumentParser(description="Unified Self-Supervised Pre-training for I/Q Data")
    parser.add_argument('--params_file', type=str, required=True)
    parser.add_argument('--pretrain_file_list', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    args = parser.parse_args()
    pl.seed_everything(42)

    IQ_LENGTH, WINDOW_WIDTH = 4096, 256
    
    model_cfg = {
        'embedding_size': 128, 'n_hidden_states_nodes': 512, 'n_hidden_states_nodes_last_layer': 256,
        'iq_shape': (2, IQ_LENGTH), 'window_width': WINDOW_WIDTH, 'weight_decay': 1.5e-6,
    }

    with open(args.params_file, 'r') as f:
        tuned_params = json.load(f)
    model_cfg.update(tuned_params)
    model_name = model_cfg.get('model_name', "Unknown")
    model_cfg['name'] = model_name
    
    if model_name == 'AutoFi':
        model_cfg.setdefault('num_symbols', 64)
        model_cfg.setdefault('symbol_head_hidden_size', 128)
    
    aug_config = {
        'gaussian_noise_prob': tuned_params.get('noise_prob', 0.0),
        'time_flip_prob': tuned_params.get('flip_prob', 0.0),
        'time_mask_prob': tuned_params.get('mask_prob', 0.0)
    }
    model_cfg['augmentations'] = aug_config

    hparams = {'dataset': {'name': 'TeraNova_IQ_Final'}, 'model': model_cfg}
    
    print("\n--- Final Model Hyperparameters Being Used ---")
    for key, value in hparams['model'].items():
        if key == 'augmentations':
            print(f"  {key}:")
            for aug_key, aug_val in value.items(): print(f"    {aug_key}: {aug_val}")
        else: print(f"  {key}: {value}")
    
    dataset_for_train = SignalDataset(root_dir=None, file_list_path=args.pretrain_file_list, augmentations=model_cfg['augmentations'])
    dataset_for_val = SignalDataset(root_dir=None, file_list_path=args.pretrain_file_list, augmentations=None)
    
    val_split_ratio = 0.1
    train_size = int(len(dataset_for_train) * (1 - val_split_ratio)); val_size = len(dataset_for_train) - train_size
    train_indices, val_indices = random_split(range(len(dataset_for_train)), [train_size, val_size])
    train_ssl_dataset, val_ssl_dataset = Subset(dataset_for_train, train_indices), Subset(dataset_for_val, val_indices)
    
    print(f"\n--- Data Loading Summary ---")
    print(f"Loaded {len(dataset_for_train)} total file paths from '{args.pretrain_file_list}'.")
    print(f"--> Training samples:   {len(train_ssl_dataset)}")
    print(f"--> Validation samples: {len(val_ssl_dataset)}")
    
    train_loader = DataLoader(train_ssl_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    
    # --- THIS IS THE FIX ---
    val_loader = DataLoader(val_ssl_dataset, batch_size=args.batch_size, num_workers=4, persistent_workers=True, drop_last=True)
    # --- END OF FIX ---
    
    model = SSLModel(hparams)
    
    progress_bar = TQDMProgressBar(refresh_rate=10)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename=f'{model_name}-champion', verbose=True)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', callbacks=[checkpoint_callback, progress_bar])
    
    print(f"\n--- STARTING FINAL PRE-TRAINING FOR {model_name} ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume_from_checkpoint)
    print(f"\n--- PRE-TRAINING COMPLETE. Best model saved at: {checkpoint_callback.best_model_path} ---")

if __name__ == '__main__':
    main()