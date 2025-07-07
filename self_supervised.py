# /Users/MAC/Projects/CAPC-replica/self_supervised.py

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from models import SSLModel
from dataset import TeraNovaDataset, DataLoader
import torch.multiprocessing

def main():
    parser = argparse.ArgumentParser(description="Faithful CAPC SSL Framework for TeraNova WiFi Sensing")
    
    parser.add_argument('--database_path', type=str, default='/Users/MAC/Downloads/5 GHz Bandwidth_Preprocessed', help="Path to the PREPROCESSED TeraNova data folder.")
    parser.add_argument('--epochs', type=int, default=300, help="The total number of epochs to train for.")
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--recurrent_block', action='store_true', default=True)
    
    # --- THIS IS THE KEY ARGUMENT FOR RESUMING ---
    # It tells the script which checkpoint file to load. Default is None (train from scratch).
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to a checkpoint file to resume training from.")
    
    parser.add_argument('--model_name', type=str, default='CAPC', choices=['CAPC'])
    
    args = parser.parse_args()
    pl.seed_everything(42)

    # --- Configuration for the model input ---
    SPECTROGRAM_HEIGHT = 129
    SPECTROGRAM_WIDTH = 184
    WINDOW_WIDTH = 23
    
    # --- Initialize the FAST dataset and dataloader ---
    unsupervised_dataset = TeraNovaDataset(root_dir=args.database_path)
    unsupervised_loader = DataLoader(
        unsupervised_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4 # Set to 0 for stability on macOS
    )
    
    # --- Define the smaller model configuration to fit in 8GB RAM ---
    base_model_cfg = {
        'embedding_size': 64,
        'n_hidden_states_nodes': 256,
        'n_hidden_states_nodes_last_layer': 256,
        'spec_shape': (1, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH),
        'window_width': WINDOW_WIDTH,
        'recurrent_block': args.recurrent_block,
        'weight_decay': 1.5e-6, 
        'shared_weights': True,
    }

    if args.model_name == 'CAPC':
        model_cfg = base_model_cfg.copy()
        model_cfg.update({
            'name': 'CAPC', 'lambd': 0.002, 'cpc_coeff': 50.0, 'timestep': 4,
            'augmentations': {'gaussian_noise': True, 'time_flip': True, 'time_mask': True},
            'losses': ['CPC', 'barlow_twin'],
        })
    else:
        raise NotImplementedError(f"Model '{args.model_name}' is not yet configured for this script.")
    
    hparams = {'dataset': {'name': 'TeraNova'}, 'model': model_cfg}

    # --- Initialize and Train the Model ---
    model = SSLModel(hparams)
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss", mode="min", save_top_k=1, 
        filename=f'{args.model_name}-TeraNova-small-{{epoch}}', verbose=True
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator='auto', callbacks=[checkpoint_callback]
    )
    
    print("\n" + "="*50)
    print(f"  STARTING SELF-SUPERVISED PRE-TRAINING WITH {args.model_name} (SMALL MODEL)")
    if args.resume_from_checkpoint:
        print(f"  Resuming from checkpoint: {args.resume_from_checkpoint}")
    print("="*50 + "\n")
    
    # --- The .fit() method uses the ckpt_path argument to resume ---
    trainer.fit(
        model, 
        train_dataloaders=unsupervised_loader, 
        ckpt_path=args.resume_from_checkpoint
    )

    print("\n" + "="*50)
    print("  TRAINING COMPLETE  ")
    print(f"  Best model saved at: {checkpoint_callback.best_model_path}")
    print("="*50 + "\n")

if __name__ == '__main__':
        # --- ADD THIS BLOCK TO FIX THE 'num_workers' HANG ---
    # This must be the first thing in the main guard.
    # It forces PyTorch to use a more stable method for creating background processes.
    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass
    # --- END OF ADDED BLOCK ---
    main()