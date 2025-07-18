# /Users/MAC/Projects/CAPC-replica/supervised.py (FINAL, CORRECTED VERSION)

import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from models import SpectrogramEncoder, LinearClassifierModel # We need the full models again
from dataset import TeraNovaDataset # We use the spectrogram dataset

def main():
    parser = argparse.ArgumentParser(description="Rigorous Supervised Linear Evaluation")
    
    # --- Arguments ---
    parser.add_argument('--ssl_model_path', type=str, required=True, help="Path to the pre-trained SSL model checkpoint.")
    parser.add_argument('--split_dir', type=str, default='data_splits', help="Directory with the downstream_files.txt list.")
    parser.add_argument('--data_path', type=str, default='/Users/MAC/Downloads/5 GHz Bandwidth_Preprocessed', help="Path to the root PREPROCESSED data folder.")
    
    parser.add_argument('--samples_per_class', type=int, default=1000, help="Number of samples per class for the downstream task pool.")
    parser.add_argument('--shots', type=int, default=10, help="Number of samples per class for training (k-shot).")
    parser.add_argument('--val_samples', type=int, default=200, help="Number of samples per class for the validation set.")
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128) # Smaller batch size for spectrograms
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    pl.seed_everything(42)

    # --- Step 1: Load the pre-trained encoder ---
    print(f"--- Loading encoder from SSL checkpoint: {args.ssl_model_path} ---")
    checkpoint = torch.load(args.ssl_model_path, map_location=torch.device('cpu'))
    hparams_ssl = checkpoint['hyper_parameters']['model']
    encoder = SpectrogramEncoder(
        spec_shape=hparams_ssl['spec_shape'],
        num_frames_per_window=hparams_ssl['window_width'],
        embedding_size=hparams_ssl['embedding_size'],
        recurrent_block=hparams_ssl.get('recurrent_block', False)
    )
    encoder_state_dict = {k.replace('encoder.', '', 1): v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state_dict)
    print("--- Encoder rebuilt and weights loaded successfully! ---")

    # --- Step 2: Load the downstream data pool ---
    downstream_file_list = os.path.join(args.split_dir, 'downstream_files.txt')
    full_downstream_pool = TeraNovaDataset(root_dir=args.data_path, file_list_path=downstream_file_list)
    
    # --- Step 3: Create the balanced 4000-sample subset ---
    print(f"\nCreating a balanced downstream pool with {args.samples_per_class} samples per class...")
    labels_np = np.array(full_downstream_pool.labels)
    num_classes = full_downstream_pool.num_classes
    pool_indices = []
    for i in range(num_classes):
        class_indices = np.where(labels_np == i)[0]
        num_to_sample = min(args.samples_per_class, len(class_indices))
        selected_indices = np.random.choice(class_indices, num_to_sample, replace=False)
        pool_indices.extend(selected_indices)

    downstream_dataset = Subset(full_downstream_pool, pool_indices)
    print(f"Total size of data pool for this experiment: {len(downstream_dataset)} samples.")

    # --- Step 4: Create the final train/val/test splits FROM THE POOL ---
    downstream_labels = np.array([downstream_dataset.dataset.labels[i] for i in downstream_dataset.indices])
    print("\n--- Creating Train/Validation/Test Split for Linear Evaluation ---")
    train_indices, val_indices = [], []
    for i in range(num_classes):
        class_indices_in_pool = np.where(downstream_labels == i)[0]
        np.random.shuffle(class_indices_in_pool)
        train_indices.extend(class_indices_in_pool[:args.shots])
        val_indices.extend(class_indices_in_pool[args.shots : args.shots + args.val_samples])

    used_indices = set(train_indices + val_indices)
    all_pool_indices = set(range(len(downstream_dataset)))
    test_indices = list(all_pool_indices - used_indices)
    
    train_dataset = Subset(downstream_dataset, train_indices)
    val_dataset = Subset(downstream_dataset, val_indices)
    test_dataset = Subset(downstream_dataset, test_indices)

    print(f"Training set size: {len(train_dataset)} samples ({args.shots} shots per class)")
    print(f"Validation set size: {len(val_dataset)} samples")
    print(f"Test set size: {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=min(len(train_dataset), args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # --- Step 5: Initialize and Train the Linear Classifier ---
    hparams_downstream = {
        'dataset': {'num_classes': num_classes},
        'model': {'epochs': args.epochs, 'lr': args.lr},
        'freeze_encoder': True # This is the key setting for linear evaluation
    }
    # This model now correctly freezes the encoder and trains a new head
    model = LinearClassifierModel(pretrained_encoder=encoder, hparams=hparams_downstream)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_acc_epoch', mode='max', filename='best-linear-eval')
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', callbacks=[checkpoint_callback])

    print("\n--- Starting SUPERVISED linear evaluation ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("\n--- Testing with best linear head ---")
    test_results = trainer.test(dataloaders=test_loader, ckpt_path='best')
    
    print("\n" + "#"*60)
    print("###            FINAL LINEAR EVALUATION RESULTS            ###")
    print("#"*60 + "\n")
    if test_results:
        final_accuracy = test_results[0].get('test_acc_epoch', 'Not found')
        if isinstance(final_accuracy, float):
            print(f"--> Final Test Accuracy: {final_accuracy:.4f}")
        else:
            print("--> Accuracy value was not found in the test results.")
    print("\n" + "#"*60)

if __name__ == '__main__':
    main()