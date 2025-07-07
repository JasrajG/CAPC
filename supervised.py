# /Users/MAC/Projects/CAPC-replica/supervised.py

import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from dataset import EmbeddingDataset

class LinearModel(pl.LightningModule):
    def __init__(self, input_size, num_classes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(input_size, num_classes)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_accuracy(y_hat, y)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def main():
    parser = argparse.ArgumentParser(description="FAST Supervised Linear Evaluation with a dedicated Validation Set")
    parser.add_argument('--embedding_dir', type=str, default='/Users/MAC/Downloads/5 GHz Bandwidth_Embeddings', help="Path to the folder with embeddings.npy and labels.npy.")
    parser.add_argument('--shots', type=int, default=10, help="Number of samples per class for training (k-shot).")
    parser.add_argument('--val_samples', type=int, default=200, help="Number of samples per class for the validation set.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    pl.seed_everything(42)

    # --- Step 1: Load the pre-computed embeddings ---
    embeddings_path = os.path.join(args.embedding_dir, "embeddings.npy")
    labels_path = os.path.join(args.embedding_dir, "labels.npy")
    full_dataset = EmbeddingDataset(embeddings_path, labels_path)

    # --- Step 2: Create a Three-Way Data Split (Train, Validation, Test) ---
    print("\n--- Creating Train/Validation/Test Split ---")
    num_classes = len(np.unique(full_dataset.labels))
    
    train_indices = []
    val_indices = []
    
    for i in range(num_classes):
        # Find all indices for the current class
        class_indices = list(np.where(full_dataset.labels == i)[0])
        np.random.shuffle(class_indices) # Shuffle them to get random samples
        
        # Take the first 'k' for training
        train_indices.extend(class_indices[:args.shots])
        
        # Take the next 'N' for validation
        val_indices.extend(class_indices[args.shots : args.shots + args.val_samples])

    # The test set is everything else
    used_indices = set(train_indices + val_indices)
    all_indices = set(range(len(full_dataset)))
    test_indices = list(all_indices - used_indices)

    # Create the PyTorch Subset objects
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"Training set size: {len(train_dataset)} samples ({args.shots} shots per class)")
    print(f"Validation set size: {len(val_dataset)} samples")
    print(f"Test set size: {len(test_dataset)} samples")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=min(len(train_dataset), args.batch_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # --- Step 3: Train the Linear Model ---
    input_size = full_dataset.embeddings.shape[1]
    model = LinearModel(input_size, num_classes, args.lr)
    
    # This now correctly monitors the validation accuracy to save the best model
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', filename='best-linear-head')
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', callbacks=[checkpoint_callback])

    print("\n--- Starting LIGHTNING-FAST training of the linear head ---")
    # This now uses the proper validation set
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("\n--- Testing with best model chosen by validation set ---")
    # This runs the final test on the unseen test data
    test_results = trainer.test(dataloaders=test_loader, ckpt_path='best')

  

if __name__ == '__main__':
    main()