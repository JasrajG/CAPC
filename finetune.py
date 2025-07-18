# /Users/MAC/Projects/CAPC-replica/finetune.py

import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from models import SpectrogramEncoder
from dataset import TeraNovaDataset

class FineTuningModel(pl.LightningModule):
    def __init__(self, encoder, input_size, num_classes, lr, lr_encoder):
        super().__init__()
        self.save_hyperparameters('input_size', 'num_classes', 'lr', 'lr_encoder')
        self.encoder = encoder
        self.linear_head = torch.nn.Linear(input_size, num_classes)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        embeddings_seq, _ = self.encoder(x)
        flat_features = embeddings_seq.reshape(embeddings_seq.shape[0], -1)
        return self.linear_head(flat_features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_accuracy(y_hat, y)
        self.log('test_acc_epoch', self.test_accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.hparams.lr_encoder},
            {'params': self.linear_head.parameters(), 'lr': self.hparams.lr}
        ])
        return optimizer

def main():
    parser = argparse.ArgumentParser(description="Semi-Supervised Fine-Tuning")
    parser.add_argument('--ssl_model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True, help="Path to the PREPROCESSED data folder to use.")
    parser.add_argument('--samples_per_class', type=int, default=1000)
    parser.add_argument('--shots', type=int, default=10)
    parser.add_argument('--val_samples', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_encoder', type=float, default=1e-5)
    args = parser.parse_args()
    pl.seed_everything(42)

    # --- Step 1: Load Encoder ---
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

    # --- Step 2: Load Spectrogram Dataset ---
    print(f"\n--- Loading downstream data from directory: {args.data_path} ---")
    full_dataset_pool = TeraNovaDataset(root_dir=args.data_path, file_list_path=None)
    
    # --- Step 3: Create the Balanced Pool ---
    print(f"\nCreating a balanced downstream pool with {args.samples_per_class} samples per class...")
    labels_np = np.array(full_dataset_pool.labels)
    num_classes = full_dataset_pool.num_classes
    pool_indices = []
    for i in range(num_classes):
        class_indices = np.where(labels_np == i)[0]
        num_to_sample = min(args.samples_per_class, len(class_indices))
        if num_to_sample > 0:
            selected_indices = np.random.choice(class_indices, num_to_sample, replace=False)
            pool_indices.extend(selected_indices)

    downstream_dataset = Subset(full_dataset_pool, pool_indices)
    print(f"Total size of data pool for this experiment: {len(downstream_dataset)} samples.")

    # --- Step 4: Create Splits ---
    subset_labels = np.array([downstream_dataset.dataset.labels[i] for i in downstream_dataset.indices])
    print("\n--- Creating Train/Validation/Test Split for Fine-Tuning ---")
    train_indices, val_indices = [], []
    for i in range(num_classes):
        class_indices_in_pool = np.where(subset_labels == i)[0]
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

    # --- Step 5: Train the Model ---
    input_size = hparams_ssl['embedding_size'] * (hparams_ssl['spec_shape'][2] // hparams_ssl['window_width'])
    model = FineTuningModel(encoder, input_size, num_classes, args.lr, args.lr_encoder)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', filename='best-finetune-model')
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', callbacks=[checkpoint_callback])

    print("\n--- Starting SEMI-SUPERVISED fine-tuning ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("\n--- Testing with best fine-tuned model ---")
    test_results = trainer.test(dataloaders=test_loader, ckpt_path='best')
    
    # --- Final Results Banner ---
    print("\n" + "#"*60)
    print("###            FINAL FINE-TUNING RESULTS            ###")
    print("#"*60 + "\n")
    if test_results:
        final_accuracy = test_results[0].get('test_acc_epoch', 'Not found')
        if isinstance(final_accuracy, float):
            print(f"--> Final Test Accuracy: {final_accuracy:.4f}")
        else:
            print("--> Accuracy value was not found in the test results.")
    else:
        print("--> No test results were returned from the trainer.")
    print("\n" + "#"*60)

# --- THIS IS THE CRUCIAL BLOCK THAT WAS MISSING ---
if __name__ == '__main__':
    main()