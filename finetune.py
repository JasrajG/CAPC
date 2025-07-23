# finetune.py (FINAL, with HIGH-PERFORMANCE data loading)

import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader, Subset

from modules import IQEncoder
from dataset import SignalDataset

class FineTuningModel(pl.LightningModule):
    def __init__(self, encoder, input_size, num_classes, lr, lr_encoder):
        super().__init__()
        self.save_hyperparameters('input_size', 'num_classes', 'lr', 'lr_encoder')
        self.encoder = encoder
        self.linear_head = torch.nn.Linear(input_size, num_classes)
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    def forward(self, x):
        embeddings_seq, _ = self.encoder(x)
        flat_features = embeddings_seq.reshape(embeddings_seq.shape[0], -1)
        return self.linear_head(flat_features)
    def training_step(self, b, bi):
        x, y = b; yh = self(x); l = torch.nn.functional.cross_entropy(yh, y); self.log('train_loss', l); return l
    def validation_step(self, b, bi):
        x, y = b; yh = self(x); self.accuracy(yh, y); self.log('val_acc', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)
    def test_step(self, b, bi):
        x, y = b; yh = self(x); self.test_accuracy(yh, y); self.log('test_acc_epoch', self.test_accuracy, on_step=False, on_epoch=True)
    def configure_optimizers(self):
        return torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': self.hparams.lr_encoder}, {'params': self.linear_head.parameters(), 'lr': self.hparams.lr}])

def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning on I/Q Data")
    parser.add_argument('--ssl_model_path', type=str, required=True)
    parser.add_argument('--downstream_file_list', type=str, required=True)
    parser.add_argument('--shots', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--val_samples', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_encoder', type=float, default=1e-5)
    args = parser.parse_args()
    pl.seed_everything(42)

    checkpoint = torch.load(args.ssl_model_path, map_location=torch.device('cpu'))
    hparams_ssl = checkpoint['hyper_parameters']['model']
    
    encoder = IQEncoder(
        iq_shape=hparams_ssl['iq_shape'],
        num_samples_per_window=hparams_ssl['window_width'],
        embedding_size=hparams_ssl['embedding_size'],
        recurrent_block=hparams_ssl.get('recurrent_block', False)
    )
    encoder.load_state_dict({k.replace('encoder.', '', 1): v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder.')})
    
    downstream_dataset = SignalDataset(root_dir=None, file_list_path=args.downstream_file_list)
    downstream_labels = np.array(downstream_dataset.labels)
    num_classes = downstream_dataset.num_classes

    train_indices, val_indices = [], []
    for i in range(num_classes):
        class_indices_in_pool = np.where(downstream_labels == i)[0]
        np.random.shuffle(class_indices_in_pool)
        train_indices.extend(class_indices_in_pool[:args.shots])
        val_indices.extend(class_indices_in_pool[args.shots : args.shots + args.val_samples])

    test_indices = list(set(range(len(downstream_dataset))) - set(train_indices + val_indices))
    train_dataset, val_dataset, test_dataset = Subset(downstream_dataset, train_indices), Subset(downstream_dataset, val_indices), Subset(downstream_dataset, test_indices)

    print(f"\n--- Downstream Data ---\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # --- THIS IS THE OPTIMIZATION ---
    train_loader = DataLoader(train_dataset, batch_size=min(len(train_dataset), args.batch_size), shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, persistent_workers=True)
    # --- END OF OPTIMIZATION ---

    input_size = (hparams_ssl['iq_shape'][1] // hparams_ssl['window_width']) * hparams_ssl['embedding_size']
    model = FineTuningModel(encoder, input_size, num_classes, args.lr, args.lr_encoder)
    
    progress_bar = TQDMProgressBar(refresh_rate=10)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', filename='best-finetune-model-iq')
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', callbacks=[checkpoint_callback, progress_bar], logger=False, enable_model_summary=False)
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_results = trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=True)
    
    if test_results:
        final_accuracy = test_results[0].get('test_acc_epoch', 0.0)
        print(f"FINAL_ACCURACY: {final_accuracy}")

if __name__ == '__main__':
    main()