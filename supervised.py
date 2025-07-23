# supervised.py (FINAL, with HIGH-PERFORMANCE data loading)
import argparse, os, torch, numpy as np, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from models import LinearClassifierModel
from modules import IQEncoder
from dataset import SignalDataset
from torch.utils.data import DataLoader, Subset

def main():
    parser = argparse.ArgumentParser(description="Linear Evaluation")
    parser.add_argument('--ssl_model_path',type=str,required=True); parser.add_argument('--downstream_file_list',type=str,required=True); parser.add_argument('--shots',type=int,required=True); parser.add_argument('--epochs',type=int,required=True); parser.add_argument('--samples_per_class',type=int,default=1000); parser.add_argument('--val_samples',type=int,default=200); parser.add_argument('--batch_size',type=int,default=128); parser.add_argument('--lr',type=float,default=1e-2); args = parser.parse_args(); pl.seed_everything(42)
    checkpoint = torch.load(args.ssl_model_path, map_location=torch.device('cpu')); hparams_ssl = checkpoint['hyper_parameters']['model']
    encoder = IQEncoder(iq_shape=hparams_ssl['iq_shape'], num_samples_per_window=hparams_ssl['window_width'], embedding_size=hparams_ssl['embedding_size'], recurrent_block=hparams_ssl.get('recurrent_block', False))
    encoder.load_state_dict({k.replace('encoder.', '', 1): v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder.')})
    downstream_dataset = SignalDataset(root_dir=None, file_list_path=args.downstream_file_list); downstream_labels = np.array(downstream_dataset.labels); num_classes = downstream_dataset.num_classes
    train_indices, val_indices = [], []
    for i in range(num_classes):
        class_indices_in_pool = np.where(downstream_labels == i)[0]; np.random.shuffle(class_indices_in_pool)
        train_indices.extend(class_indices_in_pool[:args.shots]); val_indices.extend(class_indices_in_pool[args.shots : args.shots + args.val_samples])
    test_indices = list(set(range(len(downstream_dataset))) - set(train_indices + val_indices))
    train_dataset, val_dataset, test_dataset = Subset(downstream_dataset, train_indices), Subset(downstream_dataset, val_indices), Subset(downstream_dataset, test_indices)
    print(f"\n--- Downstream Data ---\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # --- THIS IS THE FIX ---
    train_loader = DataLoader(train_dataset, batch_size=min(len(train_dataset), args.batch_size), shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, persistent_workers=True)
    # --- END OF FIX ---

    hparams_downstream = {'dataset': {'num_classes': num_classes}, 'model': {'epochs': args.epochs, 'lr': args.lr}, 'freeze_encoder': True}
    model = LinearClassifierModel(pretrained_encoder=encoder, hparams=hparams_downstream)
    progress_bar = TQDMProgressBar(refresh_rate=10); checkpoint_callback = ModelCheckpoint(monitor='val_acc_epoch', mode='max', filename='best-linear-eval-iq')
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', callbacks=[checkpoint_callback, progress_bar], logger=False, enable_model_summary=False)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_results = trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=True)
    if test_results: print(f"FINAL_ACCURACY: {test_results[0].get('test_acc_epoch', 0.0)}")

if __name__ == '__main__': main()