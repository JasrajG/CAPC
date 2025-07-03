# supervised.py
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from models import LinearClassifierModel
from dataset import data_loader
from modules import RecurrentEncoder

def main():
    parser = argparse.ArgumentParser(description="Supervised Linear Evaluation for CAPC")
    parser.add_argument('--database_path', type=str, required=True, help="Path to the root data folder.")
    parser.add_argument('--ssl_model_path', type=str, required=True, help="Path to the pre-trained SSL model .ckpt file.")
    parser.add_argument('--portion', type=int, default=10, help="The 'k' in k-shot fine-tuning.")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--recurrent_block', action='store_true')

    args = parser.parse_args()
    pl.seed_everything(42)

    print(f"--- Loading weights and stats from SSL checkpoint: {args.ssl_model_path} ---")
    checkpoint = torch.load(args.ssl_model_path, map_location=torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']
    
    global_min = hparams['model']['global_min']
    global_max = hparams['model']['global_max']
    embedding_size = hparams['model']['embedding_size']
    num_frames = hparams['model']['num_frames']
    
    print(f"--- Extracted stats from checkpoint: min={global_min:.2f}, max={global_max:.2f} ---")

    cfg = {
        'model': { 'epochs': args.epochs, 'lr': args.lr, 'weight_decay': 0 },
        'dataset': {
            'root_dir': args.database_path, 'batch_size': args.batch_size, 'type': 'SignFi',
            'SignFi_env': 'home', 'SignFi_link': 'all', 'SignFi_mode': 'single',
            'num_classes': 276, 'portion': args.portion,
            'global_min': global_min, 'global_max': global_max
        },
        'freeze_encoder': True, 'semi_supervised': False
    }

    train_loader, val_loader, test_loader, _ = data_loader(cfg['dataset'], args.num_workers)
    
    # Rebuild the encoder with the correct architecture from the checkpoint
    use_recurrent_block = hparams['model'].get('recurrent_block', False)
    encoder = RecurrentEncoder(
        input_type='SignFi', num_frames=num_frames, 
        embedding_size=embedding_size, recurrent_block=use_recurrent_block
    )
    encoder_state_dict = {k.replace('encoder.', '', 1): v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state_dict, strict=not use_recurrent_block) # Non-strict for CAPC->baseline eval mismatch
    
    model = LinearClassifierModel(pretrained_encoder=encoder, hparams=cfg)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_acc_epoch', mode='max', filename='best-finetune-acc', save_top_k=1, verbose=True)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', callbacks=[checkpoint_callback])
    
    print("\n--- Starting fine-tuning of the linear head on 'home' data ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("\n--- Starting testing with the best fine-tuned model on 'home' test set ---")
    trainer.test(dataloaders=test_loader, ckpt_path='best')

if __name__ == '__main__':
    main()