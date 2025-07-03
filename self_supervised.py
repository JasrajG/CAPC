# self_supervised.py
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from models import SSLModel
from dataset import data_loader

def _compute_stats_from_raw_data(data_dir):
    """Computes normalization stats directly from raw data files."""
    print("--- Computing normalization statistics from raw 'lab' data... ---")
    
    lab_csid_path = os.path.join(data_dir, 'dl.npy')
    lab_csiu_path = os.path.join(data_dir, 'ul.npy')

    if not (os.path.exists(lab_csid_path) and os.path.exists(lab_csiu_path)):
        raise FileNotFoundError(f"Could not find 'dl.npy' or 'ul.npy' in {data_dir} to compute stats.")

    csid_abs = np.abs(np.load(lab_csid_path))
    csiu_abs = np.abs(np.load(lab_csiu_path))
    
    global_min = min(np.min(csid_abs), np.min(csiu_abs))
    global_max = max(np.max(csid_abs), np.max(csiu_abs))
    
    print(f"--- Stats computed: min={global_min:.2f}, max={global_max:.2f} ---")
    return global_min, global_max

def main():
    parser = argparse.ArgumentParser(description="SSL Framework for WiFi Sensing")
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--recurrent_block', action='store_true')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='CAPC', 
        choices=['CAPC', 'SimCLR', 'BarlowTwins', 'AutoFi'],
        help="The SSL method to use for pre-training."
    )
    
    args = parser.parse_args()
    pl.seed_everything(42)

    global_min, global_max = _compute_stats_from_raw_data(args.database_path)

    base_model_cfg = {
        'embedding_size': 512, 'n_hidden_states_nodes': 2048, 'n_hidden_states_nodes_last_layer': 2048,
        'num_frames': 10, 'recurrent_block': args.recurrent_block,
        'global_min': global_min, 'global_max': global_max,
        'weight_decay': 1.5e-6, 'shared_weights': False,
    }

    if args.model_name == 'CAPC':
        model_cfg = base_model_cfg.copy()
        model_cfg.update({
            'name': 'CAPC', 'lambd': 0.002, 'cpc_coeff': 50.0, 'timestep': 9,
            'augmentations': {'dual_view': True, 'gaussian_noise': True}, 'losses': ['CPC', 'barlow_twin'],
        })
        dataset_mode = 'dual'
    else:
        dataset_mode = 'single'
        if args.model_name == 'BarlowTwins':
            model_cfg = base_model_cfg.copy()
            model_cfg.update({
                'name': 'BarlowTwins', 'lambd': 0.0051,
                'augmentations': {'gaussian_noise': True, 'time_mask': True}, 'losses': ['barlow_twin'],
            })
        elif args.model_name == 'SimCLR':
            model_cfg = base_model_cfg.copy()
            model_cfg.update({
                'name': 'SimCLR',
                'augmentations': {'time_mask': True}, 'losses': ['simclr'],
            })
        elif args.model_name == 'AutoFi':
            model_cfg = base_model_cfg.copy()
            model_cfg.update({
                'name': 'AutoFi',
                'augmentations': {'gaussian_noise': True, 'time_flip': True}, 'losses': ['autofi'],
            })
    
    dataset_cfg = {
        'root_dir': args.database_path, 'batch_size': args.batch_size, 'type': 'SignFi',
        'SignFi_env': 'lab', 'SignFi_link': 'all', 'SignFi_mode': dataset_mode,
        'global_min': global_min, 'global_max': global_max
    }
    hparams = {'dataset': dataset_cfg, 'model': model_cfg}

    _, _, _, unsupervised_loader = data_loader(dataset_cfg, 0)
    model = SSLModel(hparams)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss", mode="min", save_top_k=1, 
        filename=f'{args.model_name}-{{epoch}}',
        verbose=True
    )
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='auto', callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=unsupervised_loader, ckpt_path=args.resume_from_checkpoint)

    print(f"Training complete. Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()