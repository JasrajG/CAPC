# end_to_end_tune.py (FINAL AND DEFINITIVELY CORRECTED)

import argparse, json, os, numpy as np, pytorch_lightning as pl, torch, torchmetrics, sys, glob, optuna
from torch.utils.data import DataLoader, random_split, Subset
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from sklearn.model_selection import train_test_split
from models import SSLModel, LinearClassifierModel
from modules import IQEncoder
from dataset import SignalDataset

# --- THIS IS THE FIX ---
# Restoring the full, correct __init__ method for this class.
class FineTuningModel(pl.LightningModule):
    def __init__(self, encoder, input_size, num_classes, lr, lr_encoder):
        super().__init__()
        self.save_hyperparameters('input_size', 'num_classes', 'lr', 'lr_encoder')
        self.encoder = encoder
        self.linear_head = torch.nn.Linear(input_size, num_classes)
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    def forward(self, x):
        es, _ = self.encoder(x); ff = es.reshape(es.shape[0], -1); return self.linear_head(ff)
    def training_step(self, b, bi):
        x, y = b; yh = self(x); l = torch.nn.functional.cross_entropy(yh, y); self.log('train_loss', l); return l
    def validation_step(self, b, bi):
        x, y = b; yh = self(x); self.accuracy(yh, y); self.log('val_acc_epoch', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)
    def test_step(self, b, bi):
        x, y = b; yh = self(x); self.test_accuracy(yh, y); self.log('test_acc_epoch', self.test_accuracy, on_step=False, on_epoch=True)
    def configure_optimizers(self):
        return torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': self.hparams.lr_encoder}, {'params': self.linear_head.parameters(), 'lr': self.hparams.lr}])
# --- END OF FIX ---

def objective(trial, args, pretrain_file_list, downstream_file_list):
    print(f"\n{'='*80}\n--- Starting Trial {trial.number + 1} / {args.n_trials} ---\n{'='*80}")
    pl.seed_everything(42)
    model_name = args.model_name; shared_weights = True
    eval_method = trial.suggest_categorical("eval_method", ["linear_probe", "fine_tune"])
    window_width = trial.suggest_categorical("window_width", [32, 64, 128, 256, 512])
    if model_name == 'CAPC':
        max_timestep = (4096 // window_width) - 2
        timestep = trial.suggest_int("timestep", 2, max(2, max_timestep))
        cpc_coeff=trial.suggest_float("cpc_coeff",1.0,200.0,log=True); lambd=trial.suggest_float("lambd",1e-5,1e-1,log=True);
    elif model_name == 'SimCLR': temperature=trial.suggest_float("temperature",0.01,2.0)
    elif model_name == 'AutoFi':
        lambd = trial.suggest_float("lambd", 1e-5, 1e-1, log=True)
        beta_s = trial.suggest_float("beta_s", 0.1, 0.9)
        num_symbols = trial.suggest_categorical("num_symbols", [32, 64, 128])
        symbol_head_hidden_size = trial.suggest_categorical("symbol_head_hidden_size", [64, 128, 256])
    aug_config = {'gaussian_noise_prob':trial.suggest_float("noise_prob",0.0,0.8), 'time_flip_prob':trial.suggest_float("flip_prob",0.0,0.8), 'time_mask_prob':trial.suggest_float("mask_prob",0.0,0.8)}
    model_cfg = {'name':model_name, 'embedding_size':128, 'n_hidden_states_nodes':512, 'n_hidden_states_nodes_last_layer':256, 'iq_shape':(2,4096), 'window_width':window_width, 'weight_decay':1.5e-6, 'augmentations':aug_config, 'shared_weights':shared_weights}
    if model_name == 'CAPC': model_cfg['recurrent_block']=True; model_cfg.update({'cpc_coeff':cpc_coeff, 'lambd':lambd, 'timestep':timestep})
    else:
        model_cfg['recurrent_block']=False;
        if model_name == 'SimCLR': model_cfg.update({'temperature':temperature})
        elif model_name == 'AutoFi': model_cfg.update({'lambd': lambd, 'beta_s': beta_s, 'num_symbols': num_symbols, 'symbol_head_hidden_size': symbol_head_hidden_size})
    hparams_ssl = {'dataset': {'name': 'TeraNova_IQ_Pretrain'}, 'model': model_cfg}
    ssl_model = SSLModel(hparams_ssl)
    full_pretrain_dataset = SignalDataset(root_dir=None, file_list_path=pretrain_file_list, augmentations=aug_config)
    pretrain_subset_size = int(len(full_pretrain_dataset) * args.pretrain_subset_fraction)
    pretrain_indices = np.random.choice(range(len(full_pretrain_dataset)), pretrain_subset_size, replace=False)
    pretrain_dataset_for_trial = Subset(full_pretrain_dataset, pretrain_indices)
    print(f"Using a {args.pretrain_subset_fraction*100:.0f}% subset of pre-training data: {len(pretrain_dataset_for_trial)} samples.")
    pretrain_loader = DataLoader(pretrain_dataset_for_trial, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    pretrain_trainer = pl.Trainer(max_epochs=args.pretrain_epochs, accelerator='auto', enable_checkpointing=False, logger=False, callbacks=[TQDMProgressBar(refresh_rate=10)])
    pretrain_trainer.fit(ssl_model, train_dataloaders=pretrain_loader)
    encoder = ssl_model.encoder
    downstream_pool = SignalDataset(root_dir=None, file_list_path=downstream_file_list)
    num_classes = downstream_pool.num_classes
    train_indices, val_indices, test_indices = [], [], []
    labels_np = np.array(downstream_pool.labels)
    for i in range(num_classes):
        class_indices = np.where(labels_np == i)[0]; np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:args.shots]); val_indices.extend(class_indices[args.shots:args.shots+args.val_samples]); test_indices.extend(class_indices[args.shots+args.val_samples:args.shots+args.val_samples*2])
    train_dataset, val_dataset, test_dataset = Subset(downstream_pool, train_indices), Subset(downstream_pool, val_indices), Subset(downstream_pool, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=min(len(train_dataset), args.batch_size), shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, persistent_workers=True)
    if eval_method == "linear_probe":
        encoder.eval(); [p.requires_grad_(False) for p in encoder.parameters()]
        hparams_downstream = {'dataset':{'num_classes':num_classes}, 'model':{'epochs':args.linear_epochs, 'lr':1e-2}, 'freeze_encoder':True}
        eval_model = LinearClassifierModel(pretrained_encoder=encoder, hparams=hparams_downstream)
    else:
        lr_encoder = trial.suggest_float("lr_encoder", 1e-6, 1e-4, log=True)
        input_size = encoder.sequence_length * encoder.embedding_size
        eval_model = FineTuningModel(encoder, input_size, num_classes, lr=1e-3, lr_encoder=lr_encoder)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc_epoch', mode='max')
    eval_trainer = pl.Trainer(max_epochs=args.linear_epochs, accelerator='auto', callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)], logger=False, enable_model_summary=False)
    eval_trainer.fit(eval_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_results = eval_trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=False)
    final_accuracy = test_results[0].get('test_acc_epoch', 0.0) if test_results else 0.0
    return final_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Efficient End-to-End Tuning of SSL Models")
    parser.add_argument('--model_name', type=str, default='SimCLR', choices=['CAPC', 'SimCLR', 'BarlowTwins', 'AutoFi']); parser.add_argument('--n_trials', type=int, default=25); parser.add_argument('--pretrain_subset_fraction', type=float, default=0.2); parser.add_argument('--pretrain_epochs', type=int, default=3); parser.add_argument('--linear_epochs', type=int, default=30); parser.add_argument('--shots', type=int, default=5); parser.add_argument('--val_samples', type=int, default=20); parser.add_argument('--batch_size', type=int, default=16); args = parser.parse_args()
    pretrain_file_list, downstream_file_list = 'final_run_splits/pretrain_5ghz.txt', 'final_run_splits/downstream_10ghz_4k_pool.txt'
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args, pretrain_file_list, downstream_file_list), n_trials=args.n_trials)
    best_params = study.best_trial.params; best_params['model_name'] = args.model_name
    output_filename = f"best_end_to_end_params_{args.model_name}_iq_short.json"
    print("\n" + "="*50); print("      END-TO-END TUNING COMPLETE      "); print("="*50); print(f"Best downstream accuracy achieved: {study.best_value:.4f}"); print(f"The winning recipe has been saved to: {output_filename}");
    with open(output_filename, 'w') as f: json.dump(best_params, f, indent=4)
    print(json.dumps(best_params, indent=4))