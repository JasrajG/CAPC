import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
import numpy as np
import random
from modules import projector, SpectrogramEncoder
from dataset import Transform

# --- Helper loss functions ---
# These now require all hyperparameters to be passed in.

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def decorrelation_loss(z1, z2):
    batch_size = z1.shape[0]
    if batch_size <= 1:
        return 0.0
    bn = nn.BatchNorm1d(z1.shape[1], affine=False, device=z1.device)
    c = bn(z1).T @ bn(z2)
    c.div_(batch_size)
    return off_diagonal(c).pow_(2).sum()

def barlow_twin_loss(z1, z2, lambd):
    batch_size = z1.shape[0]
    if batch_size <= 1:
        return 0.0
    bn = nn.BatchNorm1d(z1.shape[1], affine=False, device=z1.device)
    c = bn(z1).T @ bn(z2)
    c.div_(batch_size)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    return on_diag + lambd * off_diag

def nt_xent_loss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    mask = torch.eye(batch_size * 2, dtype=torch.bool, device=z1.device)
    similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)
    positives = torch.cat([torch.diag(z1 @ z2.T), torch.diag(z2 @ z1.T)], dim=0)
    logits = torch.cat([positives.unsqueeze(1), similarity_matrix], dim=1)
    logits /= temperature
    target = torch.zeros(batch_size * 2, dtype=torch.long, device=z1.device)
    return F.cross_entropy(logits, target)

def cosine_similarity_loss(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z.detach(), dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()


class SSLModel(pl.LightningModule):
    # __init__ and other methods are the same as the last full version
    def __init__(self, hparams):
        super(SSLModel, self).__init__()
        self.save_hyperparameters(hparams)
        model_name = self.hparams['model']['name']

        self.encoder = SpectrogramEncoder(
            spec_shape=self.hparams['model']['spec_shape'],
            num_frames_per_window=self.hparams['model']['window_width'],
            embedding_size=self.hparams['model']['embedding_size'],
            recurrent_block=self.hparams['model'].get('recurrent_block', False)
        )
        self.encoder_2 = self.encoder if self.hparams['model'].get('shared_weights', True) else SpectrogramEncoder(
            spec_shape=self.hparams['model']['spec_shape'],
            num_frames_per_window=self.hparams['model']['window_width'],
            embedding_size=self.hparams['model']['embedding_size'],
            recurrent_block=self.hparams['model'].get('recurrent_block', False)
        )
        
        if model_name == 'CAPC':
            self.Wk = nn.ModuleList([nn.Linear(self.hparams['model']['embedding_size'], self.hparams['model']['embedding_size']) for _ in range(self.hparams['model']['timestep'])])
            self.lsoftmax = nn.LogSoftmax(dim=1)
            self.projector = projector(
                hidden_states=self.hparams['model']['n_hidden_states_nodes'],
                hidden_states_last_layer=self.hparams['model']['n_hidden_states_nodes_last_layer'],
                embedding_size=self.hparams['model']['embedding_size']
            )
        else:
            flat_feature_size = self.hparams['model']['embedding_size'] * (self.hparams['model']['spec_shape'][2] // self.hparams['model']['window_width'])
            self.projector = projector(
                hidden_states=self.hparams['model']['n_hidden_states_nodes'],
                hidden_states_last_layer=self.hparams['model']['n_hidden_states_nodes_last_layer'],
                embedding_size=flat_feature_size
            )
        
        self.val_transform = Transform(self.hparams['model'].get('augmentations', {}))

    def CPC(self, f, t_samples):
        batch_size = f.shape[0]
        c_t = f[:, t_samples, :].view(batch_size, -1)
        nce_loss = 0.0
        for i in range(self.hparams['model']['timestep']):
            if t_samples + i + 1 >= self.encoder.sequence_length: continue
            pred_k = self.Wk[i](c_t)
            true_k = f[:, t_samples + i + 1, :]
            logits = torch.mm(pred_k, true_k.T)
            nce_loss += torch.sum(torch.diag(self.lsoftmax(logits)))
        nce_loss /= (-1.0 * batch_size * self.hparams['model']['timestep'])
        return nce_loss, c_t
    
    def forward_projector(self, x):
        f, _ = self.encoder(x)
        f_flat = f.reshape(f.shape[0], -1)
        return self.projector(f_flat)

    def training_step(self, batch, batch_idx):
        (x1, x2), y = batch
        model_name = self.hparams['model']['name']

        if model_name == 'CAPC':
            f1, _ = self.encoder(x1)
            f2, _ = self.encoder_2(x2)
            t_samples = torch.randint(self.encoder.sequence_length - self.hparams['model']['timestep'] - 1, size=(1,)).long().item()
            loss_cpc1, c1 = self.CPC(f1, t_samples)
            loss_cpc2, c2 = self.CPC(f2, t_samples)
            loss_cpc = (loss_cpc1 + loss_cpc2) / 2
            z1_bt = self.projector(c1)
            z2_bt = self.projector(c2)
            # Correctly use the tuned lambda
            loss_bt = barlow_twin_loss(z1_bt, z2_bt, self.hparams['model']['lambd'])
            loss = self.hparams['model']['cpc_coeff'] * loss_cpc + loss_bt
        else:
            z1 = self.forward_projector(x1)
            z2 = self.forward_projector(x2)
            if model_name == 'SimCLR': 
                # Correctly use the tuned temperature
                loss = nt_xent_loss(z1, z2, self.hparams['model'].get('temperature', 0.5))
            elif model_name == 'BarlowTwins': 
                # Correctly use the tuned lambda
                loss = barlow_twin_loss(z1, z2, self.hparams['model'].get('lambd', 0.0051))
            elif model_name == 'AutoFi':
                loss_cosine = cosine_similarity_loss(z1, z2)
                loss_decorr = decorrelation_loss(z1, z2)
                # Correctly use the tuned lambda for the decorrelation part
                loss = loss_cosine + self.hparams['model'].get('lambd', 0.0051) * loss_decorr
            else: 
                raise ValueError(f"Unknown model name: {model_name}")
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_clean, y = batch
        model_name = self.hparams['model']['name']
        x_augmented = self.val_transform(x_clean.clone())

        if model_name == 'CAPC':
            f1, _ = self.encoder(x_clean)
            f2, _ = self.encoder_2(x_augmented)
            t_samples = torch.randint(self.encoder.sequence_length - self.hparams['model']['timestep'] - 1, size=(1,)).long().item()
            loss_cpc1, c1 = self.CPC(f1, t_samples)
            loss_cpc2, c2 = self.CPC(f2, t_samples)
            loss_cpc = (loss_cpc1 + loss_cpc2) / 2
            z1_bt = self.projector(c1)
            z2_bt = self.projector(c2)
            loss_bt = barlow_twin_loss(z1_bt, z2_bt, self.hparams['model']['lambd'])
            loss = self.hparams['model']['cpc_coeff'] * loss_cpc + loss_bt
        else:
            z_clean = self.forward_projector(x_clean)
            z_augmented = self.forward_projector(x_augmented)
            if model_name == 'SimCLR': 
                loss = nt_xent_loss(z_clean, z_augmented, self.hparams['model'].get('temperature', 0.5))
            elif model_name == 'BarlowTwins': 
                loss = barlow_twin_loss(z_clean, z_augmented, self.hparams['model'].get('lambd', 0.0051))
            elif model_name == 'AutoFi':
                loss_cosine = cosine_similarity_loss(z_clean, z_augmented)
                loss_decorr = decorrelation_loss(z_clean, z_augmented)
                loss = loss_cosine + self.hparams['model'].get('lambd', 0.0051) * loss_decorr
            else: 
                raise ValueError(f"Unknown model name: {model_name}")

        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        param_weights = [p for p in self.parameters() if p.ndim > 1]
        param_biases = [p for p in self.parameters() if p.ndim == 1]
        parameters = [{'params': param_weights}, {'params': param_biases}]
        optimizer = LARS(parameters, lr=0.2, weight_decay=self.hparams['model'].get('weight_decay', 1.5e-6))
        return optimizer


class LinearClassifierModel(pl.LightningModule):
    def __init__(self, pretrained_encoder, hparams):
        super(LinearClassifierModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.encoder = pretrained_encoder
        if self.hparams.get('freeze_encoder', True):
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        flat_feature_size = self.encoder.sequence_length * self.encoder.embedding_size
        self.linear_separation = nn.Linear(flat_feature_size, self.hparams['dataset']['num_classes'])
        self.automatic_optimization = False
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams['dataset']['num_classes'])
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams['dataset']['num_classes'])
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams['dataset']['num_classes'])

    def forward(self, x):
        with torch.no_grad():
            sequence_embedding, _ = self.encoder(x)
        batch_size = sequence_embedding.shape[0]
        flat_features = sequence_embedding.reshape(batch_size, -1)
        return self.linear_separation(flat_features)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return F.cross_entropy(y_hat, y.squeeze()), y_hat, y.squeeze()

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.train_accuracy.update(y_hat, y)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.val_accuracy.update(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.test_accuracy.update(y_hat, y)
        self.log("test_acc_epoch", loss, on_epoch=True)

    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.train_accuracy.compute(), on_step=False, on_epoch=True)
        self.train_accuracy.reset()
        sch = self.lr_schedulers()
        if sch: sch.step()

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.val_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.val_accuracy.reset()

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.test_accuracy.compute(), on_step=False, on_epoch=True)
        self.test_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.linear_separation.parameters(), lr=self.hparams['model']['lr'], weight_decay=self.hparams['model'].get('weight_decay', 0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams['model']['epochs'])
        return [optimizer], [scheduler]


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001, weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta, weight_decay_filter=weight_decay_filter, lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)
    def exclude_bias_and_norm(self, p):
        return p.ndim == 1
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad
                if dp is None: continue
                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp.add_(p, alpha=g['weight_decay'])
                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p); update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0., torch.where(update_norm > 0, (g['eta'] * param_norm / update_norm), one), one)
                    dp.mul_(q)
                param_state = self.state[p]
                if 'mu' not in param_state: param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])