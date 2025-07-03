# models.py
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import numpy as np
from modules import projector, RecurrentEncoder

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def nt_xent_loss(z1, z2, temperature=0.5):
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
    z = z.detach()
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()

class SSLModel(pl.LightningModule):
    def __init__(self, hparams):
        super(SSLModel, self).__init__()
        self.save_hyperparameters(hparams)
        if 'n_hidden_states_nodes_last_layer' not in self.hparams['model']:
            self.hparams['model']['n_hidden_states_nodes_last_layer'] = self.hparams['model']['n_hidden_states_nodes']
        self.encoder = RecurrentEncoder(
            input_type=self.hparams['dataset']['type'], num_frames=self.hparams['model']['num_frames'],
            embedding_size=self.hparams['model']['embedding_size'], recurrent_block=self.hparams['model']['recurrent_block']
        )
        self.encoder_2 = self.encoder if self.hparams['model']['shared_weights'] else RecurrentEncoder(
            input_type=self.hparams['dataset']['type'], num_frames=self.hparams['model']['num_frames'],
            embedding_size=self.hparams['model']['embedding_size'], recurrent_block=self.hparams['model']['recurrent_block']
        )
        model_name = self.hparams['model'].get('name', 'CAPC')
        projector_input_dim = self.hparams['model']['n_hidden_states_nodes_last_layer'] if model_name == 'CAPC' else self.hparams['model']['embedding_size']
        self.projector = projector(
            hidden_states=self.hparams['model']['n_hidden_states_nodes'],
            hidden_states_last_layer=self.hparams['model']['n_hidden_states_nodes_last_layer'],
            embedding_size=projector_input_dim
        )
        self.projector_2 = self.projector if self.hparams['model']['shared_weights'] else projector(
            hidden_states=self.hparams['model']['n_hidden_states_nodes'],
            hidden_states_last_layer=self.hparams['model']['n_hidden_states_nodes_last_layer'],
            embedding_size=projector_input_dim
        )
        self.bn = nn.BatchNorm1d(self.hparams['model']['n_hidden_states_nodes_last_layer'], affine=False)
        if model_name == 'CAPC':
            self.timestep = self.hparams['model']['timestep']
            self.autoregressive_model = nn.GRU(
                self.hparams['model']['embedding_size'], self.hparams['model']['n_hidden_states_nodes_last_layer'], batch_first=True
            )
            self.Wk = nn.ModuleList([nn.Linear(self.hparams['model']['n_hidden_states_nodes_last_layer'], self.hparams['model']['embedding_size']) for _ in range(self.timestep)])
            self.lsoftmax = nn.LogSoftmax(dim=1)

    def barlow_twin_loss(self, z1, z2, batch_size):
        c = self.bn(z1).T @ self.bn(z2) / batch_size
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.hparams['model']['lambd'] * off_diag

    def CPC(self, f, batch_size, t_samples):
        f = f.view(batch_size, self.encoder.sequence_length, f.shape[-1])
        encode_samples = torch.empty((self.timestep, batch_size, f.shape[-1]), dtype=f.dtype, device=f.device)
        for i in range(1, self.timestep + 1):
            encode_samples[i-1] = f[:, t_samples + i, :].view(batch_size, -1)
        forward_seq = f[:, :t_samples + 1, :]
        output, _ = self.autoregressive_model(forward_seq)
        c_t = output[:, t_samples, :].view(batch_size, -1)
        pred = torch.empty((self.timestep, batch_size, f.shape[-1]), dtype=f.dtype, device=f.device)
        for i in range(self.timestep):
            pred[i] = self.Wk[i](c_t)
        return encode_samples, pred, c_t

    def _get_views(self, batch):
        if len(batch) == 2: return batch[0], batch[0], batch[1]
        x1, x2, y = batch
        if self.hparams['model']['augmentations'].get('dual_view', False) and random.random() > 0.5:
            return x2, x1, y
        return x1, x2, y

    def training_step(self, batch, batch_idx):
        x1_raw, x2_raw, y = self._get_views(batch)
        x1, x2 = x1_raw.clone(), x2_raw.clone()

        if self.hparams['model']['augmentations'].get('gaussian_noise'):
            noise = torch.randn_like(x1) * 0.1
            x1.add_(noise); x2.add_(noise)
        if self.hparams['model']['augmentations'].get('time_flip'):
            if random.random() > 0.5: x1 = torch.flip(x1, dims=[3])
            if random.random() > 0.5: x2 = torch.flip(x2, dims=[3])
        if self.hparams['model']['augmentations'].get('time_mask'):
            mask_len = int(x1.shape[3] * 0.2)
            x1[:, :, :, random.randint(0, x1.shape[3] - mask_len):].mul_(0)
            x2[:, :, :, random.randint(0, x2.shape[3] - mask_len):].mul_(0)

        f1, _ = self.encoder(x1, view_mode='in_sequence')
        f2, _ = self.encoder_2(x2, view_mode='in_sequence')

        model_name = self.hparams['model']['name']
        if model_name == 'CAPC':
            t_samples = torch.randint(self.encoder.sequence_length - self.timestep, size=(1,)).long()
            e1, p1, c1 = self.CPC(f1, x1.size(0), t_samples)
            e2, p2, c2 = self.CPC(f2, x2.size(0), t_samples)
            nce1 = sum(torch.sum(torch.diag(self.lsoftmax(torch.mm(e1[i], p1[i].T)))) for i in range(self.timestep))
            nce2 = sum(torch.sum(torch.diag(self.lsoftmax(torch.mm(e2[i], p2[i].T)))) for i in range(self.timestep))
            loss_cpc = (nce1 + nce2) / (-1.0 * x1.size(0) * self.timestep)
            z1, z2 = self.projector(c1), self.projector_2(c2)
            loss_bt = self.barlow_twin_loss(z1, z2, x1.size(0))
            loss = self.hparams['model']['cpc_coeff'] * loss_cpc + loss_bt
        else:
            z1 = self.projector(f1.mean(dim=1))
            z2 = self.projector_2(f2.mean(dim=1))
            if model_name == 'BarlowTwins': loss = self.barlow_twin_loss(z1, z2, x1.size(0))
            elif model_name == 'SimCLR': loss = nt_xent_loss(z1, z2)
            elif model_name == 'AutoFi': loss = cosine_similarity_loss(z1, z2)
            else: raise ValueError(f"Unknown model name: {model_name}")

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        param_weights = [p for p in self.parameters() if p.ndim > 1]
        param_biases = [p for p in self.parameters() if p.ndim == 1]
        parameters = [{'params': param_weights}, {'params': param_biases}]
        optimizer = LARS(parameters, lr=0.2, weight_decay=self.hparams['model']['weight_decay'])
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
        flat_feature_size = self.encoder.embedding_size * self.encoder.sequence_length
        self.linear_separation = nn.Linear(flat_feature_size, self.hparams['dataset']['num_classes'])
        self.automatic_optimization = False
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams['dataset']['num_classes'])
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams['dataset']['num_classes'])
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams['dataset']['num_classes'])

    def forward(self, x):
        with torch.no_grad():
            x, _ = self.encoder(x, view_mode='flat')
        return self.linear_separation(x)

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
        self.log("test_loss", loss, on_epoch=True)

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
        return optimizer, scheduler

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