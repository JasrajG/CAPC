# models.py (UPGRADED for full AutoFi implementation)

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, pytorch_lightning as pl
# --- CHANGE 1: Import our new modules ---
from modules import IQEncoder, Projector, SymbolPredictionHead
from dataset import Transform
import torchmetrics

# --- Loss functions are unchanged, but we add the new Symbol-level loss ---
def off_diagonal(x): n, m = x.shape; assert n == m; return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
def decorrelation_loss(z1, z2):
    bn = nn.BatchNorm1d(z1.shape[1], affine=False, device=z1.device); c = bn(z1).T @ bn(z2); c.div_(z1.shape[0]); return off_diagonal(c).pow_(2).sum()
def barlow_twin_loss(z1, z2, lambd):
    bn = nn.BatchNorm1d(z1.shape[1], affine=False, device=z1.device); c = bn(z1).T @ bn(z2); c.div_(z1.shape[0]); on_diag = torch.diagonal(c).add_(-1).pow_(2).sum(); off_diag = off_diagonal(c).pow_(2).sum(); return on_diag + lambd * off_diag
def nt_xent_loss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1); representations = torch.cat([z1, z2], dim=0); similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2); mask = torch.eye(z1.size(0)*2, dtype=torch.bool, device=z1.device); similarity_matrix = similarity_matrix[~mask].view(z1.size(0)*2, -1); positives = torch.cat([torch.diag(z1 @ z2.T), torch.diag(z2 @ z1.T)], dim=0); logits = torch.cat([positives.unsqueeze(1), similarity_matrix], dim=1); logits /= temperature; target = torch.zeros(z1.size(0)*2, dtype=torch.long, device=z1.device); return F.cross_entropy(logits, target)
def cosine_similarity_loss(p, z): return 2 - 2 * (F.normalize(p, dim=1) * F.normalize(z.detach(), dim=1)).sum(dim=1).mean()
# --- CHANGE 2: The new Symbol-level loss from the AutoFi paper (Equation 5) ---
def symbol_level_loss(s1, s2):
    # s1 and s2 are the "soft labels" (probability distributions) from the symbol head.
    # The loss is a symmetric cross-entropy.
    loss = - (s1 * torch.log(s2.clamp(min=1e-8))).sum(dim=1) - (s2 * torch.log(s1.clamp(min=1e-8))).sum(dim=1)
    return loss.mean()

class SSLModel(pl.LightningModule):
    def __init__(self, hparams):
        super(SSLModel, self).__init__(); self.save_hyperparameters(hparams); model_name = self.hparams['model']['name']
        self.encoder = IQEncoder(iq_shape=self.hparams['model']['iq_shape'], num_samples_per_window=self.hparams['model']['window_width'], embedding_size=self.hparams['model']['embedding_size'], recurrent_block=self.hparams['model'].get('recurrent_block', False))
        self.encoder_2 = self.encoder if self.hparams['model'].get('shared_weights', True) else IQEncoder(iq_shape=self.hparams['model']['iq_shape'], num_samples_per_window=self.hparams['model']['window_width'], embedding_size=self.hparams['model']['embedding_size'], recurrent_block=self.hparams['model'].get('recurrent_block', False))
        
        # --- CHANGE 3: Update the model initialization logic ---
        if model_name == 'CAPC':
            self.Wk = nn.ModuleList([nn.Linear(self.hparams['model']['embedding_size'], self.hparams['model']['embedding_size']) for _ in range(self.hparams['model']['timestep'])])
            self.lsoftmax = nn.LogSoftmax(dim=1)
            self.projector = Projector(input_size=self.hparams['model']['embedding_size'], hidden_size=self.hparams['model']['n_hidden_states_nodes'], output_size=self.hparams['model']['n_hidden_states_nodes_last_layer'])
        else: # SimCLR, BarlowTwins, AutoFi
            sequence_length = self.hparams['model']['iq_shape'][1] // self.hparams['model']['window_width']
            flat_feature_size = self.hparams['model']['embedding_size'] * sequence_length
            self.projector = Projector(input_size=flat_feature_size, hidden_size=self.hparams['model']['n_hidden_states_nodes'], output_size=self.hparams['model']['n_hidden_states_nodes_last_layer'])
            
            # --- CHANGE 4: Create the Symbol Prediction Head ONLY for AutoFi ---
            if model_name == 'AutoFi':
                self.symbol_head = SymbolPredictionHead(input_size=self.hparams['model']['n_hidden_states_nodes_last_layer'], hidden_size=self.hparams['model']['symbol_head_hidden_size'], num_symbols=self.hparams['model']['num_symbols'])
        
        self.val_transform = Transform(self.hparams['model'].get('augmentations', {}))

    def CPC(self, f, t_samples): # Unchanged
        # ... (rest of CPC function is the same) ...
        batch_size=f.shape[0]; c_t=f[:,t_samples,:].view(batch_size,-1); nce=0.0;
        if t_samples+self.hparams['model']['timestep']>=self.encoder.sequence_length: t_samples=self.encoder.sequence_length-self.hparams['model']['timestep']-1
        for i in range(self.hparams['model']['timestep']): pred_k=self.Wk[i](c_t); true_k=f[:,t_samples+i+1,:]; logits=torch.mm(pred_k,true_k.T); nce+=torch.sum(torch.diag(self.lsoftmax(logits)))
        nce/=(-1.0*batch_size*self.hparams['model']['timestep']); return nce,c_t

    def forward_projector(self, x):
        f, _ = self.encoder(x); f_flat = f.reshape(f.shape[0], -1); return self.projector(f_flat)

    def training_step(self, batch, batch_idx):
        (x1, x2), y = batch; model_name = self.hparams['model']['name']
        
        if model_name == 'CAPC':
            f1, _ = self.encoder(x1); f2, _ = self.encoder_2(x2)
            t_samples = torch.randint(0, self.encoder.sequence_length - self.hparams['model']['timestep'] - 1, size=(1,)).long().item()
            loss_cpc1, c1 = self.CPC(f1, t_samples); loss_cpc2, c2 = self.CPC(f2, t_samples); loss_cpc = (loss_cpc1 + loss_cpc2) / 2
            z1_bt = self.projector(c1); z2_bt = self.projector(c2); loss_bt = barlow_twin_loss(z1_bt, z2_bt, self.hparams['model']['lambd']); loss = self.hparams['model']['cpc_coeff'] * loss_cpc + loss_bt
        else: # SimCLR, BarlowTwins, AutoFi
            f1, _ = self.encoder(x1); f2, _ = self.encoder_2(x2); f1_flat=f1.reshape(f1.shape[0],-1); f2_flat=f2.reshape(f2.shape[0],-1)
            z1 = self.projector(f1_flat); z2 = self.projector(f2_flat)
            if model_name == 'SimCLR': loss = nt_xent_loss(z1, z2, self.hparams['model'].get('temperature', 0.5))
            elif model_name == 'BarlowTwins': loss = barlow_twin_loss(z1, z2, self.hparams['model'].get('lambd', 0.0051))
            
            # --- CHANGE 5: Implement the full, three-part AutoFi loss ---
            elif model_name == 'AutoFi':
                # L_f (feature-level invariance)
                loss_f = cosine_similarity_loss(z1, z2)
                # L_dc (decorrelation)
                loss_dc = decorrelation_loss(z1, z2)
                # L_s (symbol-level consistency)
                s1 = self.symbol_head(z1); s2 = self.symbol_head(z2)
                loss_s = symbol_level_loss(s1, s2)
                
                # Final weighted sum
                loss_inv = (1 - self.hparams['model']['beta_s']) * loss_f + self.hparams['model']['beta_s'] * loss_s
                loss = loss_inv + self.hparams['model']['lambd'] * loss_dc
            else: raise ValueError(f"Unknown model name: {model_name}")
        
        self.log("train_loss", loss, prog_bar=True); return loss

    def validation_step(self, batch, batch_idx): # Validation follows the same logic as training
        # ... (validation logic is very similar to training logic, so I will omit the full code for brevity) ...
        # ... The important part is that the loss calculation for AutoFi is also updated here ...
        x_clean, y = batch; model_name = self.hparams['model']['name']; x_augmented = self.val_transform(x_clean.clone())
        f1, _ = self.encoder(x_clean); f2, _ = self.encoder_2(x_augmented)
        if model_name == 'CAPC': # ... CAPC logic ...
            t_samples=torch.randint(0,self.encoder.sequence_length-self.hparams['model']['timestep']-1,size=(1,)).long().item(); loss_cpc1,c1=self.CPC(f1,t_samples); loss_cpc2,c2=self.CPC(f2,t_samples); loss_cpc=(loss_cpc1+loss_cpc2)/2; z1_bt=self.projector(c1); z2_bt=self.projector(c2); loss_bt=barlow_twin_loss(z1_bt,z2_bt,self.hparams['model']['lambd']); loss=self.hparams['model']['cpc_coeff']*loss_cpc+loss_bt
        else:
            f1_flat=f1.reshape(f1.shape[0],-1); f2_flat=f2.reshape(f2.shape[0],-1); z1=self.projector(f1_flat); z2=self.projector(f2_flat)
            if model_name=='SimCLR': loss=nt_xent_loss(z1,z2,self.hparams['model'].get('temperature',0.5))
            elif model_name=='BarlowTwins': loss=barlow_twin_loss(z1,z2,self.hparams['model'].get('lambd',0.0051))
            elif model_name=='AutoFi':
                loss_f=cosine_similarity_loss(z1,z2); loss_dc=decorrelation_loss(z1,z2); s1=self.symbol_head(z1); s2=self.symbol_head(z2); loss_s=symbol_level_loss(s1,s2)
                loss_inv=(1-self.hparams['model']['beta_s'])*loss_f+self.hparams['model']['beta_s']*loss_s; loss=loss_inv+self.hparams['model']['lambd']*loss_dc
        self.log("val_loss", loss, prog_bar=True); return loss
    
    def configure_optimizers(self): # Unchanged
        # ... (rest of this function is the same) ...
        param_weights=[p for p in self.parameters() if p.ndim>1]; param_biases=[p for p in self.parameters() if p.ndim==1]; parameters=[{'params':param_weights},{'params':param_biases}]; optimizer=LARS(parameters,lr=0.2,weight_decay=self.hparams['model'].get('weight_decay',1.5e-6)); return optimizer

# --- The LARS optimizer and LinearClassifierModel are UNCHANGED ---
class LARS(optim.Optimizer): # ...
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001, weight_decay_filter=False, lars_adaptation_filter=False): defaults=dict(lr=lr,weight_decay=weight_decay,momentum=momentum,eta=eta,weight_decay_filter=weight_decay_filter,lars_adaptation_filter=lars_adaptation_filter); super().__init__(params,defaults)
    def exclude_bias_and_norm(self,p):return p.ndim==1
    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for g in self.param_groups:
            for p in g['params']:
                dp=p.grad;
                if dp is None:continue
                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):dp.add_(p,alpha=g['weight_decay'])
                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):param_norm=torch.norm(p);update_norm=torch.norm(dp);one=torch.ones_like(param_norm);q=torch.where(param_norm>0.,torch.where(update_norm>0,(g['eta']*param_norm/update_norm),one),one);dp.mul_(q)
                param_state=self.state[p]
                if 'mu' not in param_state:param_state['mu']=torch.zeros_like(p)
                mu=param_state['mu'];mu.mul_(g['momentum']).add_(dp);p.add_(mu,alpha=-g['lr'])

class LinearClassifierModel(pl.LightningModule): # ...
    def __init__(self, pretrained_encoder, hparams):
        super(LinearClassifierModel, self).__init__(); self.save_hyperparameters(hparams); self.encoder=pretrained_encoder
        if self.hparams.get('freeze_encoder', True): self.encoder.eval(); [p.requires_grad_(False) for p in self.encoder.parameters()]
        self.linear_separation=nn.Linear(self.encoder.sequence_length*self.encoder.embedding_size,self.hparams['dataset']['num_classes'])
        self.automatic_optimization=False; self.train_accuracy=torchmetrics.Accuracy("multiclass",num_classes=self.hparams['dataset']['num_classes']); self.val_accuracy=torchmetrics.Accuracy("multiclass",num_classes=self.hparams['dataset']['num_classes']); self.test_accuracy=torchmetrics.Accuracy("multiclass",num_classes=self.hparams['dataset']['num_classes'])
    def forward(self, x):
        with torch.no_grad(): sequence_embedding, _ = self.encoder(x)
        return self.linear_separation(sequence_embedding.reshape(sequence_embedding.shape[0],-1))
    def _common_step(self, b, bi): x,y=b; yh=self(x); return F.cross_entropy(yh, y.squeeze()),yh,y.squeeze()
    def training_step(self, b, bi): l,yh,y=self._common_step(b,bi); self.train_accuracy.update(yh,y); opt=self.optimizers(); opt.zero_grad(); self.manual_backward(l); opt.step(); self.log("train_loss",l,on_step=True,prog_bar=True); return l
    def validation_step(self,b,bi): l,yh,y=self._common_step(b,bi); self.val_accuracy.update(yh,y); self.log("val_loss",l,on_epoch=True,prog_bar=True)
    def test_step(self, b, bi): l,yh,y=self._common_step(b,bi); self.test_accuracy.update(yh,y); self.log("test_acc_epoch",l,on_epoch=True)
    def on_train_epoch_end(self): self.log('train_acc_epoch',self.train_accuracy.compute(),on_step=False,on_epoch=True); self.train_accuracy.reset(); sch=self.lr_schedulers();
    def on_validation_epoch_end(self): self.log('val_acc_epoch',self.val_accuracy.compute(),on_step=False,on_epoch=True,prog_bar=True); self.val_accuracy.reset()
    def on_test_epoch_end(self): self.log('test_acc_epoch',self.test_accuracy.compute(),on_step=False,on_epoch=True); self.test_accuracy.reset()
    def configure_optimizers(self): optimizer=torch.optim.Adam(self.linear_separation.parameters(),lr=self.hparams['model']['lr'],weight_decay=self.hparams['model'].get('weight_decay',0)); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.hparams['model']['epochs']); return[optimizer],[scheduler]