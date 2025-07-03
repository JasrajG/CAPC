# dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class SignFiDataset(Dataset):
    def __init__(self, root_dir, type, env, global_min, global_max, link='all', mode='single', portion=10):
        self.root_dir = root_dir
        self.type = type
        self.env = env
        self.global_min = global_min
        self.global_max = global_max
        self.link = link
        self.mode = mode
        self.portion = portion
        self.csid, self.csiu, self.csi, self.label = None, None, None, None

        self._load_data()
        self._normalize_data()
        self._combine_links()

    def _load_data(self):
        def _load_and_process(path):
            if not os.path.exists(path):
                print(f"Warning: Data file not found at {path}")
                return None
            data = np.load(path)
            data_abs = np.abs(data)
            if "reduced(in_env)" in path:
                return np.transpose(data_abs, (0, 3, 2, 1))
            else:
                return np.transpose(data_abs, (3, 2, 1, 0))

        if self.type == 'train' and self.env == 'home':
            csid_path = os.path.join(self.root_dir, f'reduced(in_env)_{self.env}_csid_{self.type}_{self.portion}.npy')
            csiu_path = os.path.join(self.root_dir, f'reduced(in_env)_{self.env}_csiu_{self.type}_{self.portion}.npy')
            label_path = os.path.join(self.root_dir, f'reduced(in_env)_{self.env}_y_{self.type}_{self.portion}.npy')
        elif self.type == 'train' and self.env == 'lab':
            csid_path = os.path.join(self.root_dir, 'dl.npy')
            csiu_path = os.path.join(self.root_dir, 'ul.npy')
            label_path = os.path.join(self.root_dir, 'label_lab.npy')
        else:
            csid_path = os.path.join(self.root_dir, f'{self.env}_csid_{self.type}.npy')
            csiu_path = os.path.join(self.root_dir, f'{self.env}_csiu_{self.type}.npy')
            label_path = os.path.join(self.root_dir, f'{self.env}_y_{self.type}.npy')
        
        if self.link in ['dl', 'all']: self.csid = _load_and_process(csid_path)
        if self.link in ['ul', 'all']: self.csiu = _load_and_process(csiu_path)
        
        if os.path.exists(label_path):
            self.label = np.load(label_path)
            if self.label is not None: self.label = self.label - 1
        else:
            print(f"Warning: Label file not found at {label_path}")

    def _normalize_data(self):
        def _normalize(data):
            if data is None: return None
            denominator = self.global_max - self.global_min
            if denominator == 0: return data - self.global_min
            return (data - self.global_min) / denominator
        self.csid = _normalize(self.csid)
        self.csiu = _normalize(self.csiu)

    def _combine_links(self):
        if self.mode == 'single':
            csi_parts = [d for d in [self.csid, self.csiu] if d is not None]
            if not csi_parts: raise FileNotFoundError(f"No CSI data found for single mode in env '{self.env}'")
            self.csi = np.concatenate(csi_parts, axis=0)
            if len(csi_parts) > 1 and self.label is not None:
                self.label = np.concatenate((self.label, self.label), axis=0)

    def __len__(self):
        if self.mode == 'single': return len(self.csi) if self.csi is not None else 0
        return len(self.csid) if self.csid is not None else 0

    def __getitem__(self, idx):
        label_to_return = 0
        if self.label is not None:
            label_to_return = self.label[idx].astype('int64')
        if self.mode == 'dual':
            return torch.FloatTensor(self.csid[idx]), torch.FloatTensor(self.csiu[idx]), label_to_return
        else:
            return torch.FloatTensor(self.csi[idx]), label_to_return

def data_loader(cfg, num_workers=0):
    env = 'lab' if cfg.get('SignFi_env') == 'lab' else cfg.get('SignFi_env')
    global_min, global_max = cfg['global_min'], cfg['global_max']
    if env == 'lab':
        dataset = SignFiDataset(
            cfg['root_dir'], 'train', 'lab', global_min, global_max,
            link=cfg.get('SignFi_link', 'all'), mode=cfg.get('SignFi_mode', 'dual')
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True, num_workers=num_workers)
        return None, None, None, loader
    elif env == 'home':
        train_dataset = SignFiDataset(
            cfg['root_dir'], 'train', 'home', global_min, global_max,
            link=cfg.get('SignFi_link', 'all'), mode='single', portion=cfg.get('portion')
        )
        val_dataset = SignFiDataset(
            cfg['root_dir'], 'val', 'home', global_min, global_max,
            link=cfg.get('SignFi_link', 'all'), mode='single'
        )
        test_dataset = SignFiDataset(
            cfg['root_dir'], 'test', 'home', global_min, global_max,
            link=cfg.get('SignFi_link', 'all'), mode='single'
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader, None
    raise ValueError(f"Invalid SignFi_env: {cfg.get('SignFi_env')}")