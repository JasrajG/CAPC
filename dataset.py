# dataset.py (UPDATED to support augmentation probabilities)

import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Transform:
    """
    UPDATED to accept specific probabilities for each augmentation from a config dict.
    This allows us to tune the intensity of our augmentation strategy.
    """
    def __init__(self, aug_config):
        self.config = aug_config if aug_config is not None else {}
        
        # --- CHANGE 1: Get specific probabilities from the config, with a default of 0.0 (off) ---
        self.p_flip = self.config.get('time_flip_prob', 0.0)
        self.p_mask = self.config.get('time_mask_prob', 0.0)
        self.p_noise = self.config.get('gaussian_noise_prob', 0.0)

    def __call__(self, x):
        # --- CHANGE 2: Use the stored probabilities instead of a hardcoded 0.5 ---
        if random.random() < self.p_flip:
            x = torch.flip(x, dims=[-1])

        if random.random() < self.p_mask:
            if x.shape[-1] > 20:
                mask_len = int(x.shape[-1] * 0.2) # You could even tune this mask percentage!
                if mask_len > 0:
                    start_pos = random.randint(0, x.shape[-1] - mask_len)
                    x[:, start_pos:start_pos+mask_len] = 0
            
        if random.random() < self.p_noise:
            noise = torch.randn_like(x) * 0.1 # And you could tune this noise magnitude!
            x = x + noise
            
        return x

class SignalDataset(Dataset):
    """ This class is UNCHANGED. It just passes the aug_config down to Transform. """
    def __init__(self, root_dir, file_list_path=None, augmentations=None):
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []
        
        self.transform = Transform(augmentations) if augmentations else None
        
        mod_types = ['4PSK', '8PSK', '16QAM', '64QAM']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(mod_types)}
        self.classes = mod_types
        self.num_classes = len(self.classes)
        
        if file_list_path:
            with open(file_list_path, 'r') as f:
                self.file_paths = [line.strip() for line in f if line.strip()]
        elif root_dir:
            for subdir, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('.npy'):
                        self.file_paths.append(os.path.join(subdir, file))
        else:
            raise ValueError("Must provide either a root_dir or a file_list_path.")

        for path in self.file_paths:
            mod_name = os.path.basename(os.path.dirname(path)).split('_')[0]
            if mod_name in self.class_to_idx:
                self.labels.append(self.class_to_idx[mod_name])
            else:
                print(f"Warning: Could not determine class for file {path}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        iq_signal = np.load(self.file_paths[idx])
        label = self.labels[idx]
        x = torch.from_numpy(iq_signal).float()

        mean, std = torch.mean(x), torch.std(x)
        if std > 0: x = (x - mean) / std
        
        if self.transform:
            x1 = self.transform(x.clone())
            x2 = self.transform(x.clone())
            return (x1, x2), torch.tensor(label, dtype=torch.long)
        else:
            return x, torch.tensor(label, dtype=torch.long)