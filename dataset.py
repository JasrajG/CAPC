# /Users/MAC/Projects/CAPC-replica/dataset.py

import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Transform:
    """
    Applies a series of random augmentations to a given spectrogram tensor
    based on a provided configuration dictionary.
    """
    def __init__(self, aug_config):
        # Store the configuration dictionary (e.g., {'time_mask': True, 'time_flip': False})
        self.config = aug_config if aug_config is not None else {}

    def __call__(self, x):
        # This function is executed when we call self.transform(x)
        
        # 1. Apply Time Flip only if enabled in the config and with 50% probability
        if self.config.get('time_flip', False) and random.random() > 0.5:
            x = torch.flip(x, dims=[-1])

        # 2. Apply Time Mask only if enabled in the config and with 50% probability
        if self.config.get('time_mask', False) and random.random() > 0.5:
            if x.shape[-1] > 5: # Check if tensor is wide enough for a mask
                mask_len = int(x.shape[-1] * 0.2)
                if mask_len > 0:
                    start_pos = random.randint(0, x.shape[-1] - mask_len)
                    x[:, :, start_pos:start_pos+mask_len] = 0
            
        # 3. Apply Gaussian Noise only if enabled in the config and with 50% probability
        if self.config.get('gaussian_noise', False) and random.random() > 0.5:
            noise = torch.randn_like(x) * 0.1
            x = x + noise
            
        return x

class TeraNovaDataset(Dataset):
    """
    UPDATED to correctly pass the augmentation configuration to the Transform class.
    """
    def __init__(self, root_dir, file_list_path=None, augmentations=None):
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []
        
        # Pass the specific augmentation config to the Transform class
        # If augmentations is None, no transforms will be applied.
        self.transform = Transform(augmentations) if augmentations else None
        
        mod_types = ['4PSK', '8PSK', '16QAM', '64QAM']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(mod_types)}
        self.classes = mod_types
        self.num_classes = len(self.classes)
        
        if file_list_path:
            print(f"Initializing dataset from file list: {file_list_path}")
            with open(file_list_path, 'r') as f:
                self.file_paths = [line.strip() for line in f if line.strip()]
        elif root_dir:
            print(f"Initializing dataset by walking directory: {root_dir}")
            for subdir, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('.npy'):
                        self.file_paths.append(os.path.join(subdir, file))
        else:
            raise ValueError("Must provide either a root_dir or a file_list_path.")

        # Get labels from the folder names of the loaded file paths
        for path in self.file_paths:
            mod_name = os.path.basename(os.path.dirname(path)).split('_')[0]
            if mod_name in self.class_to_idx:
                self.labels.append(self.class_to_idx[mod_name])
            else:
                print(f"Warning: Could not determine class for file {path}")
        print(f"Found {len(self.file_paths)} data files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spectrogram = np.load(self.file_paths[idx])
        label = self.labels[idx]
        x = torch.from_numpy(spectrogram).float()

        # Basic normalization is always applied
        mean, std = torch.mean(x), torch.std(x)
        if std > 0: x = (x - mean) / std
        
        # If we are in transform mode (for pre-training)...
        if self.transform:
            # ...create two independent, random augmentations.
            x1 = self.transform(x.clone())
            x2 = self.transform(x.clone())
            return (x1, x2), torch.tensor(label, dtype=torch.long)
        else:
            # Otherwise (for validation/testing), return a single, clean view.
            return x, torch.tensor(label, dtype=torch.long)


class EmbeddingDataset(Dataset):
    """ This class for the final evaluation is unchanged. """
    def __init__(self, embeddings_path, labels_path):
        self.embeddings = np.load(embeddings_path)
        self.labels = np.load(labels_path)
        print(f"Loaded {len(self.embeddings)} embeddings directly from file.")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return torch.from_numpy(embedding).float(), torch.from_numpy(np.array(label)).long()