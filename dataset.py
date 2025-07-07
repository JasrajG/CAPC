# /Users/MAC/Projects/CAPC-replica/dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TeraNovaDataset(Dataset):
    """
    PyTorch Dataset for PRE-PROCESSED TeraNova data.
    - Loads pre-computed .npy spectrogram files.
    - Used by the `generate_embeddings.py` script.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []
        
        mod_types = ['4PSK', '8PSK', '16QAM', '64QAM']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(mod_types)}
        self.classes = mod_types
        self.num_classes = len(self.classes)
        
        print(f"Initializing dataset from PRE-PROCESSED directory: {root_dir}")
        for subdir, _, files in os.walk(root_dir):
            try:
                mod_name = os.path.basename(subdir).split('_')[0]
                if mod_name in self.class_to_idx:
                    label = self.class_to_idx[mod_name]
                    for file in files:
                        if file.endswith('.npy'):
                            self.file_paths.append(os.path.join(subdir, file))
                            self.labels.append(label)
            except IndexError:
                continue
        print(f"Found {len(self.file_paths)} pre-processed data files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spectrogram = np.load(self.file_paths[idx])
        label = self.labels[idx]

        mean, std = np.mean(spectrogram), np.std(spectrogram)
        if std > 0: spectrogram = (spectrogram - mean) / std
        
        return torch.from_numpy(spectrogram), torch.tensor(label, dtype=torch.long)


class EmbeddingDataset(Dataset):
    """ 
    A super-fast dataset that just loads pre-computed embeddings and labels. 
    - Used by the final `supervised.py` script.
    """
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