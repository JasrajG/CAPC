# /Users/MAC/Projects/CAPC-replica/generate_embeddings.py

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from models import SpectrogramEncoder
from dataset import TeraNovaDataset

def generate_embeddings():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings using a frozen SSL encoder.")
    
    # --- NEW: Paths are now command-line arguments for flexibility ---
    parser.add_argument('--ssl_model_path', type=str, required=True, help="Path to the pre-trained SSL model .ckpt file.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the PREPROCESSED data folder to generate embeddings for.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the final embeddings.")
    
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--samples_per_class', type=int, default=1000, help="Number of samples to process from each modulation type.")
    args = parser.parse_args()

    # --- Step 1 & 2: Load the pre-trained encoder ---
    print(f"--- Loading encoder from SSL checkpoint: {args.ssl_model_path} ---")
    checkpoint = torch.load(args.ssl_model_path, map_location=torch.device('cpu'))
    hparams_ssl = checkpoint['hyper_parameters']['model']
    encoder = SpectrogramEncoder(
        spec_shape=hparams_ssl['spec_shape'],
        num_frames_per_window=hparams_ssl['window_width'],
        embedding_size=hparams_ssl['embedding_size'],
        recurrent_block=hparams_ssl['recurrent_block']
    )
    encoder_state_dict = {k.replace('encoder.', '', 1): v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()
    print("--- Encoder rebuilt successfully! ---")

    # --- Step 3: Set up the dataset from the specified data_path ---
    full_dataset = TeraNovaDataset(root_dir=args.data_path)
    
    print(f"\nCreating a balanced subset with {args.samples_per_class} samples per class...")
    labels_np = np.array(full_dataset.labels)
    num_classes = full_dataset.num_classes
    subset_indices = []
    for i in range(num_classes):
        class_indices = np.where(labels_np == i)[0]
        num_to_sample = min(args.samples_per_class, len(class_indices))
        selected_indices = np.random.choice(class_indices, num_to_sample, replace=False)
        subset_indices.extend(selected_indices)

    subset_dataset = Subset(full_dataset, subset_indices)
    print(f"Total size of subset for embedding generation: {len(subset_dataset)} samples.")
    loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Step 4: Generate embeddings ---
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Embeddings"):
            spectrograms, labels = batch
            embeddings_seq, _ = encoder(spectrograms)
            embeddings_flat = embeddings_seq.reshape(embeddings_seq.shape[0], -1)
            all_embeddings.append(embeddings_flat.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # --- Step 5: Save the results to the specified output_dir ---
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "embeddings.npy"), final_embeddings)
    np.save(os.path.join(args.output_dir, "labels.npy"), final_labels)

    print(f"\n--- EMBEDDING GENERATION FOR '{os.path.basename(args.data_path)}' COMPLETE ---")
    print(f"  Saved to: {args.output_dir}")

if __name__ == '__main__':
    generate_embeddings()