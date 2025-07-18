# /Users/MAC/Projects/CAPC-replica/generate_embeddings.py

import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from models import SpectrogramEncoder
from dataset import TeraNovaDataset

def generate_embeddings():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings using a frozen SSL encoder.")
    
    # --- Arguments ---
    parser.add_argument('--ssl_model_path', type=str, required=True, help="Path to the pre-trained SSL model checkpoint.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the root PREPROCESSED data folder.")
    parser.add_argument('--file_list_path', type=str, default=None, help="[Optional] Path to a .txt file listing specific files to process.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the final embeddings.")
    
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--samples_per_class', type=int, default=1000, help="Number of samples per class for the downstream task pool.")
    args = parser.parse_args()
    pl.seed_everything(42)

    # --- Step 1: Load the pre-trained model and its hyperparameters ---
    print(f"--- Loading encoder from SSL checkpoint: {args.ssl_model_path} ---")
    checkpoint = torch.load(args.ssl_model_path, map_location=torch.device('cpu'))
    hparams_ssl = checkpoint['hyper_parameters']['model']

    # --- Step 2: Rebuild the encoder using the SAVED hyperparameters ---
    # This is the crucial fix. It reads the hparams from the checkpoint to ensure
    # the architecture is IDENTICAL to the one that was saved.
    print("--- Rebuilding encoder with saved hyperparameters... ---")
    encoder = SpectrogramEncoder(
        spec_shape=hparams_ssl['spec_shape'],
        num_frames_per_window=hparams_ssl['window_width'],
        embedding_size=hparams_ssl['embedding_size'],
        # The .get() method safely defaults to False if 'recurrent_block' isn't in the hparams
        # This makes it compatible with all our SSL models.
        recurrent_block=hparams_ssl.get('recurrent_block', False) 
    )
    
    # Extract only the encoder's weights from the checkpoint
    encoder_state_dict = {k.replace('encoder.', '', 1): v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval() # Set to evaluation mode
    print("--- Encoder rebuilt and weights loaded successfully! ---")

    # --- Step 3: Set up the dataset ---
    full_dataset = TeraNovaDataset(root_dir=args.data_path, file_list_path=args.file_list_path)
    
    dataset_to_process = full_dataset
    # We only do the random subsetting if we are NOT using a specific file list
    if args.file_list_path is None:
        print(f"\nNo file list provided. Creating a balanced subset from directory with {args.samples_per_class} samples per class...")
        labels_np = np.array(full_dataset.labels)
        num_classes = full_dataset.num_classes
        subset_indices = []
        for i in range(num_classes):
            class_indices = np.where(labels_np == i)[0]
            num_to_sample = min(args.samples_per_class, len(class_indices))
            if num_to_sample > 0:
                selected_indices = np.random.choice(class_indices, num_to_sample, replace=False)
                subset_indices.extend(selected_indices)
        dataset_to_process = Subset(full_dataset, subset_indices)
        print(f"Total size of subset for embedding generation: {len(dataset_to_process)} samples.")
    else:
        print(f"\nProcessing all {len(dataset_to_process)} files from the provided list: {args.file_list_path}")

    loader = DataLoader(dataset_to_process, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Step 4 & 5: Generate and save embeddings ---
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Embeddings"):
            spectrograms, labels = batch
            embeddings_seq, _ = encoder(spectrograms)
            embeddings_flat = embeddings_seq.reshape(embeddings_seq.shape[0], -1)
            all_embeddings.append(embeddings_flat.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_embeddings:
        print("\nERROR: No embeddings were generated.")
        return
        
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "embeddings.npy"), final_embeddings)
    np.save(os.path.join(args.output_dir, "labels.npy"), final_labels)

    print(f"\n--- EMBEDDING GENERATION COMPLETE ---")
    print(f"  Embeddings shape: {final_embeddings.shape}")
    print(f"  Saved to: {args.output_dir}")

if __name__ == '__main__':
    generate_embeddings()