#!/usr/bin/env python
import os
import time
import torch
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd  # Added for Excel support

def load_entities_with_metadata(file_path, limit=None):
    """
    Load entries and preserve metadata (IDs and text).
    Supports both Excel (.xlsx) files and tab-delimited text files.
    Returns a list of dictionaries containing 'id' and 'text'.
    
    If limit is specified, only returns up to that many entries.
    """
    entries = []
    if file_path.lower().endswith('.xlsx'):
        try:
            df = pd.read_excel(file_path)
            if 'question' not in df.columns:
                print("Error: Excel file must contain a 'question' column.")
                return []
            
            # Apply limit if specified
            if limit is not None:
                df = df.head(limit)
                
            for i, row in df.iterrows():
                entry = {'id': f"question_{i}", 'text': str(row['question'])}
                entries.append(entry)
            print(f"Loaded {len(entries)} entries from {file_path}")
        except Exception as e:
            print(f"Error reading Excel file {file_path}: {e}")
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if limit is not None and i >= limit:
                        break
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        entries.append({'id': parts[0], 'text': " ".join(parts[1:])})
            print(f"Loaded {len(entries)} entries from {file_path}")
        except FileNotFoundError:
            print(f"File {file_path} not found.")
    return entries

def generate_embeddings(entries, model, batch_size=32):
    """
    Generate embeddings for the given entries using the provided model.
    """
    embeddings = []
    with tqdm(total=len(entries), desc="Generating Embeddings") as pbar:
        for i in range(0, len(entries), batch_size):
            batch_texts = [entry['text'] for entry in entries[i:i+batch_size]]
            with torch.cuda.amp.autocast():
                # Use model.module.encode if model is DataParallel
                if isinstance(model, torch.nn.DataParallel):
                    batch_embeddings = model.module.encode(batch_texts, convert_to_numpy=True)
                else:
                    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
            pbar.update(len(batch_texts))
            # Free GPU memory every 100 batches
            if i % 100 == 0:
                torch.cuda.empty_cache()
    return np.vstack(embeddings)

def save_embeddings_and_metadata(embeddings, entries, embeddings_file_path, metadata_file_path):
    """
    Save embeddings (as a .npy file) and metadata (as a text file) for traceability.
    """
    np.save(embeddings_file_path, embeddings)
    print(f"Embeddings saved to {embeddings_file_path}")
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(f"{entry['id']}\t{entry['text']}\n")
    print(f"Metadata saved to {metadata_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Create embeddings from an input file")
    parser.add_argument("--input_file", type=str, 
                        default="../../datasets/reproduce/test_emed_q_with_paths.xlsx",
                        help="Input file with entries. For Excel files, expect a 'question' column.")
    parser.add_argument("--output_dir", type=str, 
                        default="questions_embedding",
                        help="Output directory for embeddings and metadata")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding generation")
    args = parser.parse_args()

    # GPU setup and model initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("Initializing model...")
    model = SentenceTransformer('all-distilroberta-v1')
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using DataParallel")
        model = torch.nn.DataParallel(model)

    os.makedirs(args.output_dir, exist_ok=True)
    embeddings_file_path = os.path.join(args.output_dir, "embeddings.npy")
    metadata_file_path = os.path.join(args.output_dir, "metadata.txt")

    start_time = time.time()
    print("\nLoading entries...")
    entries = load_entities_with_metadata(args.input_file)
    if not entries:
        return

    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(entries, model, batch_size=args.batch_size)
    print("\nSaving embeddings and metadata...")
    save_embeddings_and_metadata(embeddings, entries, embeddings_file_path, metadata_file_path)
    print(f"\nProcess completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        torch.cuda.empty_cache()
