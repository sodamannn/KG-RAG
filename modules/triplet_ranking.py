#!/usr/bin/env python
import os
import json
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import requests
import torch
import argparse
import csv

from transformers import AutoTokenizer, AutoModel  # (if needed for custom encoding)

__all__ = ["main", "EmbeddingsGenerator"]

# --- Helper Functions ---

def load_embeddings_from_parquet(parquet_path):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if 'embedding' not in df.columns:
        raise ValueError(f"No 'embedding' column in {parquet_path}")
    embeddings = np.stack(df['embedding'].values, axis=0)
    return embeddings

def get_sorted_chunks(directory, embed_prefix="embeddings_chunk_", meta_prefix="metadata_chunk_"):
    embed_files = [f for f in os.listdir(directory) if f.startswith(embed_prefix) and f.endswith(".npy")]
    meta_files = [f for f in os.listdir(directory) if f.startswith(meta_prefix) and f.endswith(".json")]
    def extract_index(fname, prefix):
        base = fname.replace(prefix, "")
        base = os.path.splitext(base)[0]
        return int(base)
    embed_files = sorted(embed_files, key=lambda x: extract_index(x, embed_prefix))
    meta_files = sorted(meta_files, key=lambda x: extract_index(x, meta_prefix))
    assert len(embed_files) == len(meta_files), f"Mismatch in embeddings vs metadata count in {directory}"
    return embed_files, meta_files

def load_wikidata_triplet2(directory):
    embed_files, meta_files = get_sorted_chunks(directory)
    all_embeddings = []
    all_metadata = []
    for ef, mf in tqdm(zip(embed_files, meta_files), total=len(embed_files), desc="Loading triplet2 chunks", unit="chunk"):
        emb_path = os.path.join(directory, ef)
        meta_path = os.path.join(directory, mf)
        embeddings = np.load(emb_path)
        with open(meta_path, 'r') as f:
            metadata_list = json.load(f)
        assert len(embeddings) == len(metadata_list), f"Size mismatch in {ef} and {mf}"
        all_embeddings.append(embeddings)
        all_metadata.extend(metadata_list)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_metadata

def load_wikidata_triplet3(directory):
    files = [
        ("snomed_parent_child_triples_embeddings.npy", "snomed_parent_child_triples_metadata.json"),
        ("umls_type_groups_triples_embeddings.npy", "umls_type_groups_triples_metadata.json")
    ]
    all_embeddings = []
    all_metadata = []
    for emb_file, meta_file in files:
        emb_path = os.path.join(directory, emb_file)
        meta_path = os.path.join(directory, meta_file)
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        embeddings = np.load(emb_path)
        with open(meta_path, "r") as f:
            metadata_list = json.load(f)
        assert len(embeddings) == len(metadata_list), f"Mismatch in {emb_file} and {meta_file}"
        all_embeddings.append(embeddings)
        all_metadata.extend(metadata_list)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_metadata

def normalize_embeddings(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return emb / norms

def extract_ids_from_triplet2(meta):
    return [meta.get("head_id"), meta.get("relation_id"), meta.get("tail_id")]

def extract_ids_from_triplet3(meta):
    head_id = meta.get("head", {}).get("id")
    relation_id = meta.get("relation", {}).get("id")
    tail_id = meta.get("tail", {}).get("id")
    if head_id and relation_id and tail_id and head_id.startswith("Q") and relation_id.startswith("P") and tail_id.startswith("Q"):
        return [head_id, relation_id, tail_id]
    return []

# --- Bulk label fetching ---
label_cache = {}

def bulk_fetch_labels(entity_ids):
    ids_list = list(entity_ids)
    batch_size = 50
    for i in range(0, len(ids_list), batch_size):
        subset = ids_list[i:i+batch_size]
        query_ids = "|".join(subset)
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={query_ids}&languages=en&format=json"
        try:
            r = requests.get(url)
            if r.status_code != 200:
                print(f"Failed to fetch labels for IDs: {query_ids}")
                continue
            data = r.json()
            entities = data.get("entities", {})
            for eid, val in entities.items():
                labels = val.get("labels", {})
                if "en" in labels:
                    label_cache[eid] = labels["en"]["value"]
                else:
                    label_cache[eid] = None
        except Exception as e:
            print(f"Exception during fetching labels: {e}")
            continue

def get_english_label(entity_id):
    return label_cache.get(entity_id)

def get_english_triplet(doc_id, meta):
    if doc_id.startswith("triplet2_"):
        head_id = meta.get("head_id")
        relation_id = meta.get("relation_id")
        tail_id = meta.get("tail_id")
        if not (head_id and relation_id and tail_id):
            return None
        head_eng = get_english_label(head_id)
        relation_eng = get_english_label(relation_id)
        tail_eng = get_english_label(tail_id)
        if head_eng and relation_eng and tail_eng:
            return {
                "head_id": head_id,
                "head_entity": head_eng,
                "relation_id": relation_id,
                "relation": relation_eng,
                "tail_id": tail_id,
                "tail_entity": tail_eng,
                "english_triplet": f"<{head_eng}, {relation_eng}, {tail_eng}>"
            }
        else:
            return None
    elif doc_id.startswith("triplet3_"):
        head_id = meta.get("head", {}).get("id")
        relation_id = meta.get("relation", {}).get("id")
        tail_id = meta.get("tail", {}).get("id")
        if not (head_id and relation_id and tail_id):
            return None
        if not (head_id.startswith("Q") and relation_id.startswith("P") and tail_id.startswith("Q")):
            return None
        head_eng = get_english_label(head_id)
        relation_eng = get_english_label(relation_id)
        tail_eng = get_english_label(tail_id)
        if head_eng and relation_eng and tail_eng:
            return {
                "head_id": head_id,
                "head_entity": head_eng,
                "relation_id": relation_id,
                "relation": relation_eng,
                "tail_id": tail_id,
                "tail_entity": tail_eng,
                "english_triplet": f"<{head_eng}, {relation_eng}, {tail_eng}>"
            }
        else:
            return None
    else:
        return None

#############################
# Main function for triplet ranking
#############################
def main():
    parser = argparse.ArgumentParser(description="Rank Wikidata triplet embeddings for medical domain and output final results")
    parser.add_argument("--synthea_parquet", type=str, default="synthea_ques_embedding_full/chroma-embeddings.parquet",
                        help="Path to the Synthea question embeddings Parquet file")
    parser.add_argument("--triplet2_dir", type=str, default="wikidata_embedding_triplet2",
                        help="Directory containing wikidata_triplet2 embeddings and metadata")
    parser.add_argument("--triplet3_dir", type=str, default="wikidata_embedding_triplet3",
                        help="Directory containing wikidata_triplet3 embeddings and metadata")
    parser.add_argument("--output_json", type=str, default="synthea_top10_similar2.json",
                        help="Output JSON file for final ranking results")
    args = parser.parse_args()

    base_dir = os.getcwd()  # assume current working directory is /app/layers
    print("Loading Synthea question embeddings from Parquet...")
    try:
        synthea_embeddings = load_embeddings_from_parquet(args.synthea_parquet)
    except Exception as e:
        print(f"Error loading Synthea embeddings: {e}")
        return
    print(f"Loaded Synthea embeddings: {synthea_embeddings.shape[0]} samples.")

    print("Loading wikidata_triplet2 embeddings...")
    triplet2_embeddings, triplet2_metadata = load_wikidata_triplet2(args.triplet2_dir)
    print(f"Loaded wikidata_triplet2 embeddings: {triplet2_embeddings.shape[0]} samples.")

    print("Loading wikidata_triplet3 embeddings...")
    triplet3_embeddings, triplet3_metadata = load_wikidata_triplet3(args.triplet3_dir)
    print(f"Loaded wikidata_triplet3 embeddings: {triplet3_embeddings.shape[0]} samples.")

    print("Normalizing triplet embeddings...")
    triplet2_embeddings = normalize_embeddings(triplet2_embeddings)
    triplet3_embeddings = normalize_embeddings(triplet3_embeddings)

    # For GPU processing, assign triplet3 embeddings to available GPUs.
    device_ids = [0, 1, 2, 3, 4]  # Adjust according to your system
    devices = [f'cuda:{i}' for i in device_ids]
    num_devices = len(devices)
    print(f"Using {num_devices} GPUs for triplet3 similarity computation.")
    triplet3_tensors = []
    for dev in devices:
        triplet3_tensors.append(torch.tensor(triplet3_embeddings, dtype=torch.float32).to(dev))

    max_questions = synthea_embeddings.shape[0]
    print(f"Processing {max_questions} questions.")
    questions = synthea_embeddings[:max_questions]
    questions = normalize_embeddings(questions).astype(np.float32)

    # Distribute questions across GPUs
    questions_per_gpu = (max_questions + num_devices - 1) // num_devices
    gpu_assignments = []
    for i in range(num_devices):
        start_q = i * questions_per_gpu
        end_q = min(start_q + questions_per_gpu, max_questions)
        if start_q < end_q:
            gpu_assignments.append((i, start_q, end_q))
    print(f"Assigned questions across {num_devices} GPUs.")

    # Process triplet2 on CPU: For each question, keep top 7 indices.
    top7_triplet2_indices = {q: [] for q in range(max_questions)}
    top7_triplet2_sims = {q: [] for q in range(max_questions)}
    triplet2_chunk_size = 100000  # Adjust based on available memory
    num_triplet2 = triplet2_embeddings.shape[0]
    print("Processing triplet2 embeddings in chunks...")
    for chunk_start in tqdm(range(0, num_triplet2, triplet2_chunk_size), desc="Triplet2 chunks", unit="chunk"):
        chunk_end = min(chunk_start + triplet2_chunk_size, num_triplet2)
        triplet2_chunk = triplet2_embeddings[chunk_start:chunk_end].astype(np.float32)
        sims = np.dot(questions, triplet2_chunk.T)  # Shape: [max_questions, chunk_size]
        for q in range(max_questions):
            sim_row = sims[q]
            current = top7_triplet2_sims[q]
            if len(current) < 7:
                remaining = 7 - len(current)
                idx_part = np.argpartition(sim_row, -remaining)[-remaining:]
                sorted_idx_part = idx_part[np.argsort(sim_row[idx_part])[::-1]]
                current.extend(sim_row[sorted_idx_part].tolist())
                top7_triplet2_indices[q].extend((chunk_start + sorted_idx_part).tolist())
            else:
                combined = np.array(top7_triplet2_sims[q] + sim_row.tolist())
                combined_idx = np.array(top7_triplet2_indices[q] + (chunk_start + np.arange(chunk_end - chunk_start)).tolist())
                top7_idx = np.argpartition(combined, -7)[-7:]
                sorted_top7_idx = top7_idx[np.argsort(combined[top7_idx])[::-1]]
                top7_triplet2_sims[q] = combined[sorted_top7_idx].tolist()
                top7_triplet2_indices[q] = combined_idx[sorted_top7_idx].tolist()
    print("Finished processing triplet2 embeddings.")

    # Process triplet3 on GPUs: For each question, get top 3 on assigned GPU.
    top3_triplet3_indices = {q: [] for q in range(max_questions)}
    top3_triplet3_sims = {q: [] for q in range(max_questions)}
    print("Processing triplet3 embeddings on GPUs...")
    for gpu_id, (gpu_idx, start_q, end_q) in enumerate(gpu_assignments):
        device = devices[gpu_idx]
        triplet3_tensor = triplet3_tensors[gpu_id]
        for q in range(start_q, end_q):
            q_embedding = questions[q].reshape(1, -1)
            q_tensor = torch.tensor(q_embedding, dtype=torch.float32).to(device)
            sim = torch.matmul(q_tensor, triplet3_tensor.T).squeeze(0).cpu().numpy()
            top3_idx = np.argpartition(sim, -3)[-3:]
            sorted_top3_idx = top3_idx[np.argsort(sim[top3_idx])[::-1]]
            top3_triplet3_indices[q].extend(sorted_top3_idx.tolist())
            top3_triplet3_sims[q].extend(sim[sorted_top3_idx].tolist())
    print("Finished processing triplet3 embeddings.")

    # Combine triplet2 and triplet3 results per question and keep top 10 overall.
    all_results = []
    triplet_ids_to_fetch = set()
    for q in range(max_questions):
        triplet2_ids = [f"triplet2_{idx}" for idx in top7_triplet2_indices[q]]
        triplet3_ids = [f"triplet3_{idx}" for idx in top3_triplet3_indices[q]]
        combined_ids = triplet2_ids + triplet3_ids
        combined_sims = top7_triplet2_sims[q] + top3_triplet3_sims[q]
        sorted_order = np.argsort(combined_sims)[::-1]
        sorted_ids = [combined_ids[i] for i in sorted_order]
        sorted_sims = [combined_sims[i] for i in sorted_order]
        top10_ids = sorted_ids[:10]
        top10_sims = sorted_sims[:10]
        for doc_id in top10_ids:
            triplet_ids_to_fetch.add(doc_id)
        all_results.append({
            "question_index": q,
            "results": [
                {
                    "rank": rank,
                    "id": doc_id,
                    "similarity": float(sim),
                    "metadata": {}  # Placeholder for English label metadata
                }
                for rank, (doc_id, sim) in enumerate(zip(top10_ids, top10_sims), start=1)
            ]
        })
    print("Finished combining triplet results.")

    # Extract unique Q/P IDs from triplet metadata for label fetching.
    unique_entity_ids = set()
    for qres in all_results:
        for r in qres["results"]:
            doc_id = r["id"]
            if doc_id.startswith("triplet2_"):
                idx = int(doc_id.split("_")[1])
                meta = triplet2_metadata[idx]
                ids = extract_ids_from_triplet2(meta)
            elif doc_id.startswith("triplet3_"):
                idx = int(doc_id.split("_")[1])
                meta = triplet3_metadata[idx]
                ids = extract_ids_from_triplet3(meta)
            else:
                ids = []
            for eid in ids:
                if eid and (eid.startswith("Q") or eid.startswith("P")):
                    unique_entity_ids.add(eid)
    print(f"Total unique Q/P IDs to fetch: {len(unique_entity_ids)}")

    print("Fetching English labels from Wikidata...")
    bulk_fetch_labels(unique_entity_ids)
    print("Finished fetching English labels.")

    # Replace triplet IDs with English labels.
    final_results = []
    skipped = 0
    for qres in all_results:
        q_idx = qres["question_index"]
        formatted_results = []
        for r in qres["results"]:
            doc_id = r["id"]
            if doc_id.startswith("triplet2_"):
                idx = int(doc_id.split("_")[1])
                meta = triplet2_metadata[idx]
                eng_triplet = get_english_triplet(doc_id, meta)
            elif doc_id.startswith("triplet3_"):
                idx = int(doc_id.split("_")[1])
                meta = triplet3_metadata[idx]
                eng_triplet = get_english_triplet(doc_id, meta)
            else:
                eng_triplet = None
            if eng_triplet is not None:
                formatted_results.append({
                    "rank": r["rank"],
                    "id": doc_id,
                    "similarity": r["similarity"],
                    "metadata": eng_triplet
                })
            else:
                skipped += 1
        final_results.append({
            "question_index": q_idx,
            "results": formatted_results
        })
    print(f"Skipped {skipped} triplets due to missing English labels.")

    output_json_path = os.path.join(base_dir, "synthea_top10_similar2.json")
    with open(output_json_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Final results saved to {output_json_path}")

if __name__ == "__main__":
    main()
