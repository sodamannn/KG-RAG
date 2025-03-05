#!/usr/bin/env python
import os
import json
import requests
import torch
from chromadb import Client
from chromadb.config import Settings
from tqdm import tqdm
import csv
import numpy as np
import argparse

__all__ = ["main", "EmbeddingsGenerator"]

class EmbeddingsGenerator:
    def __init__(self):
        self.entity_label_cache = {}
        self.predicate_label_cache = {}
        self.client_questions = self.initialize_chromadb("chromadb_store_questions")
        self.collection_questions = self.client_questions.get_collection("questions_collection")
        
    def initialize_chromadb(self, persist_directory):
        return Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
    
    def get_question_embedding(self, question_id):
        result_data = self.collection_questions.get(ids=[question_id])
        if result_data and 'embeddings' in result_data and len(result_data['embeddings']) > 0:
            return np.array(result_data['embeddings'][0])
        return None
    
    def get_entity_label(self, entity_id):
        if entity_id in self.entity_label_cache:
            return self.entity_label_cache[entity_id]
        try:
            url = ("https://www.wikidata.org/w/api.php?action=wbgetentities"
                   f"&ids={entity_id}&format=json&props=labels&languages=en")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                entities = data.get('entities', {})
                entity_data = entities.get(entity_id, {})
                label_data = entity_data.get('labels', {})
                label = label_data.get('en', {}).get('value', '')
                if not label and label_data:
                    label = next(iter(label_data.values())).get('value', '')
                self.entity_label_cache[entity_id] = label
                return label
            else:
                print(f"Failed to retrieve label for entity {entity_id}. HTTP status code: {response.status_code}")
                return ''
        except Exception as e:
            print(f"Exception occurred while fetching label for entity {entity_id}: {e}")
            return ''
    
    def get_predicate_label(self, predicate_id):
        if predicate_id in self.predicate_label_cache:
            return self.predicate_label_cache[predicate_id]
        try:
            url = ("https://www.wikidata.org/w/api.php?action=wbgetentities"
                   f"&ids={predicate_id}&format=json&props=labels&languages=en")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                entities = data.get('entities', {})
                entity_data = entities.get(predicate_id, {})
                label_data = entity_data.get('labels', {})
                label = label_data.get('en', {}).get('value', '')
                if not label and label_data:
                    label = next(iter(label_data.values())).get('value', '')
                self.predicate_label_cache[predicate_id] = label
                return label
            else:
                print(f"Failed to retrieve label for predicate {predicate_id}. HTTP status code: {response.status_code}")
                return ''
        except Exception as e:
            print(f"Exception occurred while fetching label for predicate {predicate_id}: {e}")
            return ''

def main():
    parser = argparse.ArgumentParser(description="Rank BFS paths and export results")
    parser.add_argument("--bfs_results", type=str, default="cms_wikidata_paths_final_full.json", help="Input BFS results JSON file")
    parser.add_argument("--question_similar_data", type=str, default="cms_wikidata_similar_full.json", help="Input question similarity data JSON file")
    parser.add_argument("--output_csv", type=str, default="pruned_bfs_results.csv", help="Output CSV filename")
    parser.add_argument("--output_json", type=str, default="pruned_bfs_results.json", help="Output JSON filename")
    args = parser.parse_args()
    
    if not os.path.exists(args.bfs_results):
        print(f"Error: '{args.bfs_results}' file not found.")
        return
    
    # Load BFS results.
    with open(args.bfs_results, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        bfs_results = {item["question_id"]: item for item in data["results"]}
    else:
        bfs_results = data
    
    # Ensure each BFS result is a dictionary with keys "raw_paths" and "paths".
    # If a result is a string, wrap it appropriately.
    for qid, result in bfs_results.items():
        if isinstance(result, str):
            bfs_results[qid] = {
                "raw_paths": [],
                "paths": [result]
            }
        else:
            if "paths" in result and isinstance(result["paths"], str):
                result["paths"] = [result["paths"]]
            if "raw_paths" in result and isinstance(result["raw_paths"], str):
                result["raw_paths"] = [result["raw_paths"]]
    
    # Extract predicate IDs from the raw paths of all questions.
    predicate_ids_set = set()
    for question_id, result in bfs_results.items():
        raw_paths = result.get('raw_paths', [])
        for raw_path in raw_paths:
            if isinstance(raw_path, (list, tuple)):
                for element in raw_path:
                    if isinstance(element, (list, tuple)) and element[0] == 'predicate':
                        predicate_ids_set.add(element[1])
    predicate_ids = list(predicate_ids_set)
    print(f"Total unique predicates in raw paths: {len(predicate_ids)}")
    
    embeddings_generator = EmbeddingsGenerator()
    
    if not os.path.exists(args.question_similar_data):
        print(f"Error: '{args.question_similar_data}' file not found.")
        return
    
    with open(args.question_similar_data, 'r', encoding='utf-8') as f:
        question_similar_data = json.load(f)
    
    pruned_bfs_results = {}
    csv_rows = []
    fieldnames = ['question_id', 'question_text', 'path_labels', 'score']
    
    for question_id in tqdm(bfs_results.keys(), desc="Processing questions"):
        question_text = question_similar_data.get(question_id, {}).get('question_text', '')
        question_embedding = embeddings_generator.get_question_embedding(question_id)
        if question_embedding is None:
            print(f"No embedding found for {question_id}.")
            continue
        norm = np.linalg.norm(question_embedding)
        if norm != 0:
            question_embedding = question_embedding / norm
        
        result_item = bfs_results[question_id]
        raw_paths = result_item.get('raw_paths', [])
        formatted_paths = result_item.get('paths', [])
        if isinstance(formatted_paths, str):
            formatted_paths = [formatted_paths]
        
        path_scores = []
        for idx, raw_path in enumerate(raw_paths):
            if isinstance(raw_path, (list, tuple)):
                predicates_in_path = [element[1] for element in raw_path if isinstance(element, (list, tuple)) and element[0]=='predicate']
                score = sum(1 for pid in predicates_in_path if pid in predicate_ids_set)
                path_scores.append((score, idx))
        ranked_paths = sorted(path_scores, key=lambda x: x[0], reverse=True)
        if ranked_paths:
            if len(ranked_paths) == 1:
                score, idx = ranked_paths[0]
                pruned_bfs_results[question_id] = {
                    "raw_paths": [raw_paths[idx]] if idx < len(raw_paths) else [],
                    "path_labels": formatted_paths[idx] if idx < len(formatted_paths) else ""
                }
                csv_rows.append({
                    'question_id': question_id,
                    'question_text': question_text,
                    'path_labels': formatted_paths[idx] if idx < len(formatted_paths) else "",
                    'score': str(score)
                })
            else:
                top_paths = ranked_paths[:2]
                pruned_bfs_results[question_id] = {
                    "raw_paths": [raw_paths[i] for score, i in top_paths if i < len(raw_paths)],
                    "path_labels": " | ".join([formatted_paths[i] for score, i in top_paths if i < len(formatted_paths)])
                }
                scores_combined = " | ".join(str(score) for score, i in top_paths)
                csv_rows.append({
                    'question_id': question_id,
                    'question_text': question_text,
                    'path_labels': pruned_bfs_results[question_id]["path_labels"],
                    'score': scores_combined
                })
        else:
            print(f"No paths available for question {question_id}")
    
    # Write CSV output.
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"CSV file '{args.output_csv}' has been created.")
    
    # Write pruned JSON output.
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(pruned_bfs_results, f, indent=4)
    print(f"Pruned BFS results saved to '{args.output_json}'")

if __name__ == "__main__":
    main()
