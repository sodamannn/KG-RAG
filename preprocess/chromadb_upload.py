#!/usr/bin/env python
import os
import json
import numpy as np
import argparse
import chromadb
from chromadb.config import Settings

def create_client_for_collection(collection_dir):
    """
    Create a Chroma client using DuckDB+Parquet backend.
    """
    return chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=collection_dir
    ))

def add_data(embeddings_file, metadata_file, target_dir, collection_name):
    """
    Load embeddings and metadata files and add them to the specified ChromaDB collection.
    """
    os.makedirs(target_dir, exist_ok=True)
    client = create_client_for_collection(target_dir)
    # Create the collection (or get it if it already exists)
    try:
        collection = client.create_collection(collection_name)
    except Exception as e:
        if "already exists" in str(e):
            collection = client.get_collection(collection_name)
        else:
            raise e

    # Load embeddings from .npy file
    embeddings = np.load(embeddings_file)
    
    # Read metadata: expect each line to be "id<TAB>text"
    metadata_list = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                metadata_list.append({"id": parts[0], "text": " ".join(parts[1:])})
    
    if len(embeddings) != len(metadata_list):
        raise ValueError("Mismatch between number of embeddings and metadata entries.")

    # Create unique IDs for each entry in the collection
    ids = [f"{collection_name}_{i}" for i in range(len(embeddings))]
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadata_list,
        documents=[""] * len(embeddings)
    )
    client.persist()
    print(f"Completed uploading data to collection '{collection_name}' in {target_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Upload embeddings to ChromaDB")
    parser.add_argument("--data_type", type=str, choices=["entities", "relationships"], required=True,
                        help="Type of data to upload: 'entities' or 'relationships'")
    parser.add_argument("--embeddings_file", type=str, default="",
                        help="Path to the embeddings .npy file")
    parser.add_argument("--metadata_file", type=str, default="",
                        help="Path to the metadata file")
    parser.add_argument("--target_dir", type=str, default="",
                        help="Target directory for the ChromaDB")
    parser.add_argument("--collection_name", type=str, default="",
                        help="Name of the collection in ChromaDB")
    args = parser.parse_args()

    # Set defaults based on the data type if not provided
    if args.data_type == "entities":
        if not args.embeddings_file:
            args.embeddings_file = "wikidata_embedding_entities/embeddings.npy"
        if not args.metadata_file:
            args.metadata_file = "wikidata_embedding_entities/metadata.txt"
        if not args.target_dir:
            args.target_dir = "chromadb_store_wikidata_entities"
        if not args.collection_name:
            args.collection_name = "wikidata_entities"
    elif args.data_type == "relationships":
        if not args.embeddings_file:
            args.embeddings_file = "wikidata_embedding_relationships/embeddings.npy"
        if not args.metadata_file:
            args.metadata_file = "wikidata_embedding_relationships/metadata.txt"
        if not args.target_dir:
            args.target_dir = "chromadb_store_wikidata_relationships"
        if not args.collection_name:
            args.collection_name = "wikidata_relationships"

    add_data(args.embeddings_file, args.metadata_file, args.target_dir, args.collection_name)

if __name__ == "__main__":
    main()
