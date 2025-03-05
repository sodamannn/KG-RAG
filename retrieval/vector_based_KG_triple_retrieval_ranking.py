#!/usr/bin/env python
import argparse
import sys
import asyncio

def main():
    parser = argparse.ArgumentParser(description="KG-RAG4SM Triplet-Based Retrieval")
    
    # Arguments for triplet ranking
    parser.add_argument("--dataset", type=str, choices=["cms", "mimic", "synthea", "emed"], required=True,
                        help="Dataset to use (cms, mimic, synthea, emed)")
    parser.add_argument("--triplet2_dir", type=str, default="wikidata_embedding_triplet2",
                        help="Directory containing wikidata_triplet2 embeddings and metadata")
    parser.add_argument("--triplet3_dir", type=str, default="wikidata_embedding_triplet3",
                        help="Directory containing wikidata_triplet3 embeddings and metadata")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to process (for testing)")
 
    args = parser.parse_args()
 
    try:
        from modules import triplet_ranking
        
        # Determine dataset-specific paths
        dataset = args.dataset
        question_parquet = f"{dataset}_ques_embedding_full/chroma-embeddings.parquet"
        output_json = f"{dataset}_top10_similar2.json"
        
        print(f"Running triplet ranking for {dataset} dataset")
        print(f"Question embeddings: {question_parquet}")
        print(f"Output JSON: {output_json}")
        
        # Verify input files exist
        if not os.path.exists(question_parquet):
            print(f"Warning: Question embeddings file not found: {question_parquet}")
            print(f"Current working directory: {os.getcwd()}")
        
        # Set up arguments to match the triplet_ranking.py expectations
        cmd_args = [
            sys.argv[0],
            "--synthea_parquet", question_parquet,
            "--triplet2_dir", args.triplet2_dir,
            "--triplet3_dir", args.triplet3_dir,
            "--output_json", output_json
        ]
        
        # Use triplet_ranking with exactly the parameter names it expects
        sys.argv = cmd_args
        triplet_ranking.main()
        print(f"Triplet ranking completed successfully!")
        print(f"Results saved to: {output_json}")
            
    except ImportError as e:
        print(f"Error: Could not import triplet_ranking module: {e}")
        print("Please make sure the module exists and all dependencies are installed.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
 
if __name__ == "__main__":
    main()
