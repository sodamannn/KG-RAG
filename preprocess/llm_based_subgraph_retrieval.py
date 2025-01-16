import pandas as pd
import openai
import os
import sys
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

def identify_wikidata_subgraph_with_LLM(text: str) -> List[str]:
    """Use ChatGPT to identify subgraph or path from wikidata kg in the given text."""
    client = openai.Client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             """
             You are an expert in relevant subgraph extraction from large wikidata knowledge graphs. Identify the list of the relevant subgraph or path in wikidata that is helpful to answer the question from the following given text. Respond with only the list of entities with their values, separated by commas.
             If you don't know the answer or if you can not found any relevant subgraph or path in wikidata, just say null. Do not make anything up.
             For example:
             Question: "Attribute 1 person-person_id and its description 1 the person domain contains records that uniquely identify each patient in the source data who is time at-risk to have clinical observations recorded within the source systems. a unique identifier for each person. 
             Attribute 2 beneficiarysummary-desynpuf_id and its description 2 beneficiarysummary pertains to a synthetic medicare beneficiary; beneficiary code. 
             Do attribute 1 and attribute 2 are semantically matched with each other?"
                        
             The relevant entities that is relevant to above questions in wikidata are: patient(Q181600), unique identifier (Q6545185), beneficiary (Q2596417). 
             The response should be a list of paths or subgraph between these entities in wikidata, separated by "|", like this: patient (Q181600), subclass of (P279), customer (Q852835) -> customer (Q852835), subclass of (P279), beneficiary (Q2596417).
                        
             Remember the following tips when you are analyzing and extracting subgraphs or paths from the wikidata knowledge graph.
             Tips:
             (1) Identify the relevatn wikidata entitie that is relevant and helpful to answer above questions, preferly some entities from attribute 1 and its description, the others from attribute 2 and its description.
             (2) Create the entity pair based on identified entities from the wikidata knowledge graph.
             (3) Seach and find the 2-hops paths or subgraph between these entity pairs from the wikidata knowledge graph. 
             (4) Please return the path in the format of "entity1_value(entity2_id), relation1_value(relation1-property), entity2_value(entity2_id) -> entity2_value(entity3_id), relation1_value(relation1-property), entity3_value(entity3_id) " if the path is founded.
             (5) Please use the separator "|" to separate the path if multiple paths are found, like this: paths1 | paths2 | paths3.
             """
            },
             {"role": "user", "content": text}
        ]
    )
    try:
        paths = response.choices[0].message.content.split(',')
        return [path.strip() for path in paths if path.strip()]
    except (KeyError, AttributeError, IndexError) as e:
        print(f"Error processing response: {e}")
        return None

def identify_wikidata_paths(data_file_path: str) -> List[List[Dict[str, Optional[Dict[str, str]]]]]:
    # Identify wikidata subgraph from text in an Excel file and write results back to the file.
    reader = pd.read_excel(data_file_path)
    results = []
    for i in tqdm(range(reader.shape[0]), desc="Identifying the paths or subgraph from wikidata kgs for each rows"):
        text = reader.iloc[i, 9]
        paths = identify_wikidata_subgraph_with_LLM(text)
        results.append(paths if paths else ['No relevant subgraph or path found in wikidata'])
    
    # Write identified Wikidata entities back to the DataFrame
    reader['wikidata paths llm'] = [path for path in results]
    reader.to_excel(data_file_path, index=False)
    return results

def get_data_file(dataset: str) -> str:
    dataset_mapping = {
        "cms": "datasets/original/test_cms_q.xlsx",
        "mimic": "datasets/original/test_mimic_q.xlsx",
        "synthea": "datasets/original/test_synthea_q.xlsx",
        "emed": "datasets/original/test_emed_q.xlsx"
    }
    return dataset_mapping.get(dataset)

def setup_logging(log_dir: str, dataset: str, llm_model: str) -> str:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    dataset_log_dir = os.path.join(log_dir, dataset)
    if not os.path.exists(dataset_log_dir):
        os.makedirs(dataset_log_dir)

    log_filename = os.path.join(
        dataset_log_dir,
        f'LLM_sugraph_retrieval_{llm_model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s', 
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return log_filename

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cms', help='dataset: mimic, mimic, synthea, emed')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini',
                      help='Model name (e.g., gpt-4o-mini, jellyfish-8b, jellyfish-7b, mistral-7b)')
    parser.add_argument('--log_dir', type=str, default='logs/preprocess/', help='Directory for storing logs')

    args = parser.parse_args()

    # Set up OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Set up logging
    log_filename = setup_logging(args.log_dir, args.dataset, args.llm_model)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    start_time = datetime.now()

    # Load data
    try:
        data_file = get_data_file(args.dataset)
        if not data_file:
            raise ValueError(f"Invalid dataset: {args.dataset}")
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        sys.exit(1)

    wikidata_paths = identify_wikidata_paths(data_file)

    for i, path_list in enumerate(wikidata_paths):
        logging.info(f"Row {i+1} results:")
        for path in path_list:
            logging.info(path)

    end_time = datetime.now()
    duration = end_time - start_time
    logging.info("\nExecution Time Summary:")
    logging.info(f"Start Time: {start_time}")
    logging.info(f"End Time: {end_time}")
    logging.info(f"Total Duration: {duration}")

if __name__ == "__main__":
    main()