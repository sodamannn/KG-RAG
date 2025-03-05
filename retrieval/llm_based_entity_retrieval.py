import pandas as pd
import openai
import requests
import argparse
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

def identify_entities_with_LLM(text: str) -> List[str]:
    # Identify entities in the given text using the LLM model.
    client = openai.Client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             """
             You are an expert in named entity recognition. Identify the list of the relevant entities (preferably some entities from attribute 1 and its description, the others from attribute 2 and its description) in wikidata that is helpful to answer the question from the following given text. Respond with only the list of entities with their values, separated by commas.
             If you don't know the answer or if you can not found any entities in wikidata, just say null. Do not make anything up.
             For example:
             "Attribute 1 person-person_id and its description 1 the person domain contains records that uniquely identify each patient in the source data who is time at-risk to have clinical observations recorded within the source systems; a unique identifier for each person. 
             Attribute 2 beneficiarysummary-desynpuf_id and its description 2 beneficiarysummary pertain to a synthetic medicare beneficiary; beneficiary code. 
             Do attribute 1 and attribute 2 are semantically matched with each other?"
             The relevant entities in wikidata are: patient, unique identifier, beneficiary. 
             The response should be a list of entities in the given text, separated by commas, like this: patient, unique identifier, beneficiary.      
             """
            },
            {"role": "user", "content": text}
        ]
    )
    try:
        entities = response.choices[0].message.content.split(',')
        return [entity.strip() for entity in entities if entity.strip()]
    except (KeyError, AttributeError, IndexError) as e:
        print(f"Error processing response: {e}")
        return None

def query_wikidata_api(entity: str) -> Optional[Dict[str, str]]:
    # Query Wikidata API for the given entity to verify if it exist.
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": entity,
        "limit": 1
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'search' in data and len(data['search']) > 0:
            return data['search'][0]
    return None

def identify_wikidata_entities(text: str) -> List[Dict[str, Optional[Dict[str, str]]]]:
    # Identify Wikidata entities in the given text.
    entities = identify_entities_with_LLM(text)
    wikidata_entities = []
    for entity in entities:
        wikidata_entity = query_wikidata_api(entity)
        if wikidata_entity:
            wikidata_entities.append({
                "text": entity,
                "wikidata": wikidata_entity
            })
    return wikidata_entities

def identify_entity(data_file_path: str) -> List[List[Dict[str, Optional[Dict[str, str]]]]]:
    # Identify entities from text in an Excel file and write results back to the file.
    reader = pd.read_excel(data_file_path)
    results = []
    for i in tqdm(range(reader.shape[0]), desc="Identifying the wikidata entities for each rows"):
        text = reader.iloc[i, 9]
        entities = identify_wikidata_entities(text)
        results.append(entities if entities else [])
    
    # Write identified Wikidata entities back to the DataFrame
    reader['wikidata entities'] = [
        ", ".join([f"{entity['wikidata']['id']}" if entity['wikidata'] else None for entity in result])
        for result in results
    ]
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
        f'LLM_entity_retrieval_{llm_model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

    wikidata_entities = identify_entity(data_file)

    end_time = datetime.now()
    duration = end_time - start_time
    logging.info("\nExecution Time Summary:")
    logging.info(f"Start Time: {start_time}")
    logging.info(f"End Time: {end_time}")
    logging.info(f"Total Duration: {duration}")

if __name__ == "__main__":
    main()