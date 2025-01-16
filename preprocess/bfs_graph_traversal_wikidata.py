import requests
from collections import deque
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import sys, os
import argparse
import logging
from datetime import datetime
from cachetools import TTLCache

# Separate caches for different types of data
wikidata_cache = TTLCache(maxsize=1000, ttl=3600)
entity_label_cache = TTLCache(maxsize=1000, ttl=3600)
property_label_cache = TTLCache(maxsize=500, ttl=3600)
graph_cache = TTLCache(maxsize=500, ttl=3600)

# Constants for optimization
MAX_PATHS = 5
MAX_EDGES_PER_NODE = 50
REQUEST_TIMEOUT = 5
BFS_TIMEOUT = 90

def make_cache_key(*args):
    """Create a hashable cache key from arguments"""
    return str(args)

def query_wikidata(entity_id):
    if isinstance(entity_id, tuple):
        entity_id = entity_id[0]
    
    cache_key = make_cache_key('wikidata', entity_id)
    if cache_key in wikidata_cache:
        return wikidata_cache[cache_key]
    
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "languages": "en",
        "props": "claims|labels"
    }
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        result = data['entities'].get(entity_id)
        wikidata_cache[cache_key] = result
        return result
    except (requests.RequestException, KeyError, requests.Timeout) as e:
        logging.warning(f"Error querying Wikidata for {entity_id}: {str(e)}")
        wikidata_cache[cache_key] = None
        return None

def extract_entity_info(entity_data):
    if not entity_data:
        return [], []
    
    labels = []
    edges = []
    
    try:
        if 'labels' in entity_data and 'en' in entity_data['labels']:
            labels.append(entity_data['labels']['en']['value'])
        
        for prop in ['P31', 'P279']:
            if prop in entity_data.get('claims', {}):
                for claim in entity_data['claims'][prop][:3]:
                    value = claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id')
                    if value:
                        labels.append(value)
        
        if 'claims' in entity_data:
            edge_count = 0
            for prop, claims in entity_data['claims'].items():
                if edge_count >= MAX_EDGES_PER_NODE:
                    break
                for claim in claims[:5]:
                    if edge_count >= MAX_EDGES_PER_NODE:
                        break
                    if 'mainsnak' in claim and 'datavalue' in claim['mainsnak'] and \
                       claim['mainsnak']['datavalue']['type'] == 'wikibase-entityid':
                        target_id = claim['mainsnak']['datavalue']['value']['id']
                        edges.append((target_id, prop))
                        edge_count += 1
    
    except Exception as e:
        logging.warning(f"Error extracting entity info: {str(e)}")
        return labels, edges
    
    return labels, edges

def get_entity_label(entity_id, node_labels):
    """Get entity label with proper caching"""
    if entity_id in node_labels:
        return node_labels[entity_id][0]
    
    cache_key = make_cache_key('entity_label', entity_id)
    if cache_key in entity_label_cache:
        label = entity_label_cache[cache_key]
        node_labels[entity_id] = [label]
        return label
    
    try:
        entity_data = query_wikidata(entity_id)
        if entity_data and 'labels' in entity_data and 'en' in entity_data['labels']:
            label = entity_data['labels']['en']['value']
            entity_label_cache[cache_key] = label
            node_labels[entity_id] = [label]
            return label
    except Exception as e:
        logging.warning(f"Error getting entity label for {entity_id}: {str(e)}")
    
    entity_label_cache[cache_key] = entity_id
    return entity_id

def get_property_label(property_id):
    """Get property label with proper caching"""
    if not property_id:
        return ''
    
    cache_key = make_cache_key('property', property_id)
    if cache_key in property_label_cache:
        return property_label_cache[cache_key]
    
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": property_id,
        "format": "json",
        "languages": "en",
        "props": "labels"
    }
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        data = response.json()
        if 'entities' in data and property_id in data['entities'] and 'labels' in data['entities'][property_id]:
            label = data['entities'][property_id]['labels']['en']['value']
            property_label_cache[cache_key] = label
            return label
    except Exception as e:
        logging.warning(f"Error getting property label for {property_id}: {str(e)}")
    
    property_label_cache[cache_key] = str(property_id)
    return str(property_id)

def bfs_with_timeout(start_entity, end_entity=None, max_depth=3):
    """BFS implementation with proper error handling and caching"""
    cache_key = make_cache_key('bfs', start_entity, end_entity, max_depth)
    if cache_key in graph_cache:
        return graph_cache[cache_key]
    
    paths = []
    graph = {}
    node_labels = {}
    edge_labels = {}
    visited = {start_entity}
    queue = deque([(start_entity, [], time.time())])
    start_time = time.time()

    try:
        while queue and len(paths) < MAX_PATHS:
            if time.time() - start_time > BFS_TIMEOUT:
                logging.warning(f"BFS timeout for entity {start_entity}")
                break

            current_entity, path, entry_time = queue.popleft()
            
            if time.time() - entry_time > BFS_TIMEOUT / 2:
                continue

            if end_entity and current_entity == end_entity:
                paths.append(path)
                continue

            if len(path) >= max_depth:
                if not end_entity:
                    paths.append(path)
                continue

            if current_entity not in graph:
                try:
                    entity_data = query_wikidata(current_entity)
                    if entity_data:
                        labels, edges = extract_entity_info(entity_data)
                        if labels:
                            node_labels[current_entity] = labels
                        graph[current_entity] = edges
                    else:
                        graph[current_entity] = []
                except Exception as e:
                    logging.warning(f"Error querying entity {current_entity}: {str(e)}")
                    graph[current_entity] = []

            for target_id, prop in graph[current_entity][:MAX_EDGES_PER_NODE]:
                if prop not in edge_labels:
                    edge_labels[prop] = [get_property_label(prop)]
                
                if target_id not in visited:
                    visited.add(target_id)
                    new_path = path + [(current_entity, target_id, prop)]
                    queue.append((target_id, new_path, time.time()))

    except Exception as e:
        logging.warning(f"Error in BFS for {start_entity}: {str(e)}")
    
    result = (paths, graph, node_labels, edge_labels)
    graph_cache[cache_key] = result
    return result

def format_paths(paths, node_labels, edge_labels):
    """Format paths with proper error handling"""
    try:
        formatted_paths = []
        for path in paths[:MAX_PATHS]:
            path_triples = []
            for source, target, edge in path:
                source_label = get_entity_label(source, node_labels)
                target_label = get_entity_label(target, node_labels)
                edge_label = edge_labels.get(edge, [edge])[0]

                triple = f"{source_label} ({source}), {edge_label} ({edge}), {target_label} ({target})"
                path_triples.append(triple)
            formatted_paths.append(" -> ".join(path_triples))
        return " | ".join(formatted_paths)
    except Exception as e:
        logging.warning(f"Error formatting paths: {str(e)}")
        return None

def process_single_entity(entity, row_index):
    # Process single entity with proper error handling
    try:
        paths, graph, node_labels, edge_labels = bfs_with_timeout(entity)
        if paths:
            formatted_result = format_paths(paths, node_labels, edge_labels)
            return row_index, formatted_result
    except Exception as e:
        logging.warning(f"Error processing single entity {entity}: {str(e)}")
    return row_index, None

def process_entity_pair(pair, row_index):
    # Process entity pair with proper error handling
    try:
        start_entity, end_entity = pair
        paths, graph, node_labels, edge_labels = bfs_with_timeout(start_entity, end_entity)
        if paths:
            formatted_result = format_paths(paths, node_labels, edge_labels)
            return row_index, formatted_result
    except Exception as e:
        logging.warning(f"Error processing entity pair {pair}: {str(e)}")
    return row_index, None

def process_entity_and_search_path(data_file_path):
    # Main processing function with improved error handling and progress tracking
    df = pd.read_excel(data_file_path)
    updates = {}
    stats = {'empty': 0, 'single': 0, 'multiple': 0, 'success': 0, 'error': 0}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for i, row in tqdm(df.iterrows(), desc="Processing rows", total=len(df)):
            try:
                entities = [e.strip() for e in str(row.iloc[10]).split(",") if e.strip()]
                entities = [e for e in entities if e.startswith("Q")]

                if len(entities) == 0:
                    stats['empty'] += 1
                elif len(entities) == 1:
                    stats['single'] += 1
                    futures.append(executor.submit(process_single_entity, entities[0], i))
                else:
                    stats['multiple'] += 1
                    entity_pairs = list(combinations(entities[:3], 2))
                    for pair in entity_pairs:
                        futures.append(executor.submit(process_entity_pair, pair, i))
            except Exception as e:
                logging.warning(f"Error processing row {i}: {str(e)}")
                stats['error'] += 1
                continue
        
        for future in tqdm(futures, desc="Processing BFS searching and results"):
            try:
                row_index, result = future.result(timeout=BFS_TIMEOUT + 5)
                if result:
                    if row_index in updates:
                        updates[row_index].append(result)
                    else:
                        updates[row_index] = [result]
                    stats['success'] += 1
            except TimeoutError:
                logging.warning("Future processing timeout")
                stats['error'] += 1
            except Exception as e:
                logging.warning(f"Error processing future: {str(e)}")
                stats['error'] += 1
    
    # Update DataFrame and save
    for row_index, paths in updates.items():
        df.at[row_index, 'bfs paths'] = ' | '.join(paths)
    
    df.to_excel(data_file_path, index=False)
    
    # Log detailed statistics
    logging.info("\nProcessing complete:")
    logging.info(f"- Empty entity rows: {stats['empty']}")
    logging.info(f"- Single entity rows: {stats['single']}")
    logging.info(f"- Multiple entity rows: {stats['multiple']}")
    logging.info(f"- Successful processes: {stats['success']}")
    logging.info(f"- Failed processes: {stats['error']}")
    logging.info(f"- Total rows with paths: {len(updates)}")


def get_data_file(dataset: str) -> str:
    dataset_mapping = {
        "cms": "datasets/original/test_cms_bfs.xlsx",
        "mimic": "datasets/original/test_mimic_q.xlsx",
        "synthea": "datasets/original/test_synthea_q.xlsx",
        "emed": "datasets/original/test_emed_q.xlsx"
    }
    return dataset_mapping.get(dataset)

def setup_logging(log_dir: str, dataset: str) -> str:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    dataset_log_dir = os.path.join(log_dir, dataset)
    if not os.path.exists(dataset_log_dir):
        os.makedirs(dataset_log_dir)

    log_filename = os.path.join(
        dataset_log_dir,
        f'BFS_subgraph_traversal_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

    # Set up logging
    log_filename = setup_logging(args.log_dir, args.dataset)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Load data
    try:
        data_file = get_data_file(args.dataset)
        if not data_file:
            raise ValueError(f"Invalid dataset: {args.dataset}")
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        sys.exit(1)

    start_time = time.time()
    process_entity_and_search_path(data_file)
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")



if __name__ == '__main__':
     main()