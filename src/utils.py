import logging
import os
from datetime import datetime
import torch
from typing import List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score

def get_devices(device: str) -> Tuple[torch.device, torch.device]:
    if device == "cpu":
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.device_count() >= 2: 
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
            print("device0: {}, device1: {}".format(device0, device1))
        elif torch.cuda.device_count() == 1:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            raise ValueError("No GPU available")
    return device0, device1

def setup_kgrag_logging(log_dir: str, dataset: str, backbone_llm_model: str, retrieved_paths: str) -> str:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    dataset_log_dir = os.path.join(log_dir, dataset)
    if not os.path.exists(dataset_log_dir):
        os.makedirs(dataset_log_dir)

    log_filename = os.path.join(
        dataset_log_dir,
        f'KGRAG4SM_{backbone_llm_model}_{retrieved_paths}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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


def setup_llm_logging(log_dir: str, dataset: str, backbone_llm_model: str) -> str:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    dataset_log_dir = os.path.join(log_dir, dataset)
    if not os.path.exists(dataset_log_dir):
        os.makedirs(dataset_log_dir)

    log_filename = os.path.join(
        dataset_log_dir,
        f'LLM4SM_{backbone_llm_model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

def extract_label(response: str) -> int:
    for char in response:
        if char in ['0', '1']:
            return int(char)
    return -1

def calculate_metrics(y_true: List[int], y_pred: List[int]):
    y_true_filtered = []
    y_pred_filtered = []

    for true_value, pred_value in zip(y_true, y_pred):
        if pred_value is not None:
            y_pred_filtered.append(pred_value)
            y_true_filtered.append(true_value)
        else:
            continue    

    if len(y_true_filtered) == 0 or len(y_pred_filtered) == 0:
        return 0.0, 0.0, 0.0, 0.0

    precision = precision_score(y_true_filtered, y_pred_filtered, average='binary', zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, average='binary', zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='binary', zero_division=0)
    accuracy = sum(1 for t, p in zip(y_true_filtered, y_pred_filtered) if t == p) / len(y_true_filtered)
    return precision, recall, f1, accuracy