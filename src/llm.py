import os
import torch
from transformers import pipeline
from typing import List, Tuple

def initialize_llm_model(model_name: str, devices: Tuple[torch.device, torch.device]) -> pipeline:
    devices = [device.type for device in devices]
    if model_name.startswith('gpt'):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return model_name
    else:
        if model_name == "jellyfish-8b":
            llm_pipeline = pipeline("text-generation", model="NECOUDBFM/Jellyfish-8B", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        elif model_name == "jellyfish-7b":  
            llm_pipeline = pipeline("text-generation", model="NECOUDBFM/Jellyfish-7B", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        elif model_name == "mistral-7b":
            llm_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        return llm_pipeline
