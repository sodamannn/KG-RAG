import pandas as pd
import openai
import os
import torch
import sys
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Optional, Union
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

class KGRAG_for_Schema_Matching:
    def generate_system_prompt(self) -> str:
        return """
                You are an expert in schema matching and data integration. 
                Your task is to analyze the attribute 1 with its textual description 1 and attribute 2 with its textual description 2 from source and target schema in the given question, and specify if the attribute 1 from source schema is semantically matched with attribute 2 from the target schema.
                In some questions, there is the knowledge graph context that might be helpful for you to answer. In this case, you will need to consider the provided context to make the correct decision. \n\n

                Here are two examples of the schema matching questions with correct answers and explanations that you need to learn before you start to analyze the potential mappings:
                Example 1:
                Attribute 1 death-person_id and its description 1 the death domain contains the clinical event for how and when a person dies. a person can have up to one record if the source system contains evidence about the death; a foreign key identifier to the deceased person. the demographic details of that person are stored in the person table. 
                Attribute 2 beneficiarysummary-desynpuf_id and its description 2 beneficiarysummary pertain to a synthetic medicare beneficiary; beneficiary code. 
                Do attribute 1 and attribute 2 are semantically matched with each other?
                Here is the correct answer and the explanations for the above-given example question: 1 
                Explanation: they are semantically matched with each other because both of them are unique identifiers for each person. Even if the death-person_id refers to the unique identifier of the person in the death table and beneficiarysummary-desynpuf_id refers to the unique identifier of the person beneficiary from beneficiarysummary table, they are semantically matched with each other. \n\n
                        
                Example 2:
                Attribute 1 death-person_id and its description 1 the death domain contains the clinical event for how and when a person dies. a person can have up to one record if the source system contains evidence about the death.;a foreign key identifier to the deceased person. the demographic details of that person are stored in the person table. 
                Attribute 2 beneficiarysummary-bene_birth_dt and its description 2 beneficiarysummary pertain to a synthetic medicare beneficiary; date of birth. 
                Do attribute 1 and attribute 2 are semantically matched with each other?
                Here is the knowledge graph context that might be helpful for you to answer the above schema matching question: 
                death (Q4), has part(s) of the class (P2670), date of death (Q18748141) -> date of death (Q18748141), opposite of (P461), date of birth (Q2389905) | human (Q5), has characteristic (P1552), age of a person (Q185836) -> age of a person (Q185836), uses (P2283), date of birth (Q2389905)
                Here is the correct answer and the explanations for the above-given example question: 0
                Explanation: they are not semantically matched with each other, because death-person_id is a unique identifier for each person in death table and bene_birth_dt is the date of birth of person in beneficiarysummary table. From the above context, we can found that date of death is opposite of date of birth, they are not semantically matched with each other.
                        
                Remember the following tips when you are analyzing the potential mappings.
                Tips:
                (1) Some letters are extracted from the full names and merged into an abbreviation word.
                (2) Schema information sometimes is also added as the prefix of abbreviation.
                (3) Please consider the abbreviation case. 
                (4) Please consider the knowledge graph context to make the correct decision when it is provided.
                """

    def generate_user_prompt(self, question: str, paths: Optional[str]) -> str:
        prompt = f"""Based on the provided example and the following knowledge graph context, please answer the following schema matching question:
        
        {question}

        Knowledge Graph Context:
        {paths if paths and paths != ['null'] else "No available knowledge graph context, please make the decision yourself. "}

        Please respond with the label: 1 if attribute 1 and attribute 2 are semantically matched with each other, otherwise respond lable: 0.
        Do not mention that there is not enough information to decide.
        """
        return prompt
    
    def get_llm_response(self, system_prompt: str, user_prompt: str, model: Union[str, pipeline]) -> str:
        if isinstance(model, str) and model.startswith('gpt'):
            client = openai.Client()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            terminators = [
                model.tokenizer.eos_token_id,
                model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            responses = model(
                messages,
                eos_token_id=terminators,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.5,
                top_k=1,
                top_p=0.9,
                pad_token_id=model.tokenizer.eos_token_id
            )
            answer = responses[0]['generated_text'][-1]["content"].strip()
            return answer
    
    def kgrag_query_for_schema_matching(self, question: str, paths: Optional[str], model) -> Tuple[str, str, str]:
        system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_user_prompt(question, paths)
        response = self.get_llm_response(system_prompt, user_prompt, model)
        return system_prompt, user_prompt, response

def extract_label(response: str) -> int:
    for char in response:
        if char in ['0', '1']:
            return int(char)
    return -1

def calculate_metrics(y_true, y_pred):
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

def setup_logging(log_dir: str, dataset: str, backbone_llm_model:str, retrieved_paths: str) -> str:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    dataset_log_dir = os.path.join(log_dir, dataset)
    if not os.path.exists(dataset_log_dir):
        os.makedirs(dataset_log_dir)

    log_filename = os.path.join(
        dataset_log_dir,
        f'KGRAG_matcher_{backbone_llm_model}_{retrieved_paths}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s', 
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_filename

def initialize_model(model_name: str):
    if model_name.startswith('gpt'):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return model_name
    else:
        if model_name == "jellyfish-8b":
            llm_pipeline = pipeline("text-generation",  model="NECOUDBFM/Jellyfish-8B", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        elif model_name == "jellyfish-7b":  
            llm_pipeline = pipeline("text-generation",  model="NECOUDBFM/Jellyfish-7B", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        elif model_name == "mistral-7b":
            llm_pipeline = pipeline("text-generation",  model="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        return llm_pipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description='kgrag4sm')
    parser.add_argument('--dataset', type=str, default='cms', help='dataset: cms, mimic, synthea, emed')
    parser.add_argument('--backbone_llm_model', type=str, default='jellyfish-8b',
                      help='Model name (e.g., gpt-4o-mini, jellyfish-8b, jellyfish-7b, mistral-7b)')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory for storing logs')
    parser.add_argument('--retrieved_paths', type=str, default="llm_paths",
                      help='Paths retrieved by different methods: llm_paths, llm_entity_retrieval_bfs_paths, vector_entity_retrieval_bfs_paths, ver_bfs_top1, ver_bfs_top2, vector_kg_triples_retrieval_paths, v_kg_tr_top1, v_kg_tr_top2')
    return parser.parse_args()

def main():
    args = parse_arguments()
    start_time = datetime.now()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Setup logging
    log_filename = setup_logging(args.log_dir, args.dataset, args.backbone_llm_model, args.retrieved_paths)
    
    # Initialize model
    try:
        model = initialize_model(args.backbone_llm_model)
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        sys.exit(1)
    
    # Load data
    try:
        if args.dataset=="cms":
            data_file = "datasets/reproduce/test_cms_q_with_paths.xlsx"
        elif args.dataset=="mimic":
            data_file = "datasets/reproduce/test_mimic_q_with_paths.xlsx"
        elif args.dataset=="synthea":
            data_file = "datasets/reproduce/test_synthea_q_with_paths.xlsx"
        elif args.dataset=="emed":
            data_file = "datasets/reproduce/test_emed_q_with_paths.xlsx"
        reader = pd.read_excel(data_file)
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        sys.exit(1)
    
    kgrag4sm = KGRAG_for_Schema_Matching()
    y_true = []
    y_pred = []
    
    
    for i in tqdm(range(reader.shape[0]), desc="Processing questions"):
        try:
            question = reader.iloc[i, 9]

            if args.retrieved_paths=="llm_entity_retrieval_bfs_paths":
                paths = reader.iloc[i, 10]
            elif args.retrieved_paths=="llm_paths":
                paths = reader.iloc[i, 11]
            elif args.retrieved_paths=="vector_entity_retrieval_bfs_paths":
                paths = reader.iloc[i, 12]
            elif args.retrieved_paths=="ver_bfs_top1":
                paths = reader.iloc[i, 13]
            elif args.retrieved_paths=="ver_bfs_top2":
                paths = reader.iloc[i, 14]
            elif args.retrieved_paths=="vector_kg_triples_retrieval_paths":
                paths = reader.iloc[i, 15]
            elif args.retrieved_paths=="v_kg_tr_top1":
                paths = reader.iloc[i, 16]
            elif args.retrieved_paths=="v_kg_tr_top2":
                paths = reader.iloc[i, 17]
            else:
                logging.error(f"Invalid retrieved paths: {args.retrieved_paths}")

            ground_truth = reader.iloc[i, 4]
            
            if pd.isna(paths):
                paths = None
            
            _, _, response = kgrag4sm.kgrag_query_for_schema_matching(question, paths, model)
            label = extract_label(response)
            
            
            if label != -1 and not pd.isna(ground_truth):
                y_true.append(int(ground_truth))
                y_pred.append(label)
            
            logging.info(f"\nResults for row {i+1}:")
            logging.info(f"Response: {response}")
            logging.info(f"Extracted Label: {label}")
            logging.info(f"Ground Truth: {ground_truth}")
            
        except Exception as e:
            logging.error(f"Error processing row {i+1}: {e}")
            continue
    
    # Calculate and log metrics
    precision, recall, f1, accuracy = calculate_metrics(y_true, y_pred)
    
    logging.info("\nFinal Metrics:")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    
    
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info("\nExecution Time Summary:")
    logging.info(f"Start Time: {start_time}")
    logging.info(f"End Time: {end_time}")
    logging.info(f"Total Duration: {duration}")

if __name__ == "__main__":
    main()