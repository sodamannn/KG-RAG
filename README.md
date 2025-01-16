# KG-RAG4SM: Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching

This repo provides the source code & data of our paper "[Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching](https://arxiv.org/abs/2501.08686)" .

## Introduction
KG-RAG4SM is a knowledge graph-based retrieval-augmented generation (graph RAG) model for schema matching and data integration. 
- It introduces novel vector-based, graph traversal-based, and query-based graph retrievals, as well as a hybrid approach and ranking schemes that identify the most relevant subgraphs from external large knowledge graphs (KGs).
- It leverages the retrieved subgraphs to augment the LLMs and prompts for generating the final results for the complex schema-matching task. 
- It supports the mainstream backbone LLM, such as, gpt, mistral, jellyfish, llama, gemma, etc.

## Quick Start
### 1. Environment and Dependencies

* Run the following commands to create a conda environment:

```bash
conda create -y -n kgrag4sm python=3.8
```
*  Activate created conda environment and install the dependencies:
```bash
conda activate kgrag4sm
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install pandas==2.0.3
pip install openai==1.57.0
pip install transformers==4.46.3
pip install tokenizers==0.20.3
pip install accelerate==0.26.0
pip install scikit-learn==1.3.2
pip install openpyxl==3.1.0
pip install cachetools==5.5.0
pip install --upgrade huggingface_hub
```

### 2. Clone the project and configure the LLM
* Clone the project and download the data
```
git clone https://github.com/machuangtao/KG-RAG4SM.git
```
* Configure the GPT models with the OPENAI_API_KEY

```
export OPENAI_API_KEY="replace with your openai api key"
```

* Login with huggingface token for Jellyfish and Mistral models, please make sure you have been granted the access rights to the model from the huggingface.
```
huggingface-cli login
```

### 3. Run with the preprocessed data for reproduce
You can run the code with the preprocessed data (stored in `datasets/reproduce/`) with the generated schema matching questions and retrieved subgraphs from wikdiata. Make sure you have setuped the required arguments:

- `--dataset`: Choose from cms, mimic, synthea, emed
- `--backbone_llm_model`: Choose from gpt-4o-mini, jellyfish-8b, jellyfish-7b, mistral-7b
- `--retrieved_paths`: the various paths retrieved by different subgraph retrieval method

* Specifically, run kgrag4sm with the default arguments for different experiments:

```
python kgrag4sm_main.py
```
* Run the llm for schema matching without retrieved subgraphs:

```
python llm4sm_main.py
```

## Run with the raw data **(Optional)** 
If you would like to preprocess the raw data (stored in `datasets/original/`) and retrieve the subgraphs from the wikidata, you can run the subgraph retrieval according to following instructions:

### 
* Generate the schema matching questions 

```
python preprocess/generate_question.py
```
### LLM-based Entity Retrieval + BFS
* Retrieve the subgraphs with LLM-based entity retrieval + BFS graph traversal

```
python preprocess/llm_based_entity_retrieval.py
```
* Retrieve the subgraphs with LLM-based entity retrieval + BFS graph traversal. 
Make sure the entities retrieved by LLMs can be read from the columns with **index 10**. 

```
python preprocess/bfs_graph_traversal_wikidata.py
```

### LLM-based Subgraph Retrieval

* Retrieve the subgraphs with LLM-based subgraph retrieval

```
python preprocess/llm_based_subgraph_retrieval.py
```
### Vector-based Subgraph Retrieval

#### Prerequisites
KG-RAG4SM supports vector-based, graph traversal-based, and query-based graph retrievals, as well as a hybrid approach to identify the most relevant subgraphs from external large knowledge graphs (KGs).
- VectorDB. The [Chroma]{https://github.com/chroma-core/chroma} is employed to store and manage the embeddings of KG triples and entity, and relations, and implement the efficient vector similarity search.
- Docker. The Docker container is selected to manage the dependencies for creating embeddings, vector similarity search, and ranking-based subgraph refinement.

#### Start 

* Retrieve the subgraphs with vector-based entity retrieval + BFS graph traversal

* Retrieve the subgraphs with vector-based KG triples retrieval

* Subgraph refinement based on ranking

## Citation
If you find our work helpful, please cite by using the following BibTeX entry:

```bib
@article{ma2025kgrag4sm,
      title={Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching}, 
      author={Chuangtao Ma and Sriom Chakrabarti and Arijit Khan and Bálint Molnár},
      journal={arXiv preprint arXiv:2501.08686},
      year={2025}
    }
```

## Acknowledgment
The cms, synthea, and mimic dataset are originated from the following works, we thanks for their:
```
SMAT: An Attention-based Deep Learning Solution to the Automation of Schema Matching
hhttps://github.com/JZCS2018/SMAT

```