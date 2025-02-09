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

* Overview of this technique

KG‐RAG4SM supports multiple retrieval approaches for subgraph extraction from large external knowledge graphs. In the vector‐based subgraph retrieval mode, the system uses a vector database (ChromaDB) to store and manage embeddings (entities, relationships, and KG triples). Then it uses an efficient vector similarity search method (cosine similarity) to identify similar subgraphs. These subgraphs are later refined (e.g., by BFS graph traversal and ranking) to determine the most relevant subgraph(s) to augment a large language model’s input for schema matching.

#### Prerequisites
KG-RAG4SM supports vector-based, graph traversal-based, and query-based graph retrievals, as well as a hybrid approach to identify the most relevant subgraphs from external large knowledge graphs (KGs).
- VectorDB. The [Chroma]{https://github.com/chroma-core/chroma} is employed to store and manage the embeddings of KG triples and entity, and relations, and implement the efficient vector similarity search. You must have the embeddings for your entities, relationships, or KG triples. 
- Docker. The Docker container is selected to manage the dependencies for creating embeddings, vector similarity search, and ranking-based subgraph refinement.
- Other dependencies. PyTorch, Transformers, Pandas, OpenPyXL, etc.

#### Start 

1. Generate and Upload Embeddings:
   - First, you need to generate embeddings from your input schema (for example, from your Excel file containing schema questions) and upload these embeddings into ChromaDB.
   - Run the embedding generation script (e.g., create_embeddings.py) to create your embeddings.
   - Then, use the chromadb_upload.py script (or its updated location, such as under a processes folder) to upload the embeddings to ChromaDB.

2. Vector-based Retrieval:
   Using the embeddings in ChromaDB, the system retrieves candidate subgraphs. There are two retrieval modes:
   - Vector-based Entity Retrieval + BFS Graph Traversal: This approach first uses vector similarity search to identify the top candidate entities and then performs a BFS traversal on the knowledge graph to extract paths between these entities.
   - Vector-based KG Triples Retrieval: This mode directly retrieves candidate KG triples using vector similarity search on triple embeddings.
  
3. Subgraph Refinement Based on Ranking:
   - Once subgraphs are retrieved, a ranking module (e.g., the path_ranking) scores and selects the top similar subgraphs. This ranking uses the BFS results along with additional similarity data to produce a final, pruned result.


## Running using Docker

1. Build the Docker Image
   From the repository root (where your Dockerfile is located):
   - docker build -t kgrag4sm .
     
2. Run the Docker Container
   Run the container with GPU support and mount your repository into the container (adjust the local path as needed):
   - docker run --gpus all -it -v /path/to/KG-RAG4SM:/app kgrag4sm bash

3. Generate Embeddings
   Assuming you have an Excel file (for example, datasets/test_emed_q.xlsx) containing your schema-matching questions, generate embeddings using your custom embedding script. (If your       script is named create_embeddings.py and is located under the preprocess folder, run:)
   - cd /app/layers
   - python preprocess/create_embeddings.py --input_file ../datasets/test_emed_q.xlsx --output_dir questions_embedding

   This command will read the Excel file, generate embeddings using your chosen model (e.g., a SentenceTransformer or a custom model), and save the embeddings (e.g., as embeddings.npy) along      with metadata (e.g., as metadata.txt) into the questions_embedding folder.

4. Upload Embeddings to ChromaDB
   Next, upload your generated embeddings into ChromaDB. If your upload script is chromadb_upload.py (located in preprocess or another folder), run one command for entities and (if needed) one for relationships. For example:
   Upload Entity Embeddings:
   - python preprocess/chromadb_upload.py --data_type entities
     
   Upload Relationship Embeddings:
   - python preprocess/chromadb_upload.py --data_type relationships

 5. Run the retrieved pipeline
    Change to the layers directory (which contains main.py):
    - cd /app/layers
      
    a. Similarity Search
    - python main.py similarity_search
      
    This process calculates question embeddings (if not already computed), performs a similarity search against the stored embeddings, and outputs the results (typically to JSON and text files).

    b. BFS Paths Extraction
    Run the BFS paths extraction (with a maximum of 3 hops, for example):
    - python main.py bfs_paths --max_hops 3

    This process retrieves the top candidate entities from your similarity search and calculates the BFS traversal paths between them from your knowledge graph.

    c. Path Ranking
    Finally, run the path ranking process to score and select the best subgraphs. Make sure the filenames match your outputs (the defaults in the command below assume your BFS paths results are stored in cms_wikidata_paths_final_full.json and your question similarity results in cms_wikidata_similar_full.json):
    - python main.py path_ranking --bfs_results cms_wikidata_paths_final_full.json --question_similar_data cms_wikidata_similar_full.json --output_csv pruned_bfs_results.csv --output_json pruned_bfs_results.json

This command ranks the retrieved paths based on your scoring algorithm, producing a pruned set of subgraphs in both CSV and JSON formats.


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
