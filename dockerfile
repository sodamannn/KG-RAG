FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies in groups
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    pandas==1.5.3 \
    scipy==1.9.3 \
    scikit-learn==1.2.2 \
    matplotlib==3.6.3 \
    networkx==2.8.8 \
    tqdm==4.66.1

RUN pip3 install --no-cache-dir \
    tokenizers==0.12.1 \
    nltk==3.7 \
    gensim==4.3.1 \
    uvicorn==0.22.0 \
    fastapi==0.99.1 \
    typer==0.4.2 \
    cython==0.29.28

RUN pip3 install --no-cache-dir \
    huggingface_hub==0.10.0 \
    transformers==4.20.1 \
    datasets==2.7.1 \
    spacy==3.3.3

RUN pip3 install --no-cache-dir \
    allennlp==2.10.1 \
    cached-path==1.1.6

# Install torch-geometric
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    ipython \
    jupyter \
    Pillow \
    requests \
    sympy \
    pytest

# Download spaCy model
RUN python3 -m spacy download en_core_web_sm

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
