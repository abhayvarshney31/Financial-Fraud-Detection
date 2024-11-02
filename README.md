# Financial Fraud Detection Embedding-Based Retrieval

This project will use embedding-based retrieval on an unstructured financial fraud dataset. The primary approach involves querying the embedding space to find the closest neighbor set of transaction behaviors. By comparing these neighbors, the system can identify patterns that indicate whether a transaction is likely to be fraudulent based on its similarity to known fraudulent cases. This approach facilitates accurate and efficient detection of fraud by leveraging vector similarity within the embedding space.

## Overview

1. **Gather Training Dataset**: Collect unstructured data related to financial fraud.
2. **Create Embeddings**: Generate embeddings from the dataset.
3. **Ingest Embeddings into Model**: Use the embeddings to create an embedding vector space that enables retrieval for detecting similar fraudulent activities.

## Setup Steps

### 1. Install Python 3.12

Ensure Python 3.12 is installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### 2. Install Ollama for the Model

Ollama is required to pull and use the specific model for this project.

```sh
pip install ollama
ollama pull nomic-embed-text
```
### 3. Install Required Python Packages

Use pip to install the necessary Python packages listed in requirements.txt.

```sh
pip install -r requirements.txt
```

### 4. Add API Key
1. Create a file named `key.txt` in the project root directory.
2. Paste your Groq API key into `key.txt`. This key is needed to authenticate model-related requests.

### 5. Generate Embeddings from Dataset

Run `process_unstructured_dataset.py` to create embeddings from your dataset. This step prepares the vector space for fraud detection.

```sh
python process_unstructured_dataset.py
```

### 5. Test the Model on Fraudulent Transactions

Run `test_fraudulent_transaction.py` to evaluate the accuracy of the agent on detecting fraudulent transactions.

```sh
python process_unstructured_dataset.py
```