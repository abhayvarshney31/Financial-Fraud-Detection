# Financial Fraud Detection Agent

This project aims to create an agent using a Large Language Model (LLM) that is trained on an unstructured financial fraud dataset. The agent will be capable of generating access control rulesets that can be ingested into IAM roles (Identity and Access Management) services on AWS or Azure.

## Overview

1. **Gather Training Dataset**: Collect unstructured data related to financial fraud.
2. **Create Embeddings**: Generate embeddings from the dataset.
3. **Ingest Embeddings into Model**: Use the embeddings to train the model.
4. **Generate Access Control Rules**: The model will output rules that can be applied per user.

## Setup Steps

### 1. Install Python

Ensure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### 2. Install Ollama for the Model

Ollama is required to pull and use the specific model for this project.

```sh
ollama pull nomic-embed-text
```

### 3. Install Required Python Packages

Use `pip` to install the necessary Python packages.

```sh
pip install pandas openai tiktoken numpy
```