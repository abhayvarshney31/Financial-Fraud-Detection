import glob
import sys
import ollama
import pandas as pd
import tiktoken
from tqdm import tqdm
import chromadb
import os

# Constants
MODEL = "nomic-embed-text"
persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
UNSTRUCTURED_LOG_PATH = os.path.join(
    os.path.dirname(__file__),
    "unstructured/*training*.txt")
STRUCTURED_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "structured/*.csv")

# Initialize ChromaDB
client = chromadb.Client(chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory
))
collection_name = "financial_fraud_embeddings"

# Check if the collection exists; if not, create it
if collection_name in [c.name for c in client.list_collections()]:
    print("Using existing embedding collection")
    collection = client.get_collection(name=collection_name)
else:
    print("Creating new embedding collection")
    collection = client.create_collection(name=collection_name)

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """
    Count the number of tokens in the provided text using the tokenizer.

    Args:
        text (str): The text to count tokens in.

    Returns:
        int: The token count.
    """
    return len(tokenizer.encode(text))


def prepare_text_from_csv(file_path, chunk_size=1000):
    """
    Read a CSV file in chunks and prepare text data.

    Args:
        file_path (str): Path to the CSV file.
        chunk_size (int): Number of rows to read at a time.

    Yields:
        list: A list of tuples containing row text and its token count.
    """
    print("Reading CSV in chunks at path:", file_path)
    text_data = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            row_text = ", ".join(f"{col}: {val}" for col, val in row.items())
            token_count = count_tokens(row_text)
            text_data.append((row_text, token_count))
            # Process and clear data in batches to reduce memory usage
            if len(text_data) >= chunk_size:
                print("Clearing batch")
                yield text_data
                text_data = []
    if text_data:
        print("Successfully processed CSV")
        yield text_data


def create_embeddings_batch_ollama(text_batch):
    """
    Create embeddings for a batch of text using the Ollama model.

    Args:
        text_batch (list): A list of tuples containing text and its token count.

    Returns:
        list: A list of embeddings corresponding to the input texts.
    """
    embeddings = []
    for text, _ in tqdm(text_batch, desc="Creating embeddings", leave=False):
        response = ollama.embeddings(model=MODEL, prompt=text)
        embeddings.append(response['embedding'])
    return embeddings


def store_embeddings(batch, embeddings):
    """
    Store embeddings in the ChromaDB collection.

    Args:
        batch (list): The batch of text data.
        embeddings (list): The embeddings to be stored.
    """
    ids = []  # List to store IDs
    documents = []  # List to store document texts
    for idx, ((text, _), _) in enumerate(zip(batch, embeddings)):
        ids.append(f"{idx}")  # Use the index as a unique ID
        documents.append(text)  # Add the text to documents
    # Use the appropriate method to add documents
    collection.add(ids=ids, documents=documents, embeddings=embeddings)


def process_data():
    """
    Process data by reading CSV files, creating embeddings, and storing them.
    """
    # This is where you would specify the file paths to process
    for file_path in glob.glob(
            STRUCTURED_CSV_PATH):  # Update to use structured CSV paths
        for text_data in prepare_text_from_csv(file_path):
            batch = []
            for text, token_count in tqdm(
                    text_data, desc=f"Processing {file_path}", leave=False):
                batch.append((text, token_count))
                if len(batch) >= 3:
                    embeddings = create_embeddings_batch_ollama(batch)
                    store_embeddings(batch, embeddings)
                    batch = []
            if batch:
                embeddings = create_embeddings_batch_ollama(batch)
                store_embeddings(batch, embeddings)

    print("Embeddings stored in ChromaDB collection 'financial_fraud_embeddings'")


# Run the process
if __name__ == "__main__":
    process_data()
