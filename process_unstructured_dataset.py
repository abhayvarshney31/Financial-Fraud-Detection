import os
import chromadb
import glob
import ollama
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

COLLECTION_NAME = "financial_fraud_embeddings_final"
CHAT_MODEL_NAME = "gemma2-9b-it"
EMBEDDING_MODEL_NAME = "nomic-embed-text"

# Paths for log files
LOG_PATH_GOOD = os.path.join(
    os.path.dirname(__file__),
    "unstructured/training/good/*.txt")
LOG_PATH_FRAUDULENT_ATO = os.path.join(
    os.path.dirname(__file__),
    "unstructured/training/fraudulent_ato/*.txt")
LOG_PATH_FRAUDULENT_CNP = os.path.join(
    os.path.dirname(__file__),
    "unstructured/training/fraudulent_cnp/*.txt")
ID_RANGE_DUMP = os.path.join(os.path.dirname(__file__), "id_range.txt")

# Initialize ChromaDB
persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.Client(
    chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory))

# Check if collection exists; create if it doesn't
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    print("Using existing embedding collection")
    collection = client.get_collection(name=COLLECTION_NAME)
else:
    print("Creating new embedding collection")
    collection = client.create_collection(name=COLLECTION_NAME)


def anonymize_log(log_content):
    """
    Anonymizes and summarizes the log content to remove sensitive information.

    Args:
        log_content (str): Raw log content that needs to be anonymized.

    Returns:
        str: An anonymized and summarized version of the log content.
    """
    prompt = f"""
                Given the following log content, perform these steps:

                1. Summarize to Behavior: Convert the actions into a behavior summary, formatted to indicate whether it was a fraudulent action or not, along with user behavior. But make sure the summary is as detailed as possible.
                2. Anonymize: Replace any personal information (e.g., names, emails, addresses) with placeholders, like [name], [email]. A name can only be something like "User 0" so be sure to replace that with [name].
                3. Replace Pronouns: Change all gendered pronouns to 'they' to remove gender references.
                4. Remove any next lines symbols. A next line in the string can be in the format of '\\n' or '/n'.

                Here is the log content: {log_content}. Don't output anything besides the string output where each activity is separated by " * "".
            """

    output = ollama.generate(
        model=CHAT_MODEL_NAME,
        prompt=prompt,
    )

    output = output['response'].strip().replace("\n", "")
    return output


def create_combined_embedding(log_text, device_types, timestamps):
    """
    Create an embedding combining text and metadata features.
    """
    # Text-based embedding
    text_embedding = get_embedding_for_input(log_text)

    # Metadata features
    device_embeddings = [get_embedding_for_input(
        device) for device in device_types]
    time_diff_vector = [get_embedding_for_input(
        timestamp) for timestamp in timestamps]

    # Concatenate to form combined embedding
    device_embeddings_flat = np.concatenate(device_embeddings)
    combined_embedding = np.concatenate(
        [text_embedding, device_embeddings_flat, time_diff_vector])
    return combined_embedding.tolist()


def get_embedding_for_input(text):
    """
    Retrieves the embedding for a single text entry using the Ollama API.

    Args:
        text (str): Text entry to generate an embedding for.

    Returns:
        list: The embedding vector corresponding to the text.
    """
    response = ollama.embeddings(model=EMBEDDING_MODEL_NAME, prompt=text)
    return response['embedding']


def extract_ordered_device_types(log_content):
    """
    Extracts ordered device types.

    Args:
        log_content (str): Raw log content.

    Returns:
        dict: Dictionary with extracted device, timestamp, and other features.
    """
    prompt = f"""
    Given the following log content, extract the type of devices used and pair it with the appropriate timestamp.
    Don't lose order. Here is the log content: {log_content}. Return as a list of strings containing device type names.
    """

    response = ollama.generate(model=CHAT_MODEL_NAME, prompt=prompt)
    return json.loads(response['response'])


def extract_ordered_timestamps(log_content):
    """
    Extracts ordered device types.

    Args:
        log_content (str): Raw log content.

    Returns:
        dict: Dictionary with extracted device, timestamp, and other features.
    """
    prompt = f"""
    Given the following log content, extract all the timestamps. Try to extract to format 'YYYY-MM-DD HH:MM:SS'.
    Don't lose order. Here is the log content: {log_content}. Return as a list of timestamp strings.
    """

    response = ollama.generate(model=CHAT_MODEL_NAME, prompt=prompt)
    return json.loads(response['response'])


def parse_log_file(file_path, user_type):
    """
    Reads and anonymizes a log file.

    Args:
        file_path (str): Path to the log file.

    Returns:
        str: An anonymized version of the log content.
    """
    with open(file_path, "r", encoding='latin-1', errors='replace') as file:
        log_content = file.read()
    anonymized_text = anonymize_log(log_content)

    # Extract metadata (replace with actual extraction logic)
    device_types = extract_ordered_device_types(log_content)
    timestamps = extract_ordered_timestamps(log_content)

    return anonymized_text, device_types, timestamps, user_type


def parse_log_files(log_paths, user_type):
    """
    Parses multiple log files to extract anonymized content and metadata.
    """
    log_data = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                parse_log_file,
                file_path,
                user_type) for file_path in glob.glob(log_paths)]
        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parsing log files"):
            log_data.append(future.result())
    return log_data


def store_embeddings_with_metadata(log_data):
    """
    Stores embeddings and metadata in the ChromaDB collection.
    """
    ids, documents, embeddings, metadatas = [], [], [], []

    for idx, (log_text, device_types, timestamps,
              user_type) in enumerate(log_data):
        embedding = create_combined_embedding(
            log_text, device_types, timestamps)

        ids.append(str(idx))
        documents.append(log_text)
        embeddings.append(embedding)
        metadatas.append({"device_types": device_types,
                         "timestamps": timestamps, "user_type": user_type})

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas)


if __name__ == "__main__":
    # Parse and process log files
    good_logs = parse_log_files(LOG_PATH_GOOD, "good")
    ato_logs = parse_log_files(LOG_PATH_FRAUDULENT_ATO, "ato_fraud")
    cnp_logs = parse_log_files(LOG_PATH_FRAUDULENT_CNP, "cnp_fraud")

    # Combine all log data for storing
    all_log_data = good_logs + ato_logs + cnp_logs

    # Store embeddings with metadata in ChromaDB
    store_embeddings_with_metadata(all_log_data)
