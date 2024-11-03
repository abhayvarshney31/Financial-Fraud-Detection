import os
import chromadb
import glob
import ollama
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

COLLECTION_NAME = "financial_fraud_embeddings_final"
CHAT_MODEL_NAME = "gemma2-9b-it"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
LOG_PATH_GOOD = os.path.join(os.path.dirname(__file__), "unstructured/training/good/*.txt")
LOG_PATH_FRAUDULENT_ATO = os.path.join(os.path.dirname(__file__), "unstructured/training/fraudulent_ato/*.txt")
LOG_PATH_FRAUDULENT_CNP = os.path.join(os.path.dirname(__file__), "unstructured/training/fraudulent_cnp/*.txt")
ID_RANGE_DUMP = os.path.join(os.path.dirname(__file__), "id_range.txt")

# Initialize ChromaDB
persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.Client(chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory
))

# Check if the collection exists, if not create it
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    print("Using existing embedding collection")
    collection = client.get_collection(name=COLLECTION_NAME)
else:
    print("Creating new embedding collection")
    collection = client.create_collection(name=COLLECTION_NAME)


def anonymize_log(log_content):
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


# Create embeddings in batch using Ollama API
def create_embeddings_batch_ollama(text_batch):
    embeddings = []
    # Use ThreadPoolExecutor to parallelize embedding creation
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_embedding_for_input, text) for text in text_batch]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating embeddings"):
            embeddings.append(future.result())
    return embeddings


def get_embedding_for_input(text):
    response = ollama.embeddings(model=EMBEDDING_MODEL_NAME, prompt=text)
    return response['embedding']  # Ensure only the embedding is returned


def parse_log_file(file_path):
    with open(file_path, "r", encoding='latin-1', errors='replace') as file:
        log_content = file.read()
        content = anonymize_log(log_content)
        return content


# Parse transactions in parallel
def parse_log_files(log_paths):
    text_data = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(parse_log_file, file_path) for file_path in glob.glob(log_paths)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing log files"):
            text_data.append(future.result())
    return text_data


# Prepare the training data by reading the data and converting the format
def prepare_training_data():
    good_transactions = parse_log_files(LOG_PATH_GOOD)
    fraudulent_ato_transactions = parse_log_files(LOG_PATH_FRAUDULENT_ATO)
    fraudulent_cnp_transactions = parse_log_files(LOG_PATH_FRAUDULENT_CNP)
    return good_transactions, fraudulent_ato_transactions, fraudulent_cnp_transactions


# Store embeddings in the ChromaDB collection
def store_embeddings(batch, embeddings):
    ids = [str(idx) for idx in range(len(batch))]  # List to store IDs
    documents = batch  # List to store document texts
    collection.add(ids=ids, documents=documents, embeddings=embeddings)


def store_all_embeddings(
    good_transactions,
    fraudulent_ato_transactions,
    fraudulent_cnp_transactions,
    good_transactions_embeddings,
    fraudulent_ato_transactions_embeddings,
    fraudulent_cnp_transactions_embeddings):
    
    good_ids_range = (0, len(good_transactions_embeddings))
    ato_ids_range = (good_ids_range[1], good_ids_range[1] + len(fraudulent_ato_transactions_embeddings))
    cnp_ids_range = (ato_ids_range[1], ato_ids_range[1] + len(fraudulent_cnp_transactions_embeddings))
    
    all_transactions = good_transactions + fraudulent_ato_transactions + fraudulent_cnp_transactions
    all_transactions_embeddings = good_transactions_embeddings + fraudulent_ato_transactions_embeddings + fraudulent_cnp_transactions_embeddings

    store_embeddings(all_transactions, all_transactions_embeddings)

    id_ranges = {
        "good": good_ids_range,
        "ato": ato_ids_range,
        "cnp": cnp_ids_range
    }

    # Write ID ranges to a JSON file
    with open(ID_RANGE_DUMP, "w") as file:
        json.dump(id_ranges, file, indent=4)


if __name__ == "__main__":
    good_transactions, fraudulent_ato_transactions, fraudulent_cnp_transactions = prepare_training_data()

    # Generate embeddings in batches
    good_transactions_embeddings = create_embeddings_batch_ollama(good_transactions)
    fraudulent_ato_transactions_embeddings = create_embeddings_batch_ollama(fraudulent_ato_transactions)
    fraudulent_cnp_transactions_embeddings = create_embeddings_batch_ollama(fraudulent_cnp_transactions)

    store_all_embeddings(
        good_transactions,
        fraudulent_ato_transactions,
        fraudulent_cnp_transactions,
        good_transactions_embeddings,
        fraudulent_ato_transactions_embeddings,
        fraudulent_cnp_transactions_embeddings)
