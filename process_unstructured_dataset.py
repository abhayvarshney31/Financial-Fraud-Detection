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


def create_embeddings_batch_ollama(text_batch):
    """
    Creates embeddings for a batch of text data using the Ollama model.

    Args:
        text_batch (list[str]): List of text entries to generate embeddings for.

    Returns:
        list: A list of embeddings generated for each text entry.
    """
    embeddings = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                get_embedding_for_input,
                text) for text in text_batch]
        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating embeddings"):
            embeddings.append(future.result())
    return embeddings


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


def parse_log_file(file_path):
    """
    Reads and anonymizes a log file.

    Args:
        file_path (str): Path to the log file.

    Returns:
        str: An anonymized version of the log content.
    """
    with open(file_path, "r", encoding='latin-1', errors='replace') as file:
        log_content = file.read()
        content = anonymize_log(log_content)
        return content


def parse_log_files(log_paths):
    """
    Parses multiple log files in parallel.

    Args:
        log_paths (str): File path pattern for log files.

    Returns:
        list: A list of anonymized log contents.
    """
    text_data = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(parse_log_file, file_path)
                   for file_path in glob.glob(log_paths)]
        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Parsing log files"):
            text_data.append(future.result())
    return text_data


def prepare_training_data():
    """
    Prepares the training data by reading and anonymizing log files for good
    and fraudulent transactions.

    Returns:
        tuple: A tuple containing lists of good transactions, fraudulent ATO
               transactions, and fraudulent CNP transactions.
    """
    good_transactions = parse_log_files(LOG_PATH_GOOD)
    fraudulent_ato_transactions = parse_log_files(LOG_PATH_FRAUDULENT_ATO)
    fraudulent_cnp_transactions = parse_log_files(LOG_PATH_FRAUDULENT_CNP)
    return good_transactions, fraudulent_ato_transactions, fraudulent_cnp_transactions


def store_embeddings(batch, embeddings):
    """
    Stores embeddings and associated text data in the ChromaDB collection.

    Args:
        batch (list): List of document texts.
        embeddings (list): List of embedding vectors corresponding to the batch.
    """
    ids = [str(idx) for idx in range(len(batch))]
    documents = batch
    collection.add(ids=ids, documents=documents, embeddings=embeddings)


def store_all_embeddings(
        good_transactions,
        fraudulent_ato_transactions,
        fraudulent_cnp_transactions,
        good_transactions_embeddings,
        fraudulent_ato_transactions_embeddings,
        fraudulent_cnp_transactions_embeddings):
    """
    Stores all embeddings in the ChromaDB collection, with ID ranges for each
    category of transaction.

    Args:
        good_transactions (list): List of good transaction texts.
        fraudulent_ato_transactions (list): List of ATO fraudulent transaction texts.
        fraudulent_cnp_transactions (list): List of CNP fraudulent transaction texts.
        good_transactions_embeddings (list): Embeddings for good transactions.
        fraudulent_ato_transactions_embeddings (list): Embeddings for ATO fraudulent transactions.
        fraudulent_cnp_transactions_embeddings (list): Embeddings for CNP fraudulent transactions.
    """
    good_ids_range = (0, len(good_transactions_embeddings))
    ato_ids_range = (
        good_ids_range[1],
        good_ids_range[1] +
        len(fraudulent_ato_transactions_embeddings))
    cnp_ids_range = (
        ato_ids_range[1],
        ato_ids_range[1] +
        len(fraudulent_cnp_transactions_embeddings))

    all_transactions = good_transactions + \
        fraudulent_ato_transactions + fraudulent_cnp_transactions
    all_transactions_embeddings = good_transactions_embeddings + \
        fraudulent_ato_transactions_embeddings + fraudulent_cnp_transactions_embeddings

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
    good_transactions_embeddings = create_embeddings_batch_ollama(
        good_transactions)
    fraudulent_ato_transactions_embeddings = create_embeddings_batch_ollama(
        fraudulent_ato_transactions)
    fraudulent_cnp_transactions_embeddings = create_embeddings_batch_ollama(
        fraudulent_cnp_transactions)

    store_all_embeddings(
        good_transactions,
        fraudulent_ato_transactions,
        fraudulent_cnp_transactions,
        good_transactions_embeddings,
        fraudulent_ato_transactions_embeddings,
        fraudulent_cnp_transactions_embeddings)
