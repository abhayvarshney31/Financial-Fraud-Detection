import os
import chromadb
import glob
import ollama
import json

COLLECTION_NAME = "financial_fraud_embeddings_final"
CHAT_MODEL_NAME = "mistral"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
LOG_PATH_GOOD = os.path.join(os.path.dirname(__file__), "unstructured/testing/good/*.txt")
LOG_PATH_FRAUDULENT_ATO = os.path.join(os.path.dirname(__file__), "unstructured/testing/fraudulent_ato/*.txt")
LOG_PATH_FRAUDULENT_CNP = os.path.join(os.path.dirname(__file__), "unstructured/testing/fraudulent_cnp/*.txt")
ID_RANGE_DUMP = os.path.join(os.path.dirname(__file__), "id_range.txt")

persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")

client = chromadb.Client(settings=chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory
))
collection = client.get_collection(name=COLLECTION_NAME)


def anonymize_log(log_content):
    prompt = f"""
                Given the following log content, perform these steps:
                
                1. Summarize to Behavior: Convert the actions into a behavior summary, formatted to indicate whether it was a fraudulent action or not, along with user behavior. But make sure the summary is as detailed as possible.
                2. Anonymize: Replace any personal information (e.g., names, emails, addresses) with placeholders, like [name], [email]. A name can only be something like "User 0" so be sure to replace that with [name].
                3. Replace Pronouns: Change all gendered pronouns to 'they' to remove gender references.
                4. Remove any next lines symbols. A next line in the string can be in the format of '\\n' or '/n'.
                
                Here is the log content: {log_content}. Don't output anything besides the string output where each activity is separated by " * "".
            """

    # Call Ollama to generate text
    output = ollama.generate(
        model=CHAT_MODEL_NAME,
        prompt=prompt,
    )

    output = output['response'].strip().replace("\n", "")
    return output



# Parse transactions and label them
def parse_log_files(log_paths):
    text_data = []
    for file_path in glob.glob(log_paths):
        with open(file_path, "r", encoding='latin-1', errors='replace') as file:
            log_content = file.read()  # Read entire file content at once
            content = anonymize_log(log_content)
            text_data.append(content)
    return text_data


def load_id_ranges():
    # Open the JSON file and load the data
    with open(ID_RANGE_DUMP, "r") as file:
        id_ranges = json.load(file)
    
    return id_ranges


# Function to calculate fraud score
def determine_is_fraud(input_embedding, id_ranges, top_n=1):
    # Retrieve similar embeddings from the collection
    results = collection.query(
        query_embeddings=[input_embedding],
        n_results=top_n
    )
    
    index = int(results['ids'][0][0])
    
    for label, (start, end) in id_ranges.items():
        if index >= start or index <= end:
            return label != "good"
    
    raise ValueError("Couldn't find id in range")


# Helper function to embed new input
def get_embedding_for_input(text):
    response = ollama.embeddings(model=EMBEDDING_MODEL_NAME, prompt=text)
    return response['embedding']  # Ensure only the embedding is returned


if __name__ == "__main__":
    id_ranges = load_id_ranges()
    good_transactions_logs = parse_log_files(LOG_PATH_GOOD)
    fraudulent_transactions_logs = parse_log_files(LOG_PATH_FRAUDULENT_ATO) +  parse_log_files(LOG_PATH_FRAUDULENT_CNP)

    total_failures = 0
    total_transactions = len(good_transactions_logs) + len(fraudulent_transactions_logs)
    
    # validate good transactions
    for good_transaction_logs in good_transactions_logs:
        embeddings = get_embedding_for_input(good_transaction_logs)
        is_fraud = determine_is_fraud(embeddings, id_ranges)
        if is_fraud:
            total_failures += 1
    
    # validate good transactions
    for fraudulent_transaction_logs in fraudulent_transactions_logs:
        embeddings = get_embedding_for_input(fraudulent_transaction_logs)
        is_fraud = determine_is_fraud(embeddings, id_ranges)
        if not is_fraud:
            total_failures += 1
    
    success_rate = float(total_transactions - total_failures) / total_transactions
    print(f"Success rate: {round(success_rate, 2)}")
