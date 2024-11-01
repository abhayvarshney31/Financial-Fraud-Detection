import os
import chromadb
from groq import Groq
import glob
import ollama
import json

COLLECTION_NAME = "financial_fraud_embeddings_final"
CHAT_MODEL_NAME = "Llama3-8b-8192"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
GROQ_API_KEY = os.path.join(os.path.dirname(__file__), "key.txt")
LOG_PATH_GOOD = os.path.join(os.path.dirname(__file__), "unstructured/testing/good/*.txt")
LOG_PATH_FRAUDULENT_ATO = os.path.join(os.path.dirname(__file__), "unstructured/testing/fraudulent_ato/*.txt")
LOG_PATH_FRAUDULENT_CNP = os.path.join(os.path.dirname(__file__), "unstructured/testing/fraudulent_cnp/*.txt")
ID_RANGE_DUMP = os.path.join(os.path.dirname(__file__), "id_range.txt")

# Read API key
with open(GROQ_API_KEY, 'r') as file:
    api_key = file.read().strip()

# Initialize Groq for chat
groq_client = Groq(api_key=api_key)  # Make sure to set your API key

persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.Client(chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory
))
collection = client.get_collection(name=COLLECTION_NAME)


def anonymize_log(log_content):
    messages = [
        {
            "role": "system", 
            "content": f"""Anonymize this string so there's no pii remaining - for example, if you see a name, replace it with [name]. Also replace all pronouns like his or her (or any other pronouns like that) to they. Then, convert
            summarize these logs to behaviors. I'll be using this for embeddings so make sure to appropriately label the data. The output result format should be the following: 
            Type: <whether it is a fraud - the type - or not> ,
            User: <name>,
            Behavior:
            John Adams updated her email address to johnadams2@gmail.com
            John Adams updated 
            Simply just output the result and nothing else. Here is the string: {log_content}"""
        }
    ]

    # Call Groq to generate text
    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model=CHAT_MODEL_NAME,  # pick another model that the one we used to create test data
    )

    # Assuming the response contains the generated text
    output = chat_completion.choices[0].message.content.strip()

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
