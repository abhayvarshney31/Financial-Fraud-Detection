import os
import time
import chromadb
import glob
import ollama
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

COLLECTION_NAME = "financial_fraud_embeddings_final"
CHAT_MODEL_NAME = "mistral"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
LOG_PATH_GOOD = os.path.join(
    os.path.dirname(__file__),
    "unstructured/testing/good/*.txt")
LOG_PATH_FRAUDULENT_ATO = os.path.join(
    os.path.dirname(__file__),
    "unstructured/testing/fraudulent_ato/*.txt")
LOG_PATH_FRAUDULENT_CNP = os.path.join(
    os.path.dirname(__file__),
    "unstructured/testing/fraudulent_cnp/*.txt")
MANUAL_LOG_PATH_GOOD = os.path.join(
    os.path.dirname(__file__),
    "unstructured/manual-testing/good/*.txt")
MANUAL_LOG_PATH_FRAUDULENT_ATO = os.path.join(
    os.path.dirname(__file__),
    "unstructured/manual-testing/fraudulent_ato/*.txt")
MANUAL_LOG_PATH_FRAUDULENT_CNP = os.path.join(
    os.path.dirname(__file__),
    "unstructured/manual-testing/fraudulent_cnp/*.txt")
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


def parse_log_file(file_path):
    with open(file_path, "r", encoding='latin-1', errors='replace') as file:
        log_content = file.read()
        content = anonymize_log(log_content)
        return content


def parse_log_files(log_paths):
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


def load_id_ranges():
    with open(ID_RANGE_DUMP, "r") as file:
        id_ranges = json.load(file)

    return id_ranges


def determine_is_fraud(input_embedding, id_ranges, top_n=1):
    results = collection.query(
        query_embeddings=[input_embedding],
        n_results=top_n
    )

    index = int(results['ids'][0][0])

    for label, (start, end) in id_ranges.items():
        if start <= index <= end:
            return label != "good"

    raise ValueError("Couldn't find id in range")


def get_embedding_for_input(text):
    response = ollama.embeddings(model=EMBEDDING_MODEL_NAME, prompt=text)
    return response['embedding']


def analyze_result(
        good_transactions_logs,
        fraudulent_transactions_logs,
        id_ranges):
    y_true = []
    y_pred = []
    time_diffs = []  # To store time differences for embedding + fraud detection steps

    # Use ThreadPoolExecutor for embedding calculations and fraud detection
    with ThreadPoolExecutor() as executor:
        # Create a list of futures for good transactions
        good_futures = {
            executor.submit(
                get_embedding_for_input,
                log): 0 for log in good_transactions_logs}
        # Create a list of futures for fraudulent transactions
        fraudulent_futures = {
            executor.submit(
                get_embedding_for_input,
                log): 1 for log in fraudulent_transactions_logs}

        # Process good transactions
        for future in tqdm(
                as_completed(
                    good_futures.keys()),
                total=len(good_futures),
                desc="Processing good transactions"):

            # Start time before embedding calculation
            start_time = time.time()
            embeddings = future.result()

            is_fraud = determine_is_fraud(embeddings, id_ranges)
            end_time = time.time()  # End time after fraud detection

            # Calculate time difference and append to list
            time_diffs.append(end_time - start_time)

            y_true.append(0)
            y_pred.append(0 if not is_fraud else 1)

        # Process fraudulent transactions
        for future in tqdm(
                as_completed(
                    fraudulent_futures.keys()),
                total=len(fraudulent_futures),
                desc="Processing fraudulent transactions"):

            # Start time before embedding calculation
            start_time = time.time()
            embeddings = future.result()

            is_fraud = determine_is_fraud(embeddings, id_ranges)
            end_time = time.time()  # End time after fraud detection

            # Calculate time difference and append to list
            time_diffs.append(end_time - start_time)

            y_true.append(1)
            y_pred.append(0 if not is_fraud else 1)

    # Calculate confusion matrix values
    tn, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()

    # Calculate statistical metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Print metrics
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"False Positive Rate: {false_positive_rate:.2f}")

    # Calculate success rate
    total_transactions = len(y_true)
    total_failures = fp + fn
    success_rate = float(total_transactions -
                         total_failures) / total_transactions
    print(f"Success rate: {round(success_rate, 2)}")

    # Calculate and print average time for embedding + fraud detection
    avg_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0
    print(
        f"Average time for embedding calculation and fraud determination: {
            avg_time:.4f} seconds")


def test_fraudulent_transactions(
    good_logs,
    fraudulent_ato_logs,
    fraudulent_cnp_logs
):
    id_ranges = load_id_ranges()
    print("Parsing unstructured logs.")

    good_transactions_logs = parse_log_files(good_logs)
    fraudulent_transactions_logs = parse_log_files(
        fraudulent_ato_logs) + parse_log_files(fraudulent_cnp_logs)

    print("Obtained result. Now proceeding with analyzing the result.")
    analyze_result(
        good_transactions_logs,
        fraudulent_transactions_logs,
        id_ranges)


if __name__ == "__main__":
    print("Executing automated unstructured logs validation")
    test_fraudulent_transactions(
        LOG_PATH_GOOD,
        LOG_PATH_FRAUDULENT_ATO,
        LOG_PATH_FRAUDULENT_CNP)

    print("Executing manual unstructured logs validation")
    test_fraudulent_transactions(
        MANUAL_LOG_PATH_GOOD,
        MANUAL_LOG_PATH_FRAUDULENT_ATO,
        MANUAL_LOG_PATH_FRAUDULENT_CNP)
