import ollama
import chromadb
import numpy as np
import os

# Constants
MODEL = "nomic-embed-text"
collection_name = "financial_fraud_embeddings_expirement2"
persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")

# Initialize ChromaDB Client
client = chromadb.Client(chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory
))
if collection_name in [c.name for c in client.list_collections()]:
    print("Using existing embedding collection")
    collection = client.get_collection(name=collection_name)
else:
    print("Creating new embedding collection")
    collection = client.create_collection(name=collection_name)

# Helper function to embed new input
def get_embedding_for_input(text):
    response = ollama.embeddings(model=MODEL, prompt=text)
    return response['embedding']  # Ensure only the embedding is returned

# Function to calculate fraud score
def calculate_fraud_score(input_embedding, threshold=0.8, top_n=1):
    # Retrieve similar embeddings from the collection
    results = collection.query(
        query_embeddings=[input_embedding],
        n_results=top_n
    )
    
    # Extract distances from results
    distances = results['distances'][0]
    
    # Calculate similarity scores by inverting distances (lower distance means higher similarity)
    similarities = [1 / (1 + distance) for distance in distances]
    
    # Calculate the fraud score as the average similarity
    fraud_score = sum(similarities) / len(similarities)
    
    # Determine if the transaction is likely fraudulent based on the threshold
    is_fraud = fraud_score >= threshold
    
    return {
        "fraud_score": fraud_score,
        "is_fraud": is_fraud
    }

# Create embeddings in batch using Ollama API
def create_embeddings_batch_ollama(text_batch):
    embeddings = []
    for text in text_batch:
        embedding = get_embedding_for_input(text)
        embeddings.append(embedding)  # Store only the embedding (numeric vector)
    return embeddings

# Store embeddings in the ChromaDB collection
def store_embeddings(batch, embeddings):
    ids = [str(idx) for idx in range(len(batch))]  # List to store IDs
    documents = batch  # List to store document texts
    collection.add(ids=ids, documents=documents, embeddings=embeddings)

# Example usage
def main():
    batch = [
        """
        Type: Account Takeover,
        User: [name],
        Behavior:  
            - [name] updated her email address to [email].  
            - [name] updated her phone number to [phone number].  
            - [name] attempted multiple unsuccessful logins, prompting email verification.  
            - [name] received a verification email but did not confirm her email address.  
            - [name] attempted to log in multiple times without completing email verification, prompting repeated notifications.  
            - [name] updated her address to 462 Rose Lane, La Verne, CA 91750, without completing email verification.  
            - [name] attempted to update her credit card information multiple times, encountering errors (invalid expiration date, credit limit exceeded, and invalid card number).  
            - [name] was prompted to update credit card information upon viewing her account summary.  
            - [name] viewed her payment history and selected a new payment method for a transaction.
        """,
        """
        Type: Transaction Preparation,  
        User: [name],
        Behavior:  
            - [name] frequently viewed and modified her transaction history visibility settings, adjusting filters for date ranges and transaction amounts.  
            - [name] initially viewed transaction history for the last 30 days and then repeatedly adjusted the range to 60 days, sometimes filtering for transactions over $50 or $100.  
            - [name] changed settings multiple times to hide all transactions, view transactions exceeding certain amounts, and restore original filters.  
            - [name]’s actions involved viewing transaction history both with and without filters, sometimes specifying a specific debit card ending in 4344.  
            - Leading up to the transaction, [name] had a filter set for viewing the last 30 days of transaction history.
        """,
        """        
            - [Name] did update her email address to [email].  
            - [Name] did update her phone number to [phone number].  
        """
    ]
    
    # Generate and store embeddings
    embeddings = create_embeddings_batch_ollama(batch)
    store_embeddings(batch, embeddings)
    
    # Test with new log entry embeddings
    input_text_fraud = """ 
        - [name] frequently viewed and modified her transaction history visibility settings, adjusting filters for date ranges and transaction amounts.  
        - [name] did updated her phone number to [phone number]. 
        - [name] updated her email address to [email].  
    """
    
    input_embedding_fraud = get_embedding_for_input(input_text_fraud)
    
    # Calculate fraud score
    fraud_score_result = calculate_fraud_score(input_embedding_fraud)
    
    print("Fraud Score Results:")
    print(f"Fraud Risk Percentage: {fraud_score_result['fraud_score'] * 100}%")
    print(f"Is Fraudulent: {'Yes' if fraud_score_result['is_fraud'] else 'No'}")

if __name__ == "__main__":
    main()
