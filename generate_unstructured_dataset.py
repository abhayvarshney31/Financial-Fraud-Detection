import datetime
import random
import ollama
import pandas as pd
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


CREDIT_TRANSACTION_USER0_DATASET = os.path.join(
    os.path.dirname(__file__),
    "structured/User0_credit_card_transactions.csv")
USERS_DATASET = os.path.join(
    os.path.dirname(__file__),
    "structured/sd254_users.csv")

# Set up logging configuration
logging.basicConfig(
    filename='user_behavior_logs.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

CHAT_MODEL_NAME = "openchat"


def generate_log_entry(user_id, user_type, behavior):
    """
    Generates a log entry indicating user behavior in a financial app.

    Args:
        user_id (str): The ID of the user.
        behavior (str): The behavior to log.

    Returns:
        str: A log entry summarizing the behavior.
    """
    prompt = f"Generate a 40-70 log entries for user {user_id} for user type [{user_type}] indicating their behavior in a financial app or related activities. Behavior type: {behavior}. Each behavior should include as much context if possible. Make it diverse and interesting where these can be logs are things we track inside webpage/app leading up to the transaction. In these logs, feel free to include network related information and device type. Each log should be timestamped."
    output = ollama.generate(model=CHAT_MODEL_NAME, prompt=prompt)
    return output['response'].strip().replace("\n", "")


def process_user_logs(user_id, user_type, behaviors):
    """
    Process logs for a specific user.

    Args:
        user_id (str): The ID of the user.
        user_type (str): The type of user ('fraudulent' or 'normal').
        behaviors (list): List of possible behaviors for the user.

    Returns:
        dict: Log entries for the user.
    """
    log_entries = []
    behavior_limit = 1  # Number of log entries for each user

    for _ in range(behavior_limit):
        log_entry = generate_log_entry(
            user_id, user_type, random.choice(behaviors))
        log_entries.append(log_entry)

    return {'user_id': user_id, 'log_entries': log_entries}


def create_user_behavior_logs(users_df, transactions_df, total_users=100):
    user_classes = classify_users(users_df, transactions_df)

    # Calculate number of fraudulent and non-fraudulent users
    num_fraudulent = total_users // 10  # 10% fraudulent
    num_non_fraudulent = total_users - num_fraudulent  # Remaining 90% non-fraudulent

    # Separate users by type
    fraudulent_users = [user for user in users_df['Person']
                        if user_classes[user] == 'fraudulent']
    non_fraudulent_users = [
        user for user in users_df['Person'] if user_classes[user] == 'normal']

    # Randomly select the required number of users
    selected_fraudulent_users = random.sample(
        fraudulent_users, min(
            num_fraudulent, len(fraudulent_users)))
    selected_non_fraudulent_users = random.sample(
        non_fraudulent_users, min(
            num_non_fraudulent, len(non_fraudulent_users)))

    selected_users = selected_fraudulent_users + selected_non_fraudulent_users
    # Shuffle to mix fraudulent and non-fraudulent users
    random.shuffle(selected_users)

    # Define behaviors for fraudulent users
    fraud_behavior_ato = [
        "Logged in from multiple devices in a short time period, indicating possible account takeover.",
        "Reported a stolen card but continued to make purchases online.",
        "Received a large refund but immediately transferred funds out.",
        "Attempted to withdraw cash at multiple ATMs in a short time frame.",
        "Made repeated attempts to change account passwords without success.",
        "Used a VPN to mask the login location from different regions.",
        "Changed shipping addresses for online purchases just before delivery.",
        "Tried to access account from different geographical locations within a few hours.",
        "Used an expired credit card to make purchases online.",
        "Engaged in a series of failed login attempts before successfully accessing the account.",
        "Made a series of large withdrawals from different ATMs in a single day.",
        "Attempted to reset account credentials multiple times within minutes.",
        "Suddenly logged in from a foreign country without prior travel notifications."]

    fraud_behavior_cnp = [
        "Conducted several high-value purchases online using stolen credit card information.",
        "Purchased high-ticket items online using different shipping addresses.",
        "Attempted to transfer funds to multiple unverified accounts rapidly.",
        "Used virtual credit cards for transactions without the physical card.",
        "Opened multiple accounts with similar details in a short time.",
        "Used fake identification documents to verify identity for transactions.",
        "Initiated chargebacks on legitimate transactions after receiving goods.",
        "Repeatedly purchased digital goods and services without apparent intent to use.",
        "Engaged in phishing schemes by sending fraudulent emails to gain sensitive information.",
        "Used similar email addresses across different accounts to evade detection.",
        "Purchased electronics with no intention of keeping them, returning items shortly after.",
        "Used stolen identities to create new accounts with financial institutions.",
        "Frequented online gambling sites with large sums and quickly cashed out."]

    # Separate behaviors for normal users
    normal_behaviors = [
        "Made a purchase at a grocery store with a loyalty card.",
        "Browsed financial news articles for investment insights.",
        "Set a savings goal in the app and tracked progress weekly.",
        "Engaged with customer support for account verification.",
        "Consistently used the same device for logging in and making purchases.",
        "Utilized budgeting tools within the app to manage expenses.",
        "Participated in promotional campaigns to earn rewards.",
        "Updated account information regularly to maintain security.",
        "Reviewed transaction history for discrepancies periodically.",
        "Requested assistance to set up two-factor authentication for added security.",
        "Inquired about investment options through the app.",
        "Shared feedback about a recent purchase in the app.",
        "Created a recurring payment schedule for bills and subscriptions.",
        "Attended a financial literacy workshop organized by the app.",
        "Saved receipts digitally for easy tracking of expenses.",
        "Consulted with financial advisors through the app for retirement planning.",
        "Regularly donated to charity through the app's donation feature.",
        "Set up alerts for unusual account activity to stay informed.",
        "Regularly compared insurance rates using the app's comparison tool.",
        "Tracked rewards points and redeemed them for cashback or discounts.",
        "Monitored credit scores using the app and took steps to improve it."]

    logs = []

    # Create directories if they don't exist
    os.makedirs('./unstructured/fraudulent_ato', exist_ok=True)
    os.makedirs('./unstructured/fraudulent_cnp', exist_ok=True)
    os.makedirs('./unstructured/good', exist_ok=True)

    # Use ThreadPoolExecutor to process user logs in parallel
    with ThreadPoolExecutor() as executor:
        future_to_user = {
            executor.submit(process_user_logs, user_id, user_classes[user_id],
                            fraud_behavior_ato if user_classes[user_id] == 'fraudulent' and random.choice(['ato', 'cnp']) == 'ato'
                            else fraud_behavior_cnp if user_classes[user_id] == 'fraudulent'
                            else normal_behaviors): user_id
            for user_id in selected_users
        }

        # Use tqdm to visualize progress
        for future in tqdm(
                as_completed(future_to_user),
                total=len(future_to_user)):
            logs.append(future.result())

    # Split logs into training and validation sets (70/30 split)
    num_training = int(len(logs) * 0.7)
    training_logs = logs[:num_training]
    validation_logs = logs[num_training:]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for log in training_logs:
        folder = './unstructured/good'
        if user_classes[log['user_id']] == 'fraudulent':
            folder = './unstructured/fraudulent_ato' if random.choice(
                ['ato', 'cnp']) == 'ato' else './unstructured/fraudulent_cnp'

        with open(f'{folder}/user_behavior_logs_training_{log["user_id"]}_{timestamp}.txt', 'w') as f:
            f.write(f"User ID: {log['user_id']}\n")
            for entry in log['log_entries']:
                f.write(f"- {entry}\n")
            f.write("\n")

    for log in validation_logs:
        folder = './unstructured/good'
        if user_classes[log['user_id']] == 'fraudulent':
            folder = './unstructured/fraudulent_ato' if random.choice(
                ['ato', 'cnp']) == 'ato' else './unstructured/fraudulent_cnp'

        with open(f'{folder}/user_behavior_logs_validation_{log["user_id"]}_{timestamp}.txt', 'w') as f:
            f.write(f"User ID: {log['user_id']}\n")
            for entry in log['log_entries']:
                f.write(f"- {entry}\n")
            f.write("\n")

    return logs  # Generate user behavior logs for a specified number of users

# Function to classify users as normal or fraudulent


def classify_users(users_df, transactions_df):
    fraudulent_user_indices = transactions_df[transactions_df['Is Fraud?']
                                              == 'Yes']['User']
    fraudulent_users = set(users_df.loc[fraudulent_user_indices, 'Person'])
    result = {
        person: 'fraudulent' if person in fraudulent_users else 'normal' for person in users_df['Person']}
    return result


if __name__ == "__main__":
    users_df = pd.read_csv(USERS_DATASET)
    transactions_df = pd.read_csv(CREDIT_TRANSACTION_USER0_DATASET)
    user_logs = create_user_behavior_logs(
        users_df, transactions_df, total_users=200)
