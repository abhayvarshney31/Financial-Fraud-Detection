import datetime
import random
import ollama
import pandas as pd
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


CREDIT_TRANSACTION_DATASET = os.path.join(
    os.path.dirname(__file__),
    "structured/credit_card_transactions-ibm_v2.csv")
USERS_DATASET = os.path.join(
    os.path.dirname(__file__),
    "structured/sd254_users.csv")
UNSTRUCTURED_LOGS_STORAGE_TRAINING = os.path.join(
    os.path.dirname(__file__),
    "unstructured/training")
UNSTRUCTURED_LOGS_STORAGE_VALIDATION = os.path.join(
    os.path.dirname(__file__),
    "unstructured/validation")

# Set up logging configuration
logging.basicConfig(
    filename='user_behavior_logs.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

CHAT_MODEL_NAME = "openchat"


def generate_log_entry(user_id, user_type, behavior, initial_context="", chunk_size=10):
    """
    Generates a log entry indicating user behavior in a financial app.

    Args:
        user_id (str): The ID of the user.
        behavior (str): The behavior to log.

    Returns:
        str: A log entry summarizing the behavior.
    """
    log_entries = initial_context
    log_range = random.randint(50, 100)

    for _ in range(0, log_range, chunk_size):
        prompt = (
            f"Continue generating {chunk_size} new, unique log entries for user {user_id} "
            f"of type [{user_type}] showing their behavior: {behavior}. "
            "Each entry should add context to the previous ones. "
            "Maintain network and device information, timestamp each log, and make entries flow naturally and separated by next line.\n\n"
            f"Previous logs:\n{log_entries[-1000:]}"  # Include only the last portion to give context
        )
        output = ollama.generate(model=CHAT_MODEL_NAME, prompt=prompt)
        log_entries += "\n" + output['response'].strip()

    return log_entries


def process_user_logs(user_id, user_type, behaviors):
    """
    Process logs for a specific user.

    Args:
        user_id (str): The ID of the user.
        user_type (str): The type of user ('fraudulent' or 'normal').
        behaviors (dict): Dictionary containing list of possible behaviors for the user and behavior type.

    Returns:
        dict: Log entries for the user.
    """
    log_entries = []
    behavior_limit = 1  # Number of log entries for each user

    for _ in range(behavior_limit):
        log_entry = generate_log_entry(
            user_id, user_type, random.choice(behaviors['behaviors']))
        log_entries.append(log_entry)

    return {'user_id': user_id, 'log_entries': log_entries, 'log_type': behaviors['type']}


def create_user_behavior_logs(users_df, transactions_df, total_users=100):
    user_classes = classify_users(users_df, transactions_df)

    # Calculate number of fraudulent and non-fraudulent users
    num_fraudulent = total_users // 20  # 10% fraudulent
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
        "Made a purch`ase at a grocery store with a loyalty card.",
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
        "Monitored credit scores using the app and took steps to improve it.",
        "Viewed personalized tips to save on monthly expenses.",
        "Used the app to split expenses with friends or family.",
        "Transferred money between personal savings and checking accounts.",
        "Read educational content on retirement savings plans.",
        "Viewed account balances and recent transactions daily.",
        "Used the app's calculator for mortgage or loan estimations.",
        "Transferred funds to a family member as a gift.",
        "Set a spending limit on specific categories to control budget.",
        "Used the app to generate a tax report on interest earned.",
        "Viewed monthly summaries to analyze spending trends.",
        "Updated their profile picture and account details.",
        "Watched a video tutorial on managing debt effectively.",
        "Checked and analyzed investment performance in real-time.",
        "Redeemed points for travel rewards in a partner program.",
        "Set reminders for upcoming bill payments.",
        "Participated in a referral program to earn rewards.",
        "Scanned receipts for cashback or points redemption.",
        "Used app's features to search for nearby ATMs or branches.",
        "Signed up for text alerts on account balance thresholds.",
        "Reviewed spending categories to optimize budget allocation.",
        "Sent feedback about app features or reported minor bugs.",
        "Enrolled in a savings round-up program to grow savings.",
        "Researched student loan repayment options.",
        "Enabled location-based security features for logins.",
        "Registered for online seminars on financial planning.",
        "Explored options for consolidating credit card debt.",
        "Accessed FAQs or support documentation for quick help.",
        "Viewed retirement calculators to estimate future needs.",
        "Linked external accounts for a consolidated view of finances.",
        "Changed PIN or password as part of regular security hygiene.",
        "Set up automatic transfers to a high-yield savings account.",
        "Explored charity donation matching options through the app.",
        "Used app’s budgeting tool to manage holiday spending.",
        "Accessed special offers for credit card holders.",
        "Opted into environmental initiatives to offset transaction carbon footprint.",
        "Applied for a travel insurance policy before a planned trip.",
        "Signed up for a beginner's guide to stock market investments.",
        "Activated an international transaction feature for travel abroad."
    ]


    logs = []

    # Create directories if they don't exist
    os.makedirs(f'${UNSTRUCTURED_LOGS_STORAGE_TRAINING}/fraudulent_ato', exist_ok=True)
    os.makedirs(f'${UNSTRUCTURED_LOGS_STORAGE_TRAINING}/fraudulent_cnp', exist_ok=True)
    os.makedirs(f'${UNSTRUCTURED_LOGS_STORAGE_TRAINING}/good', exist_ok=True)

    # Use ThreadPoolExecutor to process user logs in parallel
    with ThreadPoolExecutor() as executor:
        future_to_user = {
            executor.submit(process_user_logs, user_id, user_classes[user_id],
                            {'type': 'fraud_ato', 'behaviors': fraud_behavior_ato } if user_classes[user_id] == 'fraudulent' and random.choice(['ato', 'cnp']) == 'ato'
                            else {'type': 'fraud_cnp', 'behaviors': fraud_behavior_cnp } if user_classes[user_id] == 'fraudulent'
                            else {'type': 'normal', 'behaviors': normal_behaviors }): user_id
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

    for log in training_logs:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f'{UNSTRUCTURED_LOGS_STORAGE_TRAINING}/good'
        if log['log_type'] == 'fraud_ato':
            folder = f'${UNSTRUCTURED_LOGS_STORAGE_TRAINING}/fraudulent_ato'
        elif log['log_type'] == 'fraud_cnp':
            folder = f'${UNSTRUCTURED_LOGS_STORAGE_TRAINING}/fraudulent_cnp'

        with open(f'{folder}/user_behavior_logs_training_{log["user_id"]}_{timestamp}.txt', 'w') as f:
            f.write(f"User ID: {log['user_id']}\n")
            for entry in log['log_entries']:
                f.write(f"- {entry}\n")
            f.write("\n")

    for log in validation_logs:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f'{UNSTRUCTURED_LOGS_STORAGE_VALIDATION}/good'
        if log['log_type'] == 'fraud_ato':
            folder = f'${UNSTRUCTURED_LOGS_STORAGE_VALIDATION}/fraudulent_ato'
        elif log['log_type'] == 'fraud_cnp':
            folder = f'${UNSTRUCTURED_LOGS_STORAGE_VALIDATION}/fraudulent_cnp'

        with open(f'{folder}/user_behavior_logs_training_{log["user_id"]}_{timestamp}.txt', 'w') as f:
            f.write(f"User ID: {log['user_id']}\n")
            for entry in log['log_entries']:
                f.write(f"- {entry}\n")
            f.write("\n")

    return logs  # Generate user behavior logs for a specified number of users


def classify_users(users_df, transactions_df):
    fraudulent_user_indices = transactions_df[transactions_df['Is Fraud?']
                                              == 'Yes']['User']
    fraudulent_users = set(users_df.loc[fraudulent_user_indices, 'Person'])
    result = {
        person: 'fraudulent' if person in fraudulent_users else 'normal' for person in users_df['Person']}
    return result


if __name__ == "__main__":
    users_df = pd.read_csv(USERS_DATASET)
    transactions_df = pd.read_csv(CREDIT_TRANSACTION_DATASET)
    user_logs = create_user_behavior_logs(
        users_df, transactions_df, total_users=10000)


# isolation forest
# cosine similarity = why is this the best option?
