import logging
import os
from datetime import datetime

# Set up logging for Telegram user interactions
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "user_interactions.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_user(update) -> None:
    """Log details of incoming user interaction from Telegram updates."""
    try:
        user = update.effective_user
        message_text = update.message.text if update.message else "N/A"
        user_info = f"User ID: {user.id}, Username: @{user.username}, Name: {user.first_name} {user.last_name or ''}, Input: '{message_text}'"
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {user_info}")
        logging.info(user_info)
    except Exception as e:
        print(f"Error logging user interaction: {e}")
        logging.error(f"Error logging user interaction: {e}")
