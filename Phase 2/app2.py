import joblib
import numpy as np
import os
import sys
import requests
import json
import asyncio
import nest_asyncio

# Ensure Unified directory is accessible for feature_engine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNIFIED_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Unified"))
if UNIFIED_DIR not in sys.path:
    sys.path.insert(0, UNIFIED_DIR)

# Import local modules
from tracker import log_user
from feature_engine import build_feature_vector, get_activity_name, estimate_heart_rate, classify_activity

# Import TensorFlow safely
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import custom_object_scope
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

nest_asyncio.apply()

# --- Constants & Path Resolution ---
SEQUENCE_LENGTH = 10
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8298994512:AAGtxHM33SCChhNW5KUdWOo8GgjJQ37eM9Q")

# --- Load Config and Models ---
config = {"name": "Patient", "age": 45, "gender": "male", "height_cm": 175, "weight_kg": 75, "fitness_level": "average", "resting_hr": 70}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

xgb_model = None
lgb_model = None
rf_model = None
ensemble_model = None
scaler = None
lstm_model = None

try:
    if os.path.exists(os.path.join(MODELS_DIR, "ensemble_model.joblib")):
        xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
        lgb_model = joblib.load(os.path.join(MODELS_DIR, "lgb_model.joblib"))
        rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib"))
        ensemble_model = joblib.load(os.path.join(MODELS_DIR, "ensemble_model.joblib"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

    if TF_AVAILABLE and os.path.exists(os.path.join(MODELS_DIR, "lstm_model.h5")):
        def heteroscedastic_loss(y_true, y_pred):
            mean, variance = tf.split(y_pred, num_or_size_splits=2, axis=-1)
            precision = 1. / variance
            return tf.reduce_mean(precision * (y_true - mean) ** 2 + tf.math.log(variance))

        with custom_object_scope({'heteroscedastic_loss': heteroscedastic_loss}):
            lstm_model = load_model(os.path.join(MODELS_DIR, "lstm_model.h5"))
except Exception as e:
    print(f"Warning: Could not load some ML models in Phase 2: {e}")

# --- Prediction Functions ---
def make_prediction(model, features):
    """Predict using a scikit-learn model with proper feature scaling."""
    if model is None or scaler is None:
        return [115.0]
    features_scaled = scaler.transform([features])
    return model.predict(features_scaled)

def predict_lstm(features):
    """Predict using the LSTM model, handling dual-output (mean, log_variance)."""
    if lstm_model is None or scaler is None:
        return 117.5
    features_scaled = scaler.transform([features])
    X_seq = np.tile(features_scaled, (SEQUENCE_LENGTH, 1))[np.newaxis, :, :]
    lstm_raw = lstm_model.predict(X_seq, verbose=0)
    if isinstance(lstm_raw, (list, tuple)):
        return float(lstm_raw[0].flatten()[0])
    return float(lstm_raw.flatten()[0])

def generate_response_with_ollama(prediction, lstm_prediction, user_query, activity_name, estimated_hr):
    """Generate a natural language response using Ollama (LLaMA 3.2)."""
    prompt = (
        f"You are a health AI assistant for a Digital Twin Heart project. "
        f"A user asked about the activity: '{user_query}'. "
        f"The activity was classified as: {activity_name}. "
        f"Based on the user's profile and our ML models:\n"
        f"- Ensemble model predicted heart rate: {prediction:.1f} bpm\n"
        f"- LSTM model predicted heart rate: {lstm_prediction:.1f} bpm\n"
        f"- Estimated baseline HR for this activity (Karvonen formula): {estimated_hr:.1f} bpm\n\n"
        f"User profile: Name={config['name']}, Age={config['age']}, Gender={config['gender']}, "
        f"Height={config['height_cm']}cm, Weight={config['weight_kg']}kg, "
        f"Fitness Level={config['fitness_level']}, Resting HR={config['resting_hr']} bpm.\n\n"
        f"Generate a short, user-friendly, plain text response (no asterisks or special formatting). "
        f"Explain what this heart rate means for the user during this activity. "
        f"Include a brief health tip if relevant. Keep it under 150 words."
    )
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt}, stream=True, timeout=10
        )
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    full_response += chunk.get('response', '')
                except json.JSONDecodeError:
                    pass
        return full_response.strip() if full_response else "(Ollama returned empty response.)"
    except requests.exceptions.ConnectionError:
        return "(Ollama is not running — displaying ML prediction without LLM insight.)"
    except requests.exceptions.Timeout:
        return "(Ollama response timed out — displaying ML prediction.)"

# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    log_user(update)
    await update.message.reply_text(
        f'Hello {update.effective_user.first_name}! I am your Digital Twin Heart Bot.\n\n'
        f'Send me any activity (e.g., "running 1km", "swimming", "sleeping") '
        f'and I\'ll predict your heart rate!'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    log_user(update)
    await update.message.reply_text(
        'Send me any activity to get a heart rate prediction.\n\n'
        'Examples:\n'
        '- "What if I run for 30 minutes?"\n'
        '- "Heart rate while swimming"\n'
        '- "Playing chess"\n'
        '- "Gym workout"\n'
        '- "Sleeping"'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    log_user(update)
    user_query = update.message.text.strip()

    try:
        features = build_feature_vector(user_query, config)
        activity_name = get_activity_name(user_query)
        estimated_hr = estimate_heart_rate(classify_activity(user_query), config)

        pred_ensemble = make_prediction(ensemble_model, features)
        pred_lstm = predict_lstm(features)

        response_text = generate_response_with_ollama(
            pred_ensemble[0], pred_lstm, user_query, activity_name, estimated_hr
        )

        full_report = (
            f"*Activity:* {user_query}\n"
            f"*Classified as:* {activity_name}\n"
            f"*Estimated HR (Karvonen):* {estimated_hr:.1f} bpm\n\n"
            f"*Ensemble Prediction:* {pred_ensemble[0]:.2f} bpm\n"
            f"*LSTM Prediction:* {pred_lstm:.2f} bpm\n\n"
            f"*AI Insight:*\n{response_text}"
        )
        await update.message.reply_text(full_report, parse_mode='Markdown')
    except Exception as e:
        print(f"Error handling message: {e}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text("Sorry, an error occurred while calculating prediction.")

# --- Main Bot Logic ---
async def main() -> None:
    if not TELEGRAM_AVAILABLE:
        print("Error: python-telegram-bot library is missing. Install with 'pip install python-telegram-bot'")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Digital Twin Heart Bot (Telegram integration enabled)")
    print("Bot is polling for updates... User interactions will be logged.")
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    asyncio.run(main())
