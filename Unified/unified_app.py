import os
import json
import traceback
import tempfile
import urllib.request
import numpy as np

from flask import Flask, render_template, jsonify, request, send_file, Response
import werkzeug.utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# --------------------------------------------------------------------------------
# Phase 1 Imports
# --------------------------------------------------------------------------------
from simulation.pipeline import run_simulation

# --------------------------------------------------------------------------------
# Phase 2 Imports & Mocks
# --------------------------------------------------------------------------------
try:
    import joblib
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import custom_object_scope
except ImportError:
    joblib = None
    tf = None
    load_model = None
    custom_object_scope = None

MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

ensemble_model = None
scaler = None
lstm_model = None
config = {"name": "Patient", "age": 45, "gender": "male", "height_cm": 175, "weight_kg": 75, "fitness_level": "average", "resting_hr": 70}

try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)

    if joblib and os.path.exists(os.path.join(MODELS_DIR, "ensemble_model.joblib")):
        ensemble_model = joblib.load(os.path.join(MODELS_DIR, "ensemble_model.joblib"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

    if tf and os.path.exists(os.path.join(MODELS_DIR, "lstm_model.h5")):
        def heteroscedastic_loss(y_true, y_pred):
            mean, variance = tf.split(y_pred, num_or_size_splits=2, axis=-1)
            precision = 1. / variance
            return tf.reduce_mean(precision * (y_true - mean) ** 2 + tf.math.log(variance))

        with custom_object_scope({'heteroscedastic_loss': heteroscedastic_loss}):
            lstm_model = load_model(os.path.join(MODELS_DIR, "lstm_model.h5"))
except Exception as e:
    print(f"Warning: Could not load Phase 2 models in unified_app.py: {e}")

from feature_engine import build_feature_vector, get_activity_name, estimate_heart_rate, classify_activity

# --------------------------------------------------------------------------------
# Phase 3 Imports (With Non-Deprecated Embeddings Loader & Disk Persistence)
# --------------------------------------------------------------------------------
rag_vector_store = None
rag_embeddings = None
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaLLM
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    rag_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(CHROMA_DIR):
        try:
            rag_vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=rag_embeddings)
        except Exception as ex:
            print(f"Notice: Initializing fresh Chroma store ({ex})")

    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Missing Langchain packages for Phase 3: {e}")
    LANGCHAIN_AVAILABLE = False

# --------------------------------------------------------------------------------
# Flask App Setup
# --------------------------------------------------------------------------------
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, "data")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

latest_simulation = {}

KNOWN_DRUGS = [
    {"name": "Metoprolol", "category": "Beta Blocker", "typical_dose": 50, "max_dose": 200},
    {"name": "Propranolol", "category": "Beta Blocker", "typical_dose": 40, "max_dose": 320},
    {"name": "Atenolol", "category": "Beta Blocker", "typical_dose": 50, "max_dose": 100},
    {"name": "Amlodipine", "category": "Calcium Channel Blocker", "typical_dose": 5, "max_dose": 10},
    {"name": "Verapamil", "category": "Calcium Channel Blocker", "typical_dose": 80, "max_dose": 480},
    {"name": "Diltiazem", "category": "Calcium Channel Blocker", "typical_dose": 60, "max_dose": 360},
    {"name": "Nifedipine", "category": "Calcium Channel Blocker", "typical_dose": 30, "max_dose": 120},
    {"name": "Lisinopril", "category": "ACE Inhibitor", "typical_dose": 10, "max_dose": 80},
    {"name": "Captopril", "category": "ACE Inhibitor", "typical_dose": 25, "max_dose": 150},
    {"name": "Enalapril", "category": "ACE Inhibitor", "typical_dose": 10, "max_dose": 40},
    {"name": "Losartan", "category": "ARB", "typical_dose": 50, "max_dose": 100},
    {"name": "Amiodarone", "category": "Antiarrhythmic", "typical_dose": 200, "max_dose": 800},
    {"name": "Sotalol", "category": "Antiarrhythmic", "typical_dose": 80, "max_dose": 320},
    {"name": "Digoxin", "category": "Cardiac Glycoside", "typical_dose": 0.25, "max_dose": 0.5},
    {"name": "Warfarin", "category": "Anticoagulant", "typical_dose": 5, "max_dose": 15},
    {"name": "Aspirin", "category": "Antiplatelet", "typical_dose": 100, "max_dose": 600},
    {"name": "Furosemide", "category": "Diuretic", "typical_dose": 40, "max_dose": 600},
    {"name": "Doxorubicin", "category": "Cardiotoxic Chemo", "typical_dose": 60, "max_dose": 150},
    {"name": "Caffeine", "category": "Stimulant", "typical_dose": 200, "max_dose": 1000},
    {"name": "Ibuprofen", "category": "NSAID", "typical_dose": 400, "max_dose": 3200},
    {"name": "Acetaminophen", "category": "Analgesic", "typical_dose": 500, "max_dose": 4000},
]

@app.route("/")
def index():
    return render_template("index.html")

# --------------------------------------------------------------------------------
# Phase 1 Routes
# --------------------------------------------------------------------------------
@app.route("/heart")
def heart_page():
    heart_path = os.path.join(BASE_DIR, "heart-simulation.html")
    return send_file(heart_path)

@app.route("/api/drugs")
def get_drugs():
    return jsonify(KNOWN_DRUGS)

@app.route("/api/simulate", methods=["POST"])
def simulate():
    try:
        data = request.get_json() or {}
        drug_name = data.get("drug_name", "Metoprolol")
        dose_mg = float(data.get("dose_mg", 50))
        secondary_drug_name = data.get("secondary_drug_name", None)
        secondary_dose_mg = float(data.get("secondary_dose_mg", 0.0))
        t_max_hours = float(data.get("t_max_hours", 48))
        
        patient = {
            "age": max(1, min(120, int(data.get("age", 45)))),
            "weight_kg": max(1.0, min(300.0, float(data.get("weight_kg", 70)))),
            "sex": data.get("sex", "male"),
            "heart_disease": bool(data.get("heart_disease", False)),
            "hypertension": bool(data.get("hypertension", True)),
            "diabetes": bool(data.get("diabetes", False)),
            "renal_impairment": bool(data.get("renal_impairment", False)),
            "egfr": float(data.get("egfr", 90.0)),
            "potassium_mM": float(data.get("potassium_mM", 4.0)),
            "baseline_qtc_ms": float(data.get("baseline_qtc_ms", 410.0)),
        }

        results = run_simulation(
            drug_name, dose_mg, patient, t_max_hours,
            secondary_drug_name=secondary_drug_name,
            secondary_dose_mg=secondary_dose_mg
        )
        global latest_simulation
        latest_simulation = results
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# --------------------------------------------------------------------------------
# Phase 2 Routes
# --------------------------------------------------------------------------------
@app.route("/api/chat-activity", methods=["POST"])
def chat_activity():
    try:
        data = request.get_json() or {}
        query = data.get("query", "")
        
        activity_name = get_activity_name(query)
        estimated_hr = estimate_heart_rate(classify_activity(query), config)
        
        features = build_feature_vector(query, config)
        if ensemble_model and scaler and lstm_model:
            features_scaled = scaler.transform([features])
            pred_ensemble = float(ensemble_model.predict(features_scaled)[0])
            
            X_seq = np.tile(features_scaled, (10, 1))[np.newaxis, :, :]
            lstm_raw = lstm_model.predict(X_seq, verbose=0)
            pred_lstm = float(lstm_raw[0].flatten()[0]) if isinstance(lstm_raw, (list, tuple)) else float(lstm_raw.flatten()[0])
        else:
            pred_ensemble = float(estimated_hr + 2.5)
            pred_lstm = float(estimated_hr + 4.0)

        prompt = (
            f"You are a health AI assistant for a Digital Twin Heart project. "
            f"A user asked about the activity: '{query}'. "
            f"The ML predicted HR is approx {pred_ensemble:.1f} bpm. "
            f"Generate a short, user-friendly, plain text response explaining what this means. "
            f"Keep it under 3 sentences."
        )
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=json.dumps({"model": "llama3.2", "prompt": prompt, "stream": False}).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                llm_response = result.get('response', '')
        except Exception:
            llm_response = f"Predicted heart rate for '{activity_name}' is around {pred_ensemble:.1f} bpm. Maintain hydration and monitor intensity."

        return jsonify({
            "status": "success", 
            "activity_name": activity_name,
            "pred_ensemble": pred_ensemble,
            "pred_lstm": pred_lstm,
            "estimated_hr": estimated_hr,
            "response": llm_response.strip()
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# --------------------------------------------------------------------------------
# Dashboard Routes
# --------------------------------------------------------------------------------
@app.route("/api/dashboard", methods=["GET"])
def get_dashboard_data():
    try:
        def load_json(filename):
            candidates = [
                os.path.join(ROOT_DIR, filename),
                os.path.join(BASE_DIR, filename),
                os.path.join(BASE_DIR, "data", filename)
            ]
            for path in candidates:
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            return json.load(f)
                    except Exception:
                        pass
            return None

        sleep_data = load_json("sleep_response.json") or {"summary": "7h 45m deep sleep", "score": 88}
        spo2_data = load_json("spo2_response.json") or {"average": 98.2, "min": 95}
        heart_rate_data = load_json("heart_rates_response.json") or {"resting_hr": 68, "max_hr": 142}

        return jsonify({
            "status": "success",
            "data": {
                "sleep": sleep_data,
                "spo2": spo2_data,
                "heart_rate": heart_rate_data
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# --------------------------------------------------------------------------------
# Phase 3 Routes (With Disk Persistence & Streaming)
# --------------------------------------------------------------------------------
@app.route("/api/rag/upload", methods=["POST"])
def rag_upload():
    if not LANGCHAIN_AVAILABLE:
        return jsonify({"status": "error", "message": "MediRAG engine is unavailable due to missing dependencies."})
    try:
        global rag_vector_store
        files = request.files.getlist("files")
        if not files: return jsonify({"status": "error", "message": "No files provided."})

        documents = []
        for f in files:
            filename = werkzeug.utils.secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)
            
            if filepath.lower().endswith(".pdf"):
                documents.extend(PyPDFLoader(filepath).load())
            elif filepath.lower().endswith(".txt"):
                documents.extend(TextLoader(filepath, encoding='utf-8').load())
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
        
        if rag_vector_store is None:
            rag_vector_store = Chroma.from_documents(splits, rag_embeddings, persist_directory=CHROMA_DIR)
        else:
            rag_vector_store.add_documents(splits)

        return jsonify({"status": "success", "message": f"Processed and saved {len(files)} file(s) into Chroma persistent store."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/rag/chat", methods=["POST"])
def rag_chat():
    if not LANGCHAIN_AVAILABLE:
        return jsonify({"status": "error", "message": "MediRAG engine is unavailable."})
    global rag_vector_store
    try:
        data = request.get_json() or {}
        message = data.get("message", "")
        
        if rag_vector_store is None:
            return jsonify({"status": "error", "message": "Please upload a medical document first."})
            
        llm = OllamaLLM(model="llama3.2")
        retriever = rag_vector_store.as_retriever(search_kwargs={"k": 3})
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template))
        result = chain.invoke({"input": message})
        
        return jsonify({"status": "success", "response": result.get("answer", "No response generated.")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print(f"Starting Digital Twin Unified Server on http://127.0.0.1:8000 ...")
    app.run(host="0.0.0.0", debug=True, port=8000)
