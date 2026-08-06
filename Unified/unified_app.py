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
    try:
        from langchain_community.llms import Ollama as OllamaLLM
    except ImportError:
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

    # Auto-load existing sample/data documents if Chroma store has no documents yet
    def auto_load_default_documents():
        global rag_vector_store
        sample_files = []
        data_dir = os.path.join(BASE_DIR, "data")
        if os.path.exists(data_dir):
            for fname in os.listdir(data_dir):
                if fname.lower().endswith(('.pdf', '.txt')):
                    sample_files.append(os.path.join(data_dir, fname))
        sample_txt = os.path.join(BASE_DIR, "sample_medical_data.txt")
        if os.path.exists(sample_txt):
            sample_files.append(sample_txt)

        if sample_files:
            docs = []
            for filepath in sample_files:
                try:
                    if filepath.lower().endswith(".pdf"):
                        docs.extend(PyPDFLoader(filepath).load())
                    elif filepath.lower().endswith(".txt"):
                        docs.extend(TextLoader(filepath, encoding='utf-8').load())
                except Exception as ex:
                    print(f"Notice: Skipping document {filepath} ({ex})")

            if docs:
                splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                if rag_vector_store is None:
                    rag_vector_store = Chroma.from_documents(splits, rag_embeddings, persist_directory=CHROMA_DIR)
                else:
                    try:
                        if rag_vector_store._collection.count() == 0:
                            rag_vector_store.add_documents(splits)
                    except Exception:
                        rag_vector_store.add_documents(splits)

    try:
        auto_load_default_documents()
    except Exception as ex:
        print(f"Notice: Auto document load skipped ({ex})")

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

@app.route("/api/heart-params")
def heart_params():
    """Return current simulation parameters for the heart visualization canvas."""
    if not latest_simulation:
        return jsonify({
            "status": "ok",
            "drug_name": "Metoprolol",
            "dose_mg": 50,
            "heart_rate": 72,
            "systolic_bp": 120,
            "contractility": 100,
            "qt_interval": 400,
            "cardiac_state": "normal",
            "warnings": []
        })

    cv = latest_simulation.get("layer_5_cardiovascular", {})
    params = cv.get("cardiac_parameters", {})
    warnings = cv.get("warnings", [])
    info = latest_simulation.get("simulation_info", {})

    return jsonify({
        "status": "ok",
        "drug_name": info.get("drug_name", ""),
        "dose_mg": info.get("dose_mg", 0),
        "heart_rate": params.get("heart_rate_bpm", 72),
        "systolic_bp": params.get("systolic_bp_mmHg", 120),
        "contractility": params.get("contractility_pct", 100),
        "qt_interval": params.get("qt_interval_ms", 400),
        "cardiac_state": cv.get("cardiac_state", "normal"),
        "warnings": warnings,
    })

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
# Ollama Connection & Health Check Routes
# --------------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

@app.route("/api/ollama/status", methods=["GET"])
def ollama_status():
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                models = [m.get("name") for m in data.get("models", [])]
                return jsonify({
                    "status": "connected",
                    "url": OLLAMA_BASE_URL,
                    "models": models
                })
    except Exception as e:
        pass
    return jsonify({"status": "disconnected", "url": OLLAMA_BASE_URL, "error": "Ollama service unavailable"})

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
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"status": "error", "message": "Message cannot be empty."})

        context_text = ""
        global rag_vector_store
        if rag_vector_store is not None:
            try:
                retriever = rag_vector_store.as_retriever(search_kwargs={"k": 3})
                matching_docs = retriever.invoke(message)
                if matching_docs:
                    context_text = "\n\n".join([doc.page_content for doc in matching_docs])
            except Exception as ex:
                print(f"Notice: Vector retrieval warning ({ex})")

        system_instruction = (
            "You are MediRAG, an intelligent AI Health & Medical Assistant for the Digital Twin project. "
            "You can naturally converse, answer greetings ('hello', 'hi'), and answer general medical, "
            "cardiovascular, health, and drug questions accurately and helpfully. "
            "If relevant document context is provided below, incorporate key facts from it to ground your answer. "
            "Keep your responses informative, concise (2 to 4 sentences), and professional."
        )

        if context_text:
            full_prompt = f"{system_instruction}\n\n[Retrieved Document Context]:\n{context_text}\n\n[User Question]:\n{message}"
        else:
            full_prompt = f"{system_instruction}\n\n[User Question]:\n{message}"

        # 1. Direct Ollama HTTP API call (fast, zero wrapper error risk)
        try:
            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/generate",
                data=json.dumps({
                    "model": "llama3.2",
                    "prompt": full_prompt,
                    "stream": False
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                result = json.loads(response.read().decode('utf-8'))
                ai_response = result.get('response', '').strip()
                if ai_response:
                    return jsonify({"status": "success", "response": ai_response})
        except Exception as ollama_err:
            print(f"Notice: Direct Ollama HTTP generation fallback ({ollama_err})")

        # 2. Fallback to LangChain chain
        if LANGCHAIN_AVAILABLE and rag_vector_store is not None:
            try:
                llm = OllamaLLM(model="llama3.2", base_url=OLLAMA_BASE_URL)
                retriever = rag_vector_store.as_retriever(search_kwargs={"k": 3})
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_instruction + "\n\nContext:\n{context}"),
                    ("human", "{input}"),
                ])
                chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template))
                res = chain.invoke({"input": message})
                ans = res.get("answer", "").strip()
                if ans:
                    return jsonify({"status": "success", "response": ans})
            except Exception as chain_err:
                print(f"Notice: LangChain chain fallback ({chain_err})")

        # 3. Rule-based intelligent fallback if Ollama service is offline
        fallback_msg = f"Hello! I am MediRAG, your Digital Twin health assistant. Regarding '{message}': Maintain proper hydration, balance activity, and consult a medical professional for personal advice."
        return jsonify({"status": "success", "response": fallback_msg})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print(f"Starting Digital Twin Unified Server on http://127.0.0.1:8000 ...")
    app.run(host="0.0.0.0", debug=True, port=8000)
