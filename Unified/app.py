"""
Drug-Heart Digital Twin Simulation
Flask Application Server

Provides:
- Web dashboard at /
- REST API at /api/simulate
- Known drugs list at /api/drugs
"""

from flask import Flask, render_template, jsonify, request, send_file
from simulation.pipeline import run_simulation
import json
import traceback
import os

app = Flask(__name__)

# Store latest simulation results for heart viz sync
latest_simulation = {}


# Pre-defined drug list for the UI
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


@app.route("/heart")
def heart_page():
    """Serve the standalone anatomical heart simulation."""
    heart_path = os.path.join(os.path.dirname(__file__), "heart-simulation.html")
    return send_file(heart_path)


@app.route("/api/drugs")
def get_drugs():
    return jsonify(KNOWN_DRUGS)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    try:
        data = request.get_json()

        drug_name = data.get("drug_name", "Metoprolol")
        dose_mg = float(data.get("dose_mg", 50))
        t_max_hours = float(data.get("t_max_hours", 48))

        patient = {
            "age": int(data.get("age", 45)),
            "weight_kg": float(data.get("weight_kg", 70)),
            "sex": data.get("sex", "male"),
            "heart_disease": bool(data.get("heart_disease", False)),
            "hypertension": bool(data.get("hypertension", True)),
            "diabetes": bool(data.get("diabetes", False)),
            "renal_impairment": bool(data.get("renal_impairment", False)),
        }

        results = run_simulation(drug_name, dose_mg, patient, t_max_hours)

        # Store for heart viz sync
        global latest_simulation
        latest_simulation = results

        return jsonify({"status": "success", "data": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heart-params")
def heart_params():
    """Return current simulation parameters for the heart visualization."""
    if not latest_simulation:
        return jsonify({"status": "no_data"})

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
