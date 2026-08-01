"""
Pipeline orchestrator: Runs all 7 layers sequentially with support for Polypharmacy and Advanced Physiology parameters.

Drug Input → Representation → Target Interaction → PK → PD → CV Model → ML → Digital Twin Output
"""

from simulation.drug_representation import represent_drug
from simulation.drug_target import predict_drug_target_interactions
from simulation.pharmacokinetics import compute_pk
from simulation.pharmacodynamics import compute_pd
from simulation.cardiovascular import simulate_cardiovascular
from simulation.ml_predictions import run_ml_predictions
import numpy as np
import time


def run_simulation(drug_name: str, dose_mg: float, patient: dict = None,
                   t_max_hours: float = 48.0,
                   secondary_drug_name: str = None,
                   secondary_dose_mg: float = 0.0) -> dict:
    """
    Run the complete 7-layer Drug-Heart simulation pipeline.

    Supports Polypharmacy (combination therapy) and physiological parameters:
    - egfr: Renal clearance rate (default: 90 mL/min)
    - potassium_mM: Serum potassium concentration (default: 4.0 mM)
    - baseline_qtc_ms: Baseline QTc interval (default: 410 ms)
    """
    start_time = time.time()

    if patient is None:
        patient = {}

    patient_profile = {
        "age": int(patient.get("age", 45)),
        "weight_kg": float(patient.get("weight_kg", 70)),
        "sex": patient.get("sex", "male"),
        "heart_disease": bool(patient.get("heart_disease", False)),
        "hypertension": bool(patient.get("hypertension", True)),
        "diabetes": bool(patient.get("diabetes", False)),
        "renal_impairment": bool(patient.get("renal_impairment", False)),
        "egfr": float(patient.get("egfr", 90.0)),
        "potassium_mM": float(patient.get("potassium_mM", 4.0)),
        "baseline_qtc_ms": float(patient.get("baseline_qtc_ms", 410.0)),
    }

    layer_times = {}

    # ═══════════════════════════════════════════
    # Layer 1: Drug Representation (Primary + Secondary)
    # ═══════════════════════════════════════════
    t0 = time.time()
    drug_repr = represent_drug(drug_name)
    
    polypharmacy_active = bool(secondary_drug_name and secondary_dose_mg > 0 and secondary_drug_name != "None")
    secondary_repr = None
    if polypharmacy_active:
        secondary_repr = represent_drug(secondary_drug_name)

    layer_times["drug_representation"] = round(time.time() - t0, 4)

    # ═══════════════════════════════════════════
    # Layer 2: Pharmacokinetics (PK)
    # ═══════════════════════════════════════════
    t0 = time.time()
    clearance_factor = max(0.2, min(2.0, patient_profile["egfr"] / 90.0))
    if patient_profile["renal_impairment"]:
        clearance_factor *= 0.65

    pk_results = compute_pk(
        drug_name, dose_mg, drug_repr["properties"],
        patient_weight_kg=patient_profile["weight_kg"],
        t_max_hours=t_max_hours
    )
    if clearance_factor != 1.0:
        pk_results["one_compartment"]["cmax"] *= (1.0 / clearance_factor)
        pk_results["one_compartment"]["auc"] *= (1.0 / clearance_factor)

    secondary_pk = None
    if polypharmacy_active:
        secondary_pk = compute_pk(
            secondary_drug_name, secondary_dose_mg, secondary_repr["properties"],
            patient_weight_kg=patient_profile["weight_kg"],
            t_max_hours=t_max_hours
        )
        if clearance_factor != 1.0:
            secondary_pk["one_compartment"]["cmax"] *= (1.0 / clearance_factor)

    layer_times["pharmacokinetics"] = round(time.time() - t0, 4)

    # ═══════════════════════════════════════════
    # Layer 3: Drug-Target Interaction
    # ═══════════════════════════════════════════
    t0 = time.time()
    fingerprint = np.array(drug_repr["fingerprint"], dtype=np.float32)
    cmax = pk_results.get("one_compartment", {}).get("cmax", 1.0)
    drug_targets = predict_drug_target_interactions(drug_name, fingerprint, cmax)

    if polypharmacy_active and secondary_pk:
        sec_fp = np.array(secondary_repr["fingerprint"], dtype=np.float32)
        sec_cmax = secondary_pk.get("one_compartment", {}).get("cmax", 1.0)
        sec_targets = predict_drug_target_interactions(secondary_drug_name, sec_fp, sec_cmax)
        
        for channel in ["hERG", "Nav1.5", "Cav1.2"]:
            primary_dict = drug_targets["interactions"].get(channel, {})
            sec_dict = sec_targets["interactions"].get(channel, {})
            primary_val = primary_dict.get("binding_strength", 0.0) if isinstance(primary_dict, dict) else 0.0
            sec_val = sec_dict.get("binding_strength", 0.0) if isinstance(sec_dict, dict) else 0.0
            
            combined_val = min(0.98, primary_val + sec_val - (primary_val * sec_val))
            if channel in drug_targets["interactions"]:
                drug_targets["interactions"][channel]["binding_strength"] = round(combined_val, 4)
                drug_targets["interactions"][channel]["occupancy_pct"] = round(combined_val * 100, 1)

    layer_times["drug_target_interaction"] = round(time.time() - t0, 4)

    # ═══════════════════════════════════════════
    # Layer 4: Pharmacodynamics (PD)
    # ═══════════════════════════════════════════
    t0 = time.time()
    pd_effects = compute_pd(drug_name, pk_results, drug_repr["properties"], drug_targets)

    # Hypokalemia electrolyte sensitivity factor
    if patient_profile["potassium_mM"] < 3.5:
        if pd_effects["arrhythmia_risk_from_qt"] == "low":
            pd_effects["arrhythmia_risk_from_qt"] = "moderate"
        elif pd_effects["arrhythmia_risk_from_qt"] == "moderate":
            pd_effects["arrhythmia_risk_from_qt"] = "high"

    layer_times["pharmacodynamics"] = round(time.time() - t0, 4)

    # ═══════════════════════════════════════════
    # Layer 5: Cardiovascular System Model
    # ═══════════════════════════════════════════
    t0 = time.time()
    cv_results = simulate_cardiovascular(pd_effects, drug_targets)

    if patient_profile["baseline_qtc_ms"] != 410.0:
        delta_qtc = cv_results["ecg"]["qtc_interval_ms"] - 410.0
        cv_results["ecg"]["qtc_interval_ms"] = patient_profile["baseline_qtc_ms"] + delta_qtc

    layer_times["cardiovascular_model"] = round(time.time() - t0, 4)

    # ═══════════════════════════════════════════
    # Layer 6: ML Predictions
    # ═══════════════════════════════════════════
    t0 = time.time()
    ml_results = run_ml_predictions(
        drug_repr, drug_targets, pk_results, pd_effects, cv_results, patient_profile
    )

    if polypharmacy_active:
        herg_binding = drug_targets["interactions"].get("hERG", {}).get("binding_strength", 0.0)
        ml_results["predictions"]["drug_safety"]["polypharmacy_warning"] = (
            f"Simulating combined effect of {drug_name} ({dose_mg}mg) + {secondary_drug_name} ({secondary_dose_mg}mg). "
            f"Combined hERG binding strength: {herg_binding:.2f}."
        )

    layer_times["ml_predictions"] = round(time.time() - t0, 4)
    total_time = round(time.time() - start_time, 4)

    # ═══════════════════════════════════════════
    # Layer 7: Digital Twin Output
    # ═══════════════════════════════════════════
    return {
        "simulation_info": {
            "drug_name": drug_name,
            "dose_mg": dose_mg,
            "secondary_drug_name": secondary_drug_name if polypharmacy_active else None,
            "secondary_dose_mg": secondary_dose_mg if polypharmacy_active else 0.0,
            "polypharmacy_active": polypharmacy_active,
            "patient": patient_profile,
            "t_max_hours": t_max_hours,
            "total_computation_time_seconds": total_time,
            "layer_times": layer_times,
        },
        "layer_1_drug_representation": {
            "smiles": drug_repr["smiles"],
            "properties": drug_repr["properties"],
            "fingerprint_density": drug_repr["fingerprint_density"],
            "n_bits_on": drug_repr["n_bits_on"],
            "secondary_smiles": secondary_repr["smiles"] if polypharmacy_active else None,
        },
        "layer_2_drug_target": {
            "interactions": drug_targets["interactions"],
            "primary_targets": drug_targets["primary_targets"],
            "secondary_targets": drug_targets["secondary_targets"],
            "max_binding_strength": drug_targets["max_binding_strength"],
            "selectivity_index": drug_targets["selectivity_index"],
        },
        "layer_3_pharmacokinetics": {
            "pk_parameters": pk_results["pk_parameters"],
            "adme": pk_results["adme"],
            "concentration_profile": {
                "time": pk_results["one_compartment"]["time_hours"],
                "concentration": pk_results["one_compartment"]["concentration_ugml"],
            },
            "two_compartment": {
                "time": pk_results["two_compartment"]["time_hours"],
                "central": pk_results["two_compartment"]["concentration_central"],
                "peripheral": pk_results["two_compartment"]["concentration_peripheral"],
            },
            "cmax": pk_results["one_compartment"]["cmax"],
            "tmax": pk_results["one_compartment"]["tmax"],
            "auc": pk_results["one_compartment"]["auc"],
        },
        "layer_4_pharmacodynamics": {
            "effects_at_cmax": pd_effects["effects_at_cmax"],
            "effect_profiles": {
                param: {
                    "time": profile["time_hours"],
                    "values": profile["effect_values"],
                    "peak": profile["peak_effect"],
                }
                for param, profile in pd_effects["effect_profiles"].items()
            },
            "arrhythmia_risk_from_qt": pd_effects["arrhythmia_risk_from_qt"],
        },
        "layer_5_cardiovascular": {
            "cardiac_parameters": cv_results["cardiac_parameters"],
            "windkessel": {
                "time": cv_results["windkessel"]["time_seconds"],
                "pressure": cv_results["windkessel"]["arterial_pressure_mmHg"],
                "systolic": cv_results["windkessel"]["systolic_pressure"],
                "diastolic": cv_results["windkessel"]["diastolic_pressure"],
                "map": cv_results["windkessel"]["mean_arterial_pressure"],
            },
            "ecg": {
                "time": cv_results["ecg"]["time_seconds"],
                "voltage": cv_results["ecg"]["voltage_mV"],
                "qt_interval": cv_results["ecg"]["qt_interval_ms"],
                "qtc_interval": cv_results["ecg"]["qtc_interval_ms"],
            },
            "cardiac_output": cv_results["cardiac_output"],
            "arrhythmia_risk": cv_results["arrhythmia_risk"],
            "cardiac_state": cv_results["cardiac_state"],
            "warnings": cv_results["warnings"],
        },
        "layer_6_ml_predictions": ml_results["predictions"],
    }
