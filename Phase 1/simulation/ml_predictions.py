"""
Layer 6: Machine Learning Prediction Layer
Combines all simulation outputs to make clinical predictions.

Predictions:
- Arrhythmia risk score
- Heart attack probability
- Blood pressure response category
- Treatment effectiveness score
- Adverse event probability
- Drug safety classification

Uses ensemble of:
- Logistic regression-like risk models
- Bayesian risk estimation
- Feature-weighted scoring
"""

import numpy as np
import hashlib


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _normalize(value, min_val, max_val):
    """Normalize to [0, 1] range."""
    if max_val == min_val:
        return 0.5
    return np.clip((value - min_val) / (max_val - min_val), 0, 1)


def compute_risk_features(drug_repr: dict, drug_targets: dict,
                           pk_results: dict, pd_effects: dict,
                           cv_results: dict, patient: dict) -> dict:
    """
    Extract features from all previous layers for ML predictions.
    """
    effects = pd_effects.get("effects_at_cmax", {})
    interactions = drug_targets.get("interactions", {})
    cardiac = cv_results.get("cardiac_parameters", {})
    arrhythmia_data = cv_results.get("arrhythmia_risk", {})
    co_data = cv_results.get("cardiac_output", {})
    props = drug_repr.get("properties", {})

    features = {
        # Drug features
        "molecular_weight": props.get("mw", 300),
        "logp": props.get("logp", 2.0),
        "tpsa": props.get("tpsa", 80),
        "fingerprint_density": drug_repr.get("fingerprint_density", 0.05),

        # Target interaction features
        "herg_binding": interactions.get("hERG", {}).get("binding_strength", 0),
        "nav15_binding": interactions.get("Nav1.5", {}).get("binding_strength", 0),
        "cav12_binding": interactions.get("Cav1.2", {}).get("binding_strength", 0),
        "beta1_binding": interactions.get("Beta1", {}).get("binding_strength", 0),
        "max_binding": drug_targets.get("max_binding_strength", 0),
        "selectivity": drug_targets.get("selectivity_index", 1),
        "n_targets": drug_targets.get("n_significant_interactions", 0),

        # PK features
        "cmax": pk_results.get("one_compartment", {}).get("cmax", 0),
        "auc": pk_results.get("one_compartment", {}).get("auc", 0),
        "half_life": pk_results.get("pk_parameters", {}).get("half_life", 4),
        "bioavailability": pk_results.get("pk_parameters", {}).get("bioavailability", 0.5),
        "protein_binding": pk_results.get("pk_parameters", {}).get("protein_binding", 0.5),

        # PD features (effects at Cmax)
        "hr_effect": effects.get("heart_rate", 72) - 72,
        "sbp_effect": effects.get("systolic_bp", 120) - 120,
        "contractility_effect": effects.get("contractility", 100) - 100,
        "qt_change": effects.get("qt_interval", 400) - 400,

        # CV features
        "heart_rate": cardiac.get("heart_rate_bpm", 72),
        "systolic_bp": cardiac.get("systolic_bp_mmHg", 120),
        "diastolic_bp": cardiac.get("diastolic_bp_mmHg", 80),
        "qt_interval": cardiac.get("qt_interval_ms", 400),
        "qtc_interval": arrhythmia_data.get("qtc_interval_ms", 400),
        "arrhythmia_prob": arrhythmia_data.get("arrhythmia_probability", 0),
        "cardiac_output": co_data.get("cardiac_output_Lmin", 5.0),
        "ejection_fraction": co_data.get("ejection_fraction_pct", 60),

        # Patient features
        "age": patient.get("age", 45),
        "weight": patient.get("weight_kg", 70),
        "has_heart_disease": float(patient.get("heart_disease", False)),
        "has_hypertension": float(patient.get("hypertension", False)),
        "has_diabetes": float(patient.get("diabetes", False)),
        "has_renal_impairment": float(patient.get("renal_impairment", False)),
    }

    return features


def predict_arrhythmia_risk(features: dict) -> dict:
    """Predict arrhythmia risk using weighted logistic model."""
    # Weighted features (simulating trained model weights)
    score = (
        0.3 * features["herg_binding"] +
        0.15 * features["nav15_binding"] +
        0.1 * _normalize(features["qt_change"], 0, 100) +
        0.15 * _normalize(features["qtc_interval"], 400, 550) +
        0.1 * features["arrhythmia_prob"] +
        0.05 * _normalize(features["age"], 20, 90) +
        0.05 * features["has_heart_disease"] +
        0.05 * _normalize(abs(features["hr_effect"]), 0, 30) +
        0.05 * _normalize(features["n_targets"], 0, 5)
    )

    probability = float(_sigmoid((score - 0.3) * 5))

    if probability > 0.6:
        level = "HIGH"
    elif probability > 0.3:
        level = "MODERATE"
    elif probability > 0.1:
        level = "LOW"
    else:
        level = "MINIMAL"

    return {
        "probability": round(probability, 4),
        "risk_level": level,
        "contributing_factors": {
            "hERG_blockade": round(features["herg_binding"], 3),
            "QT_prolongation": round(features["qt_change"], 1),
            "QTc_interval": round(features["qtc_interval"], 1),
            "age_factor": round(_normalize(features["age"], 20, 90), 3),
        },
    }


def predict_cardiac_event_risk(features: dict) -> dict:
    """Predict major adverse cardiac event (MACE) risk."""
    score = (
        0.15 * _normalize(features["age"], 30, 90) +
        0.10 * features["has_heart_disease"] +
        0.10 * features["has_hypertension"] +
        0.08 * features["has_diabetes"] +
        0.10 * _normalize(abs(features["sbp_effect"]), 0, 40) +
        0.12 * _normalize(abs(features["contractility_effect"]), 0, 30) +
        0.10 * _normalize(features["herg_binding"], 0, 1) +
        0.08 * _normalize(5.0 - features["cardiac_output"], 0, 3) +
        0.07 * _normalize(60 - features["ejection_fraction"], 0, 40) +
        0.05 * features["has_renal_impairment"] +
        0.05 * _normalize(features["half_life"], 0, 50)
    )

    probability = float(_sigmoid((score - 0.35) * 4))

    return {
        "probability": round(probability, 4),
        "risk_level": "HIGH" if probability > 0.5 else "MODERATE" if probability > 0.2 else "LOW",
        "contributing_factors": {
            "contractility_change": round(features["contractility_effect"], 1),
            "cardiac_output": round(features["cardiac_output"], 2),
            "ejection_fraction": round(features["ejection_fraction"], 1),
            "comorbidities": sum([
                features["has_heart_disease"],
                features["has_hypertension"],
                features["has_diabetes"],
            ]),
        },
    }


def predict_bp_response(features: dict) -> dict:
    """Predict blood pressure response category."""
    sbp_change = features["sbp_effect"]
    dbp_change = features["sbp_effect"] * 0.65  # approximate

    if sbp_change < -20:
        category = "Strong reduction"
        effectiveness = min(1.0, abs(sbp_change) / 30)
    elif sbp_change < -10:
        category = "Moderate reduction"
        effectiveness = 0.5 + abs(sbp_change) / 40
    elif sbp_change < -5:
        category = "Mild reduction"
        effectiveness = 0.3
    elif sbp_change > 10:
        category = "Increase (adverse)"
        effectiveness = 0.0
    else:
        category = "Minimal effect"
        effectiveness = 0.1

    return {
        "systolic_change_mmHg": round(sbp_change, 1),
        "diastolic_change_mmHg": round(dbp_change, 1),
        "response_category": category,
        "bp_treatment_effectiveness": round(effectiveness, 3),
    }


def predict_treatment_effectiveness(features: dict, drug_category: str) -> dict:
    """Predict overall treatment effectiveness."""
    effectiveness = 0.5  # baseline

    if drug_category == "beta_blocker":
        effectiveness += 0.2 * features["beta1_binding"]
        effectiveness += 0.1 * _normalize(-features["hr_effect"], 0, 30)
        effectiveness += 0.1 * _normalize(-features["sbp_effect"], 0, 20)
        target_goals = {"heart_rate_reduction": features["hr_effect"],
                        "bp_reduction": features["sbp_effect"]}

    elif drug_category == "calcium_channel_blocker":
        effectiveness += 0.2 * features["cav12_binding"]
        effectiveness += 0.15 * _normalize(-features["sbp_effect"], 0, 30)
        target_goals = {"bp_reduction": features["sbp_effect"],
                        "vasodilation": features["cav12_binding"]}

    elif drug_category in ("ace_inhibitor", "arb"):
        effectiveness += 0.15 * _normalize(-features["sbp_effect"], 0, 25)
        effectiveness += 0.1 * _normalize(features["contractility_effect"], 0, 10)
        target_goals = {"bp_reduction": features["sbp_effect"],
                        "cardiac_remodeling": "positive" if features["contractility_effect"] > 0 else "neutral"}

    elif drug_category == "antiarrhythmic":
        if features["qt_change"] > 0 and features["arrhythmia_prob"] < 0.3:
            effectiveness += 0.25
        effectiveness += 0.1 * features["herg_binding"]
        target_goals = {"rhythm_control": "effective" if features["arrhythmia_prob"] < 0.3 else "partial"}

    elif drug_category == "cardiac_glycoside":
        effectiveness += 0.2 * _normalize(features["contractility_effect"], 0, 30)
        effectiveness += 0.1 * _normalize(-features["hr_effect"], 0, 20)
        target_goals = {"inotropy": features["contractility_effect"],
                        "rate_control": features["hr_effect"]}

    else:
        target_goals = {"general": "assessed"}
        effectiveness += 0.1 * (1 - features["arrhythmia_prob"])

    # Safety penalty
    if features["arrhythmia_prob"] > 0.5:
        effectiveness *= 0.5
    if features["cardiac_output"] < 3.5:
        effectiveness *= 0.7

    effectiveness = float(np.clip(effectiveness, 0, 1))

    return {
        "effectiveness_score": round(effectiveness, 3),
        "effectiveness_category": (
            "Highly Effective" if effectiveness > 0.75 else
            "Effective" if effectiveness > 0.5 else
            "Partially Effective" if effectiveness > 0.3 else
            "Low Effectiveness"
        ),
        "target_goals": target_goals,
        "drug_category": drug_category,
    }


def compute_drug_safety_score(features: dict) -> dict:
    """Compute overall drug safety classification."""
    # Safety demerits
    demerits = 0.0

    # hERG liability
    if features["herg_binding"] > 0.5:
        demerits += 3.0
    elif features["herg_binding"] > 0.3:
        demerits += 1.5
    elif features["herg_binding"] > 0.1:
        demerits += 0.5

    # QT prolongation
    if features["qt_change"] > 50:
        demerits += 3.0
    elif features["qt_change"] > 20:
        demerits += 1.5
    elif features["qt_change"] > 10:
        demerits += 0.5

    # Excessive BP changes
    if abs(features["sbp_effect"]) > 30:
        demerits += 1.5
    elif abs(features["sbp_effect"]) > 20:
        demerits += 0.5

    # Excessive HR changes
    if abs(features["hr_effect"]) > 25:
        demerits += 1.0

    # Contractility reduction
    if features["contractility_effect"] < -20:
        demerits += 2.0
    elif features["contractility_effect"] < -10:
        demerits += 0.5

    # Non-selectivity
    if features["n_targets"] > 3:
        demerits += 1.0

    # Long half-life (hard to reverse)
    if features["half_life"] > 24:
        demerits += 0.5

    safety_score = max(0, 10 - demerits)

    if safety_score >= 8:
        classification = "Safe"
        color = "green"
    elif safety_score >= 6:
        classification = "Caution"
        color = "yellow"
    elif safety_score >= 4:
        classification = "Warning"
        color = "orange"
    else:
        classification = "Dangerous"
        color = "red"

    return {
        "safety_score": round(safety_score, 1),
        "max_score": 10,
        "classification": classification,
        "color": color,
        "demerits_breakdown": {
            "hERG_liability": round(min(3, features["herg_binding"] * 5), 1),
            "QT_prolongation": round(min(3, features["qt_change"] / 20), 1),
            "BP_instability": round(min(1.5, abs(features["sbp_effect"]) / 20), 1),
            "HR_changes": round(min(1, abs(features["hr_effect"]) / 25), 1),
            "contractility_risk": round(min(2, max(0, -features["contractility_effect"]) / 10), 1),
        },
    }


def run_ml_predictions(drug_repr: dict, drug_targets: dict,
                       pk_results: dict, pd_effects: dict,
                       cv_results: dict, patient: dict) -> dict:
    """
    Full ML prediction pipeline.
    Combines all layers to generate clinical predictions.
    """
    # Extract features
    features = compute_risk_features(
        drug_repr, drug_targets, pk_results, pd_effects, cv_results, patient
    )

    drug_category = drug_repr.get("properties", {}).get("category", "unknown")

    # Run all prediction models
    arrhythmia = predict_arrhythmia_risk(features)
    cardiac_event = predict_cardiac_event_risk(features)
    bp_response = predict_bp_response(features)
    treatment = predict_treatment_effectiveness(features, drug_category)
    safety = compute_drug_safety_score(features)

    return {
        "predictions": {
            "arrhythmia_risk": arrhythmia,
            "cardiac_event_risk": cardiac_event,
            "bp_response": bp_response,
            "treatment_effectiveness": treatment,
            "drug_safety": safety,
        },
        "feature_summary": {
            "n_features_used": len(features),
            "key_features": {
                "hERG_binding": features["herg_binding"],
                "QT_change": features["qt_change"],
                "heart_rate": features["heart_rate"],
                "cardiac_output": features["cardiac_output"],
                "ejection_fraction": features["ejection_fraction"],
            },
        },
        "patient_profile": patient,
    }
