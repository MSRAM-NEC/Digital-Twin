"""
Layer 4: Pharmacodynamics (PD) Model
Models the relationship between drug concentration and physiological effect.

Key models:
- Emax model: E = Emax * C / (EC50 + C)
- Sigmoid Emax: E = Emax * C^n / (EC50^n + C^n)
- Linear model: E = slope * C
- Effect compartment model with hysteresis

Drug effects on cardiac parameters:
- Heart rate
- Blood pressure (systolic/diastolic)
- Contractility
- QT interval
- Arrhythmia risk
"""

import numpy as np


# Drug-specific PD parameters
PD_PARAMETERS = {
    "metoprolol": {
        "heart_rate": {"emax": -25, "ec50": 0.05, "hill": 1.2, "baseline": 72},
        "systolic_bp": {"emax": -15, "ec50": 0.08, "hill": 1.0, "baseline": 120},
        "diastolic_bp": {"emax": -10, "ec50": 0.08, "hill": 1.0, "baseline": 80},
        "contractility": {"emax": -15, "ec50": 0.1, "hill": 1.0, "baseline": 100},
        "qt_interval": {"emax": 5, "ec50": 0.15, "hill": 1.0, "baseline": 400},
    },
    "propranolol": {
        "heart_rate": {"emax": -30, "ec50": 0.04, "hill": 1.3, "baseline": 72},
        "systolic_bp": {"emax": -20, "ec50": 0.06, "hill": 1.0, "baseline": 120},
        "diastolic_bp": {"emax": -12, "ec50": 0.06, "hill": 1.0, "baseline": 80},
        "contractility": {"emax": -20, "ec50": 0.08, "hill": 1.0, "baseline": 100},
        "qt_interval": {"emax": 3, "ec50": 0.2, "hill": 1.0, "baseline": 400},
    },
    "amlodipine": {
        "heart_rate": {"emax": 5, "ec50": 0.005, "hill": 1.0, "baseline": 72},
        "systolic_bp": {"emax": -30, "ec50": 0.003, "hill": 1.2, "baseline": 120},
        "diastolic_bp": {"emax": -20, "ec50": 0.003, "hill": 1.2, "baseline": 80},
        "contractility": {"emax": -10, "ec50": 0.01, "hill": 1.0, "baseline": 100},
        "qt_interval": {"emax": -5, "ec50": 0.008, "hill": 1.0, "baseline": 400},
    },
    "lisinopril": {
        "heart_rate": {"emax": 0, "ec50": 0.1, "hill": 1.0, "baseline": 72},
        "systolic_bp": {"emax": -25, "ec50": 0.02, "hill": 1.0, "baseline": 120},
        "diastolic_bp": {"emax": -15, "ec50": 0.02, "hill": 1.0, "baseline": 80},
        "contractility": {"emax": 5, "ec50": 0.05, "hill": 1.0, "baseline": 100},
        "qt_interval": {"emax": 0, "ec50": 1.0, "hill": 1.0, "baseline": 400},
    },
    "amiodarone": {
        "heart_rate": {"emax": -15, "ec50": 0.5, "hill": 1.0, "baseline": 72},
        "systolic_bp": {"emax": -10, "ec50": 0.8, "hill": 1.0, "baseline": 120},
        "diastolic_bp": {"emax": -5, "ec50": 0.8, "hill": 1.0, "baseline": 80},
        "contractility": {"emax": -5, "ec50": 1.0, "hill": 1.0, "baseline": 100},
        "qt_interval": {"emax": 60, "ec50": 0.3, "hill": 1.5, "baseline": 400},
    },
    "digoxin": {
        "heart_rate": {"emax": -20, "ec50": 0.001, "hill": 1.8, "baseline": 72},
        "systolic_bp": {"emax": 10, "ec50": 0.0015, "hill": 1.0, "baseline": 120},
        "diastolic_bp": {"emax": 5, "ec50": 0.0015, "hill": 1.0, "baseline": 80},
        "contractility": {"emax": 30, "ec50": 0.001, "hill": 2.0, "baseline": 100},
        "qt_interval": {"emax": -20, "ec50": 0.002, "hill": 1.0, "baseline": 400},
    },
    "verapamil": {
        "heart_rate": {"emax": -20, "ec50": 0.1, "hill": 1.0, "baseline": 72},
        "systolic_bp": {"emax": -25, "ec50": 0.08, "hill": 1.2, "baseline": 120},
        "diastolic_bp": {"emax": -15, "ec50": 0.08, "hill": 1.2, "baseline": 80},
        "contractility": {"emax": -20, "ec50": 0.15, "hill": 1.0, "baseline": 100},
        "qt_interval": {"emax": -10, "ec50": 0.12, "hill": 1.0, "baseline": 400},
    },
    "aspirin": {
        "heart_rate": {"emax": 0, "ec50": 1.0, "hill": 1.0, "baseline": 72},
        "systolic_bp": {"emax": -2, "ec50": 5.0, "hill": 1.0, "baseline": 120},
        "diastolic_bp": {"emax": -1, "ec50": 5.0, "hill": 1.0, "baseline": 80},
        "contractility": {"emax": 0, "ec50": 1.0, "hill": 1.0, "baseline": 100},
        "qt_interval": {"emax": 0, "ec50": 1.0, "hill": 1.0, "baseline": 400},
    },
    "caffeine": {
        "heart_rate": {"emax": 15, "ec50": 2.0, "hill": 1.0, "baseline": 72},
        "systolic_bp": {"emax": 8, "ec50": 3.0, "hill": 1.0, "baseline": 120},
        "diastolic_bp": {"emax": 5, "ec50": 3.0, "hill": 1.0, "baseline": 80},
        "contractility": {"emax": 10, "ec50": 2.5, "hill": 1.0, "baseline": 100},
        "qt_interval": {"emax": -5, "ec50": 4.0, "hill": 1.0, "baseline": 400},
    },
    "doxorubicin": {
        "heart_rate": {"emax": 10, "ec50": 0.1, "hill": 1.0, "baseline": 72},
        "systolic_bp": {"emax": -15, "ec50": 0.2, "hill": 1.0, "baseline": 120},
        "diastolic_bp": {"emax": -10, "ec50": 0.2, "hill": 1.0, "baseline": 80},
        "contractility": {"emax": -35, "ec50": 0.05, "hill": 2.0, "baseline": 100},
        "qt_interval": {"emax": 40, "ec50": 0.08, "hill": 1.5, "baseline": 400},
    },
    "sotalol": {
        "heart_rate": {"emax": -20, "ec50": 1.0, "hill": 1.2, "baseline": 72},
        "systolic_bp": {"emax": -10, "ec50": 1.5, "hill": 1.0, "baseline": 120},
        "diastolic_bp": {"emax": -8, "ec50": 1.5, "hill": 1.0, "baseline": 80},
        "contractility": {"emax": -10, "ec50": 2.0, "hill": 1.0, "baseline": 100},
        "qt_interval": {"emax": 50, "ec50": 0.8, "hill": 1.3, "baseline": 400},
    },
}


def sigmoid_emax(concentration: float, emax: float, ec50: float,
                  hill: float = 1.0, baseline: float = 0.0) -> float:
    """
    Sigmoid Emax model with a linear off-target toxicity bleed.
    E = baseline + Emax * C^n / (EC50^n + C^n) + off_target_toxic_effect
    """
    if concentration <= 0 or ec50 <= 0:
        return baseline

    c_n = concentration ** hill
    ec50_n = ec50 ** hill

    # Primary receptor-mediated saturation
    effect = baseline + emax * c_n / (ec50_n + c_n)
    
    # Massive Supramaximal Overdose Mechanic:
    # Once plasma concentration significantly exceeds the EC50 (e.g., >3x EC50), 
    # the drug begins exhibiting unbounded off-target systemic toxicities.
    # Rather than plateauing perfectly at Emax, severe toxicity causes the effect to keep bleeding.
    off_target_threshold = ec50 * 3.0
    if concentration > off_target_threshold:
        excess_conc = concentration - off_target_threshold
        # Toxicity slope scales directionally with the primary action
        # e.g., Stimulants cause boundless HR spikes; Blockers cause boundless bradycardia
        tox_slope = (emax * 0.15) / ec50 
        effect += tox_slope * excess_conc

    return effect


def compute_effect_over_time(concentration_profile: list, time_hours: list,
                              pd_params: dict) -> dict:
    """
    Compute drug effect over time given a concentration-time profile.
    """
    effects = []
    for c in concentration_profile:
        e = sigmoid_emax(
            c,
            pd_params["emax"],
            pd_params["ec50"],
            pd_params.get("hill", 1.0),
            pd_params.get("baseline", 0.0)
        )
        effects.append(round(e, 2))

    return {
        "time_hours": time_hours,
        "effect_values": effects,
        "emax_theoretical": pd_params["emax"] + pd_params.get("baseline", 0.0),
        "ec50": pd_params["ec50"],
        "baseline": pd_params.get("baseline", 0.0),
        "peak_effect": float(min(effects) if pd_params["emax"] < 0 else max(effects)),
        "time_to_peak_effect": float(time_hours[
            np.argmin(effects) if pd_params["emax"] < 0 else np.argmax(effects)
        ]),
    }


def _generate_pd_params(drug_name: str, drug_properties: dict,
                         drug_target_interactions: dict) -> dict:
    """Generate PD parameters from drug properties and target interactions."""
    interactions = drug_target_interactions.get("interactions", {})

    # Base PD effects from target interactions
    hr_effect = 0
    sbp_effect = 0
    dbp_effect = 0
    contractility_effect = 0
    qt_effect = 0

    for target, data in interactions.items():
        strength = data["binding_strength"]
        if strength < 0.1:
            continue

        if target in ("Beta1", "Beta2"):
            hr_effect -= strength * 30
            contractility_effect -= strength * 20
            sbp_effect -= strength * 15
        elif target == "Alpha1":
            sbp_effect -= strength * 20
            dbp_effect -= strength * 15
        elif target == "Cav1.2":
            sbp_effect -= strength * 25
            dbp_effect -= strength * 18
            contractility_effect -= strength * 15
            hr_effect += strength * 5
        elif target == "hERG":
            qt_effect += strength * 50
        elif target == "Nav1.5":
            qt_effect += strength * 20
            hr_effect -= strength * 10
        elif target in ("ACE", "AT1"):
            sbp_effect -= strength * 25
            dbp_effect -= strength * 15
        elif target == "NaK_ATPase":
            contractility_effect += strength * 30
            hr_effect -= strength * 15
        elif target == "PDE3":
            contractility_effect += strength * 20
            hr_effect += strength * 10

    # Estimate EC50 from drug properties
    mw = drug_properties.get("mw", 300)
    logp = drug_properties.get("logp", 2.0)
    ec50 = max(0.001, 0.1 * (mw / 300) * (1 / (1 + abs(logp))))

    return {
        "heart_rate": {"emax": round(hr_effect, 1), "ec50": ec50,
                       "hill": 1.0, "baseline": 72},
        "systolic_bp": {"emax": round(sbp_effect, 1), "ec50": ec50,
                        "hill": 1.0, "baseline": 120},
        "diastolic_bp": {"emax": round(dbp_effect, 1), "ec50": ec50,
                         "hill": 1.0, "baseline": 80},
        "contractility": {"emax": round(contractility_effect, 1), "ec50": ec50,
                          "hill": 1.0, "baseline": 100},
        "qt_interval": {"emax": round(qt_effect, 1), "ec50": ec50,
                        "hill": 1.0, "baseline": 400},
    }


def compute_pd(drug_name: str, pk_results: dict, drug_properties: dict,
               drug_target_interactions: dict) -> dict:
    """
    Full PD pipeline.
    Computes effect-time profiles for all cardiac parameters.
    """
    name_lower = drug_name.lower().strip()

    # Get PD parameters
    if name_lower in PD_PARAMETERS:
        pd_params = PD_PARAMETERS[name_lower]
    else:
        pd_params = _generate_pd_params(drug_name, drug_properties, drug_target_interactions)

    # Get concentration profile
    time_hours = pk_results["one_compartment"]["time_hours"]
    concentrations = pk_results["one_compartment"]["concentration_ugml"]

    # Compute effects for each parameter
    effects = {}
    for param_name, params in pd_params.items():
        effects[param_name] = compute_effect_over_time(concentrations, time_hours, params)

    # Current effect at Cmax
    cmax = pk_results["one_compartment"]["cmax"]
    current_effects = {}
    for param_name, params in pd_params.items():
        current_effects[param_name] = round(sigmoid_emax(
            cmax, params["emax"], params["ec50"],
            params.get("hill", 1.0), params.get("baseline", 0.0)
        ), 1)

    # Risk assessment
    qt_at_cmax = current_effects.get("qt_interval", 400)
    arrhythmia_risk = "low"
    if qt_at_cmax > 500:
        arrhythmia_risk = "high"
    elif qt_at_cmax > 460:
        arrhythmia_risk = "moderate"

    return {
        "drug_name": drug_name,
        "pd_parameters": pd_params,
        "effect_profiles": effects,
        "effects_at_cmax": current_effects,
        "arrhythmia_risk_from_qt": arrhythmia_risk,
    }
