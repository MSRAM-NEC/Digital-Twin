"""
Layer 3: Pharmacokinetics (PK) Model
Models ADME: Absorption, Distribution, Metabolism, Excretion

Computes drug concentration over time using compartmental models:
- One-compartment model with first-order absorption
- Two-compartment model for drugs with tissue distribution

Key equation:
    C(t) = (D * ka / (Vd * (ka - ke))) * (e^(-ke*t) - e^(-ka*t))

Where:
    D = dose (mg)
    ka = absorption rate constant
    ke = elimination rate constant
    Vd = volume of distribution (L)
    C(t) = plasma concentration at time t
"""

import numpy as np
from scipy.integrate import solve_ivp

# NumPy compat: trapezoid was added in 2.0, older versions have trapz
_trapz = getattr(np, 'trapezoid', None) or np.trapz


# PK parameters for known drugs (clinical values)
PK_PARAMETERS = {
    "metoprolol": {
        "bioavailability": 0.50, "vd": 290, "clearance": 63,
        "ka": 1.2, "protein_binding": 0.12, "half_life": 3.5,
        "tmax": 1.5, "route": "oral",
    },
    "propranolol": {
        "bioavailability": 0.26, "vd": 270, "clearance": 60,
        "ka": 1.5, "protein_binding": 0.90, "half_life": 4.0,
        "tmax": 1.5, "route": "oral",
    },
    "amlodipine": {
        "bioavailability": 0.64, "vd": 1400, "clearance": 25,
        "ka": 0.4, "protein_binding": 0.98, "half_life": 35.0,
        "tmax": 8.0, "route": "oral",
    },
    "lisinopril": {
        "bioavailability": 0.25, "vd": 21, "clearance": 10,
        "ka": 0.6, "protein_binding": 0.0, "half_life": 12.0,
        "tmax": 7.0, "route": "oral",
    },
    "warfarin": {
        "bioavailability": 0.99, "vd": 10, "clearance": 0.2,
        "ka": 2.0, "protein_binding": 0.99, "half_life": 40.0,
        "tmax": 2.0, "route": "oral",
    },
    "digoxin": {
        "bioavailability": 0.70, "vd": 500, "clearance": 12,
        "ka": 0.8, "protein_binding": 0.25, "half_life": 36.0,
        "tmax": 2.0, "route": "oral",
    },
    "amiodarone": {
        "bioavailability": 0.50, "vd": 5000, "clearance": 15,
        "ka": 0.3, "protein_binding": 0.96, "half_life": 58.0,
        "tmax": 5.0, "route": "oral",
    },
    "aspirin": {
        "bioavailability": 0.68, "vd": 10, "clearance": 39,
        "ka": 3.0, "protein_binding": 0.50, "half_life": 3.1,
        "tmax": 0.5, "route": "oral",
    },
    "atenolol": {
        "bioavailability": 0.50, "vd": 63, "clearance": 10.5,
        "ka": 0.5, "protein_binding": 0.05, "half_life": 6.5,
        "tmax": 3.0, "route": "oral",
    },
    "verapamil": {
        "bioavailability": 0.22, "vd": 300, "clearance": 65,
        "ka": 1.0, "protein_binding": 0.90, "half_life": 6.0,
        "tmax": 1.5, "route": "oral",
    },
    "captopril": {
        "bioavailability": 0.65, "vd": 57, "clearance": 46,
        "ka": 2.5, "protein_binding": 0.30, "half_life": 2.0,
        "tmax": 1.0, "route": "oral",
    },
    "losartan": {
        "bioavailability": 0.33, "vd": 34, "clearance": 36,
        "ka": 1.5, "protein_binding": 0.99, "half_life": 2.0,
        "tmax": 1.0, "route": "oral",
    },
    "sotalol": {
        "bioavailability": 0.90, "vd": 100, "clearance": 8.3,
        "ka": 0.8, "protein_binding": 0.0, "half_life": 12.0,
        "tmax": 2.5, "route": "oral",
    },
    "nifedipine": {
        "bioavailability": 0.45, "vd": 80, "clearance": 28,
        "ka": 2.0, "protein_binding": 0.96, "half_life": 2.0,
        "tmax": 0.5, "route": "oral",
    },
    "caffeine": {
        "bioavailability": 0.99, "vd": 37, "clearance": 8.0,
        "ka": 3.0, "protein_binding": 0.35, "half_life": 5.0,
        "tmax": 0.5, "route": "oral",
    },
    "ibuprofen": {
        "bioavailability": 0.80, "vd": 10, "clearance": 3.5,
        "ka": 2.5, "protein_binding": 0.99, "half_life": 2.0,
        "tmax": 1.0, "route": "oral",
    },
    "acetaminophen": {
        "bioavailability": 0.85, "vd": 50, "clearance": 20,
        "ka": 3.5, "protein_binding": 0.25, "half_life": 2.5,
        "tmax": 0.5, "route": "oral",
    },
    "doxorubicin": {
        "bioavailability": 1.0, "vd": 800, "clearance": 40,
        "ka": 5.0, "protein_binding": 0.75, "half_life": 48.0,
        "tmax": 0.1, "route": "iv",
    },
}


def _get_pk_params(drug_name: str, drug_properties: dict) -> dict:
    """Get PK parameters from database or estimate from properties."""
    name_lower = drug_name.lower().strip()

    if name_lower in PK_PARAMETERS:
        return PK_PARAMETERS[name_lower]

    # Estimate PK parameters from physicochemical properties
    mw = drug_properties.get("mw", 300)
    logp = drug_properties.get("logp", 2.0)
    tpsa = drug_properties.get("tpsa", 80)

    # Rule-of-thumb estimations
    bioavailability = max(0.1, min(0.95, 1.0 - 0.1 * max(0, logp - 3) - 0.002 * max(0, tpsa - 120)))
    vd = max(5, 10 * (1 + 2 * max(0, logp)))  # Lipophilic drugs distribute widely
    protein_binding = max(0.0, min(0.99, 0.5 + 0.1 * logp))
    half_life = drug_properties.get("half_life", max(1, 2 + logp * 3))
    ke = 0.693 / half_life
    clearance = ke * vd
    ka = max(0.3, 3.0 - 0.005 * mw)  # Larger molecules absorb slower
    tmax = max(0.25, np.log(ka / ke) / (ka - ke)) if abs(ka - ke) > 0.01 else 1.0

    return {
        "bioavailability": round(bioavailability, 3),
        "vd": round(vd, 1),
        "clearance": round(clearance, 2),
        "ka": round(ka, 3),
        "protein_binding": round(protein_binding, 3),
        "half_life": round(half_life, 2),
        "tmax": round(tmax, 2),
        "route": "oral",
    }


def one_compartment_model(dose_mg: float, pk_params: dict,
                           t_max_hours: float = 48.0, n_points: int = 500) -> dict:
    """
    One-compartment PK model with first-order absorption.

    Returns concentration-time profile.
    """
    F = pk_params["bioavailability"]
    Vd = pk_params["vd"]
    ka = pk_params["ka"]
    ke = 0.693 / pk_params["half_life"]

    t = np.linspace(0, t_max_hours, n_points)

    # Oral absorption model
    if pk_params.get("route") == "iv":
        # IV bolus: instant absorption
        C = (F * dose_mg / Vd) * np.exp(-ke * t)
    else:
        # Oral: first-order absorption
        if abs(ka - ke) < 0.001:
            # Edge case: ka ≈ ke
            C = (F * dose_mg * ka / Vd) * t * np.exp(-ke * t)
        else:
            C = (F * dose_mg * ka / (Vd * (ka - ke))) * (np.exp(-ke * t) - np.exp(-ka * t))

    # Ensure non-negative
    C = np.maximum(C, 0)

    # Convert to μg/mL (assuming mg dose, L volume)
    C_ugml = C  # mg/L = μg/mL

    return {
        "time_hours": t.tolist(),
        "concentration_ugml": C_ugml.tolist(),
        "cmax": float(np.max(C_ugml)),
        "tmax": float(t[np.argmax(C_ugml)]),
        "auc": float(_trapz(C_ugml, t)),
        "half_life": pk_params["half_life"],
    }


def two_compartment_model(dose_mg: float, pk_params: dict,
                           t_max_hours: float = 48.0, n_points: int = 500) -> dict:
    """
    Two-compartment PK model for deeper tissue distribution analysis.
    Uses ODE solver for more accurate simulation.
    """
    F = pk_params["bioavailability"]
    Vd = pk_params["vd"]
    ka = pk_params["ka"]
    ke = 0.693 / pk_params["half_life"]

    # Distribution parameters (estimated)
    k12 = ke * 0.5  # central → peripheral rate
    k21 = ke * 0.3  # peripheral → central rate

    Vc = Vd * 0.4   # Central compartment volume
    Vp = Vd * 0.6   # Peripheral compartment volume

    def odes(t, y):
        A_gut, A_central, A_periph = y
        dA_gut = -ka * A_gut
        dA_central = ka * A_gut - ke * A_central - k12 * A_central + k21 * A_periph
        dA_periph = k12 * A_central - k21 * A_periph
        return [dA_gut, dA_central, dA_periph]

    # Initial conditions
    if pk_params.get("route") == "iv":
        y0 = [0, F * dose_mg, 0]
    else:
        y0 = [F * dose_mg, 0, 0]

    t_span = (0, t_max_hours)
    t_eval = np.linspace(0, t_max_hours, n_points)

    # Use BDF (stiff solver) to prevent numerical explosion with high ka values
    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method="BDF", max_step=0.5)

    # Biological plausibility clipping (prevent numerical artifacts)
    theoretical_max = (F * dose_mg) / Vc
    C_central = np.clip(sol.y[1] / Vc, 0, theoretical_max)
    C_periph = np.clip(sol.y[2] / Vp, 0, theoretical_max)

    return {
        "time_hours": sol.t.tolist(),
        "concentration_central": C_central.tolist(),
        "concentration_peripheral": C_periph.tolist(),
        "cmax_central": float(np.max(C_central)),
        "cmax_peripheral": float(np.max(C_periph)),
        "tmax": float(sol.t[np.argmax(C_central)]),
        "auc_central": float(_trapz(C_central, sol.t)),
        "auc_peripheral": float(_trapz(C_periph, sol.t)),
    }


def compute_pk(drug_name: str, dose_mg: float, drug_properties: dict,
               patient_weight_kg: float = 70.0, t_max_hours: float = 48.0) -> dict:
    """
    Full PK pipeline.
    Computes drug concentration profiles using one- and two-compartment models.
    """
    pk_params = _get_pk_params(drug_name, drug_properties)

    # Adjust dose for patient weight if needed
    adjusted_dose = dose_mg  # Most drugs are flat-dosed

    # Run models
    one_comp = one_compartment_model(adjusted_dose, pk_params, t_max_hours)
    two_comp = two_compartment_model(adjusted_dose, pk_params, t_max_hours)

    # ADME summary
    adme = {
        "absorption": {
            "bioavailability": pk_params["bioavailability"],
            "ka": pk_params["ka"],
            "route": pk_params["route"],
            "tmax_hours": one_comp["tmax"],
        },
        "distribution": {
            "vd_liters": pk_params["vd"],
            "protein_binding": pk_params["protein_binding"],
            "free_fraction": round(1 - pk_params["protein_binding"], 3),
        },
        "metabolism": {
            "clearance_L_hr": pk_params["clearance"],
            "half_life_hours": pk_params["half_life"],
        },
        "excretion": {
            "elimination_constant": round(0.693 / pk_params["half_life"], 4),
            "time_to_95pct_eliminated": round(pk_params["half_life"] * 4.32, 1),
        },
    }

    return {
        "drug_name": drug_name,
        "dose_mg": dose_mg,
        "patient_weight_kg": patient_weight_kg,
        "pk_parameters": pk_params,
        "one_compartment": one_comp,
        "two_compartment": two_comp,
        "adme": adme,
    }
