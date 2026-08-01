"""
Layer 2: Drug-Target Interaction
Predicts how strongly a drug binds to cardiac-relevant protein targets.

Targets include:
- Ion channels (hERG, Nav1.5, Cav1.2)
- Adrenergic receptors (β1, β2, α1)
- ACE (Angiotensin Converting Enzyme)
- AT1 (Angiotensin II receptor)
- COX enzymes
- Na/K ATPase
"""

import numpy as np
import hashlib


# Cardiac-relevant protein targets with physiological roles
PROTEIN_TARGETS = {
    "hERG": {
        "full_name": "hERG Potassium Channel",
        "role": "Controls cardiac repolarization; blocking causes QT prolongation",
        "effect": "arrhythmia_risk",
        "embedding_seed": 42,
    },
    "Nav1.5": {
        "full_name": "Cardiac Sodium Channel",
        "role": "Controls cardiac depolarization and conduction velocity",
        "effect": "conduction_velocity",
        "embedding_seed": 101,
    },
    "Cav1.2": {
        "full_name": "L-type Calcium Channel",
        "role": "Controls cardiac contraction and vascular smooth muscle tone",
        "effect": "contractility",
        "embedding_seed": 203,
    },
    "Beta1": {
        "full_name": "β1-Adrenergic Receptor",
        "role": "Controls heart rate and contractility via sympathetic nervous system",
        "effect": "heart_rate",
        "embedding_seed": 307,
    },
    "Beta2": {
        "full_name": "β2-Adrenergic Receptor",
        "role": "Controls bronchodilation and vasodilation",
        "effect": "vasodilation",
        "embedding_seed": 411,
    },
    "Alpha1": {
        "full_name": "α1-Adrenergic Receptor",
        "role": "Controls vascular smooth muscle contraction",
        "effect": "vasoconstriction",
        "embedding_seed": 509,
    },
    "ACE": {
        "full_name": "Angiotensin Converting Enzyme",
        "role": "Converts angiotensin I to II; inhibition lowers blood pressure",
        "effect": "blood_pressure",
        "embedding_seed": 613,
    },
    "AT1": {
        "full_name": "Angiotensin II Type 1 Receptor",
        "role": "Mediates vasoconstriction and aldosterone release",
        "effect": "blood_pressure",
        "embedding_seed": 719,
    },
    "COX1": {
        "full_name": "Cyclooxygenase-1",
        "role": "Produces thromboxane A2; inhibition reduces platelet aggregation",
        "effect": "platelet_aggregation",
        "embedding_seed": 823,
    },
    "NaK_ATPase": {
        "full_name": "Na+/K+ ATPase",
        "role": "Maintains ion gradient; inhibition increases intracellular Ca2+",
        "effect": "contractility",
        "embedding_seed": 929,
    },
    "PDE3": {
        "full_name": "Phosphodiesterase 3",
        "role": "Degrades cAMP; inhibition increases contractility",
        "effect": "contractility",
        "embedding_seed": 1031,
    },
}

# Known drug-target binding affinities (Ki in nM, lower = stronger)
KNOWN_INTERACTIONS = {
    "metoprolol": {"Beta1": 0.85, "Beta2": 0.15, "hERG": 0.05},
    "propranolol": {"Beta1": 0.80, "Beta2": 0.75, "hERG": 0.10},
    "atenolol": {"Beta1": 0.82, "Beta2": 0.10, "hERG": 0.03},
    "amlodipine": {"Cav1.2": 0.90, "Beta1": 0.05, "Alpha1": 0.10},
    "verapamil": {"Cav1.2": 0.85, "hERG": 0.35, "Beta1": 0.10},
    "diltiazem": {"Cav1.2": 0.80, "hERG": 0.15, "Nav1.5": 0.10},
    "nifedipine": {"Cav1.2": 0.88, "Alpha1": 0.05},
    "lisinopril": {"ACE": 0.92},
    "captopril": {"ACE": 0.85},
    "enalapril": {"ACE": 0.88},
    "losartan": {"AT1": 0.90},
    "amiodarone": {"hERG": 0.70, "Nav1.5": 0.55, "Cav1.2": 0.45, "Beta1": 0.30},
    "sotalol": {"hERG": 0.65, "Beta1": 0.60, "Beta2": 0.40},
    "digoxin": {"NaK_ATPase": 0.92, "hERG": 0.10},
    "warfarin": {"COX1": 0.05},
    "aspirin": {"COX1": 0.85},
    "doxorubicin": {"hERG": 0.40, "Cav1.2": 0.30, "NaK_ATPase": 0.25},
    "furosemide": {"hERG": 0.05},
    "caffeine": {"PDE3": 0.30, "Beta1": 0.10, "Beta2": 0.10},
    "ibuprofen": {"COX1": 0.75},
    "acetaminophen": {"COX1": 0.20},
}


def _generate_protein_embedding(target_name: str, dim: int = 128) -> np.ndarray:
    """Generate a deterministic protein embedding vector."""
    seed = PROTEIN_TARGETS[target_name]["embedding_seed"]
    rng = np.random.RandomState(seed)
    embedding = rng.randn(dim).astype(np.float32)
    return embedding / np.linalg.norm(embedding)


def _predict_interaction(drug_fingerprint: np.ndarray, protein_embedding: np.ndarray) -> float:
    """
    Predict drug-protein interaction strength using
    a simplified neural-network-like computation.
    Returns a score between 0 and 1.
    """
    # Dimensionality adaptation
    drug_dim = len(drug_fingerprint)
    prot_dim = len(protein_embedding)

    # Create interaction features
    # Down-project drug vector
    seed = int(abs(np.sum(protein_embedding[:4]) * 1000))
    rng = np.random.RandomState(seed % (2**31))
    W = rng.randn(prot_dim, drug_dim).astype(np.float32) * 0.01
    drug_proj = W @ drug_fingerprint

    # Bilinear interaction
    interaction = np.dot(drug_proj, protein_embedding)

    # Sigmoid activation
    score = 1.0 / (1.0 + np.exp(-interaction))

    return float(np.clip(score, 0.0, 1.0))


def predict_drug_target_interactions(drug_name: str, drug_fingerprint: np.ndarray, cmax_ugml: float = 1.0) -> dict:
    """
    Predict target occupancy based on drug intrinsic affinity and physiological PK concentration.
    
    Returns:
        Dictionary with target names as keys and dynamic interaction data as values
    """
    name_lower = drug_name.lower().strip()
    results = {}

    for target_name, target_info in PROTEIN_TARGETS.items():
        # Use known interaction if available (affinity)
        if name_lower in KNOWN_INTERACTIONS and target_name in KNOWN_INTERACTIONS[name_lower]:
            intrinsic_affinity = KNOWN_INTERACTIONS[name_lower][target_name]
            source = "experimental_database"
        else:
            protein_embedding = _generate_protein_embedding(target_name)
            intrinsic_affinity = _predict_interaction(drug_fingerprint, protein_embedding) * 0.3
            source = "predicted_model"
            
        # Convert intrinsic affinity and concentration to dynamic occupancy (Hill equation analog)
        # Higher affinity means lower concentration needed to achieve binding.
        # This scales the binding dynamically with the user's dose!
        effective_concentration = max(1e-4, cmax_ugml)
        ec50_target = max(0.01, 1.0 - intrinsic_affinity) * 2.0  # Approx nominal target EC50
        occupancy = (effective_concentration / (ec50_target + effective_concentration)) * intrinsic_affinity
        
        # Max out at 1.0 to prevent runaway parameters
        binding_strength = min(1.0, occupancy)

        results[target_name] = {
            "binding_strength": round(binding_strength, 4),
            "intrinsic_affinity": round(intrinsic_affinity, 4),
            "occupancy_pct": round(binding_strength * 100, 1),
            "protein_name": target_info["full_name"],
            "physiological_role": target_info["role"],
            "cardiac_effect": target_info["effect"],
            "prediction_source": source,
            "is_significant": binding_strength > 0.3,
        }

    # Sort by binding strength
    results = dict(sorted(results.items(), key=lambda x: x[1]["binding_strength"], reverse=True))

    # Determine primary targets
    primary_targets = [k for k, v in results.items() if v["binding_strength"] > 0.3]
    secondary_targets = [k for k, v in results.items() if 0.1 < v["binding_strength"] <= 0.3]

    max_binding = max(v["binding_strength"] for v in results.values()) if results else 0.0

    return {
        "drug_name": drug_name,
        "interactions": results,
        "primary_targets": primary_targets,
        "secondary_targets": secondary_targets,
        "n_significant_interactions": len(primary_targets),
        "max_binding_strength": float(max_binding),
        "selectivity_index": round(
            max_binding / (np.mean([v["binding_strength"] for v in results.values()]) + 1e-8), 2
        ),
    }
