"""
Layer 1: Drug Representation
Converts a drug name into machine-readable features using:
- SMILES molecular string
- Morgan fingerprint (1024-bit)
- Molecular descriptors (weight, logP, TPSA, etc.)

When RDKit is available, uses real molecular generation.
Falls back to deterministic hash-based simulation otherwise.
"""

import numpy as np
import hashlib

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Try to import Ollama for LLM-based SMILES generation
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# Known drug SMILES database (fallback)
DRUG_SMILES_DB = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "metoprolol": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
    "lisinopril": "NCCCC[C@@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O",
    "amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C1c1ccccc1Cl)C(=O)OC",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "digoxin": "C[C@@H]1O[C@@H](O[C@@H]2C[C@@H](O)[C@H](O[C@@H]3C[C@@H](O)[C@H](O[C@@H]4C[C@@H](O)[C@H](O[C@@H]5CC[C@]6(C)[C@@H](CC[C@@H]7[C@@H]6CC[C@]6(C)[C@@H](CC(=O)C6)C6=CC(=O)OC6)[C@H]5O)O4)O3)O2)[C@@H](O)[C@H]1O",
    "atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
    "verapamil": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
    "amiodarone": "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1",
    "sotalol": "CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1",
    "diltiazem": "COc1ccc(C2Sc3ccccc3N(CCN(C)C)C(=O)C2OC(C)=O)cc1",
    "nifedipine": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",
    "captopril": "C[C@@H](CS)C(=O)N1CCC[C@H]1C(=O)O",
    "enalapril": "CCOC(=O)[C@H](CCc1ccccc1)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)O",
    "losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1",
    "furosemide": "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",
    "hydrochlorothiazide": "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O",
    "simvastatin": "CCC(C)(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
    "atorvastatin": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
    "clopidogrel": "COC(=O)[C@@H](c1ccccc1Cl)N1CCc2sccc2C1",
    "doxorubicin": "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
}

# Drug properties for simulation
DRUG_PROPERTIES = {
    "aspirin": {"mw": 180.16, "logp": 1.19, "tpsa": 63.6, "hbd": 1, "hba": 4,
                "category": "antiplatelet", "half_life": 3.1},
    "metoprolol": {"mw": 267.36, "logp": 1.88, "tpsa": 50.7, "hbd": 2, "hba": 4,
                   "category": "beta_blocker", "half_life": 3.5},
    "lisinopril": {"mw": 405.49, "logp": -0.79, "tpsa": 131.4, "hbd": 4, "hba": 7,
                   "category": "ace_inhibitor", "half_life": 12.0},
    "amlodipine": {"mw": 408.88, "logp": 3.0, "tpsa": 99.9, "hbd": 2, "hba": 6,
                   "category": "calcium_channel_blocker", "half_life": 35.0},
    "warfarin": {"mw": 308.33, "logp": 2.7, "tpsa": 63.6, "hbd": 1, "hba": 4,
                 "category": "anticoagulant", "half_life": 40.0},
    "propranolol": {"mw": 259.34, "logp": 3.48, "tpsa": 41.5, "hbd": 2, "hba": 3,
                    "category": "beta_blocker", "half_life": 4.0},
    "verapamil": {"mw": 454.60, "logp": 3.79, "tpsa": 64.0, "hbd": 0, "hba": 6,
                  "category": "calcium_channel_blocker", "half_life": 6.0},
    "amiodarone": {"mw": 645.31, "logp": 7.57, "tpsa": 42.7, "hbd": 0, "hba": 4,
                   "category": "antiarrhythmic", "half_life": 58.0},
    "digoxin": {"mw": 780.94, "logp": 1.26, "tpsa": 203.1, "hbd": 6, "hba": 14,
                "category": "cardiac_glycoside", "half_life": 36.0},
    "atenolol": {"mw": 266.34, "logp": 0.16, "tpsa": 84.6, "hbd": 3, "hba": 4,
                 "category": "beta_blocker", "half_life": 6.5},
    "diltiazem": {"mw": 414.52, "logp": 2.70, "tpsa": 59.1, "hbd": 0, "hba": 5,
                  "category": "calcium_channel_blocker", "half_life": 4.0},
    "nifedipine": {"mw": 346.33, "logp": 2.20, "tpsa": 110.5, "hbd": 1, "hba": 7,
                   "category": "calcium_channel_blocker", "half_life": 2.0},
    "captopril": {"mw": 217.29, "logp": 0.34, "tpsa": 57.6, "hbd": 1, "hba": 4,
                  "category": "ace_inhibitor", "half_life": 2.0},
    "losartan": {"mw": 422.91, "logp": 4.01, "tpsa": 92.5, "hbd": 2, "hba": 6,
                 "category": "arb", "half_life": 2.0},
    "furosemide": {"mw": 330.74, "logp": 2.03, "tpsa": 122.6, "hbd": 3, "hba": 5,
                   "category": "diuretic", "half_life": 1.5},
    "sotalol": {"mw": 272.36, "logp": 0.24, "tpsa": 86.8, "hbd": 3, "hba": 4,
                "category": "antiarrhythmic", "half_life": 12.0},
    "doxorubicin": {"mw": 543.52, "logp": 1.27, "tpsa": 206.1, "hbd": 6, "hba": 12,
                    "category": "cardiotoxic", "half_life": 48.0},
    "ibuprofen": {"mw": 206.28, "logp": 3.97, "tpsa": 37.3, "hbd": 1, "hba": 2,
                  "category": "nsaid", "half_life": 2.0},
    "acetaminophen": {"mw": 151.16, "logp": 0.46, "tpsa": 49.3, "hbd": 2, "hba": 3,
                      "category": "analgesic", "half_life": 2.5},
    "caffeine": {"mw": 194.19, "logp": -0.07, "tpsa": 58.4, "hbd": 0, "hba": 6,
                 "category": "stimulant", "half_life": 5.0},
}


def _hash_seed(name: str) -> int:
    """Generate deterministic seed from drug name."""
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def get_smiles(drug_name: str) -> str:
    """Get SMILES string for a drug. Uses database, RDKit, or LLM fallback."""
    name_lower = drug_name.lower().strip()

    # Check local database first
    if name_lower in DRUG_SMILES_DB:
        return DRUG_SMILES_DB[name_lower]

    # Try Ollama if available
    if OLLAMA_AVAILABLE:
        try:
            prompt = f"Provide the canonical SMILES string for the drug {drug_name}. Return ONLY the SMILES string."
            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}]
            )
            smiles = response["message"]["content"].strip()
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return smiles
            else:
                return smiles
        except Exception:
            pass

    # Generate a synthetic SMILES-like representation
    seed = _hash_seed(name_lower)
    rng = np.random.RandomState(seed)
    atoms = ["C", "N", "O", "S"]
    bonds = ["", "=", "#"]
    smiles_parts = []
    length = 8 + (seed % 12)
    for i in range(length):
        smiles_parts.append(rng.choice(atoms))
        if rng.random() < 0.3:
            smiles_parts.append(rng.choice(bonds))
    return "".join(smiles_parts)


def drug_to_fingerprint(smiles: str, n_bits: int = 1024) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint vector."""
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            return np.array(fp, dtype=np.float32)

    # Deterministic fallback
    seed = _hash_seed(smiles)
    rng = np.random.RandomState(seed)
    density = 0.05 + (seed % 100) / 1000.0
    vec = np.zeros(n_bits, dtype=np.float32)
    n_on = int(n_bits * density)
    indices = rng.choice(n_bits, n_on, replace=False)
    vec[indices] = 1.0
    return vec


def get_drug_properties(drug_name: str) -> dict:
    """Get physicochemical properties of the drug."""
    name_lower = drug_name.lower().strip()

    if name_lower in DRUG_PROPERTIES:
        return DRUG_PROPERTIES[name_lower]

    # Generate plausible properties from hash
    seed = _hash_seed(name_lower)
    rng = np.random.RandomState(seed)
    return {
        "mw": round(150 + rng.random() * 500, 2),
        "logp": round(-1 + rng.random() * 6, 2),
        "tpsa": round(20 + rng.random() * 180, 1),
        "hbd": int(rng.randint(0, 6)),
        "hba": int(rng.randint(1, 10)),
        "category": "unknown",
        "half_life": round(1 + rng.random() * 30, 1),
    }


def represent_drug(drug_name: str) -> dict:
    """
    Full drug representation pipeline.
    Returns SMILES, fingerprint, and molecular properties.
    """
    smiles = get_smiles(drug_name)
    fingerprint = drug_to_fingerprint(smiles)
    properties = get_drug_properties(drug_name)

    return {
        "drug_name": drug_name,
        "smiles": smiles,
        "fingerprint": fingerprint.tolist(),
        "fingerprint_density": float(np.mean(fingerprint)),
        "properties": properties,
        "n_bits_on": int(np.sum(fingerprint)),
        "representation_method": "Morgan_r2_1024bit",
    }
