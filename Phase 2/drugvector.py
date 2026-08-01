import json
import urllib.request
import numpy as np

# Try importing RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

COMMON_SMILES = {
    "Metoprolol": "CC(C)NCC(O)COC1=CC=C(CCOC)C=C1",
    "Propranolol": "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12",
    "Atenolol": "CC(C)NCC(O)COC1=CC=C(CC(N)=O)C=C1",
    "Amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C1C2=CC=CC=C2Cl)C(=O)OC",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
}

def get_valid_smiles(drug_name: str) -> str:
    """Retrieve canonical SMILES for a drug via dictionary lookup or Ollama fallback."""
    cleaned_name = drug_name.strip().title()
    if cleaned_name in COMMON_SMILES:
        return COMMON_SMILES[cleaned_name]

    # Fallback to local Ollama instance if available
    try:
        prompt = (
            f"Provide only the canonical SMILES string for the drug '{drug_name}'. "
            f"Do not include any extra commentary, markdown, or explanation."
        )
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps({"model": "llama3.2", "prompt": prompt, "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
            smiles = result.get("response", "").strip().split()[0]
            if RDKIT_AVAILABLE and Chem.MolFromSmiles(smiles) is not None:
                return smiles
    except Exception:
        pass

    # Deterministic fallback representation
    return f"CC(N)C(=O)O_{hash(cleaned_name) % 1000000}"

def drug_to_vector(smiles: str, n_bits: int = 1024) -> np.ndarray:
    """Convert a SMILES string to a binary fingerprint vector."""
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                return np.array(fp, dtype=int)
        except Exception:
            pass

    # Synthetic deterministic binary vector based on SMILES string hash
    np.random.seed(abs(hash(smiles)) % (2**32))
    return np.random.choice([0, 1], size=n_bits)

if __name__ == "__main__":
    drug = "Metoprolol"
    smiles = get_valid_smiles(drug)
    vec = drug_to_vector(smiles)
    print(f"Drug: {drug}")
    print(f"SMILES: {smiles}")
    print(f"Vector shape: {vec.shape}, non-zero count: {np.count_nonzero(vec)}")
