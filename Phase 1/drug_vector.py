import ollama
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def get_valid_smiles(drug_name):

    prompt = f"""
    Provide the canonical SMILES string for the drug {drug_name}.
    Return ONLY the SMILES string.
    Do NOT return molecular formula.
    """

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    smiles = response["message"]["content"].strip()

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print("LLM returned invalid SMILES:", smiles)
        return None

    return smiles


def drug_to_vector(smiles):

    mol = Chem.MolFromSmiles(smiles)

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=1024
    )

    vector = np.array(fingerprint)

    return vector


def main():

    drug = input("Enter drug name: ")

    smiles = get_valid_smiles(drug)

    if smiles is None:
        print("Could not generate valid SMILES")
        return

    print("\nSMILES:", smiles)

    vector = drug_to_vector(smiles)

    print("\nVector length:", len(vector))
    print("\nVector values:\n")

    for v in vector:
        print(v, end=" ")


if __name__ == "__main__":
    main()