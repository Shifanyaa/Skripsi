from rdkit import Chem
from rdkit.Chem import MolToSmiles
import pandas as pd

df = pd.read_csv("genotoxic_with_smiles.csv")
df["original_SMILES"] = df["SMILES"]

def canonicalize_smiles(smiles):
    if pd.isna(smiles) or smiles.strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol:
            return MolToSmiles(mol, canonical=True)
    except:
        pass
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol:
            return MolToSmiles(mol, canonical=True)
    except:
        pass
    return None

df["canonical_SMILES"] = df["SMILES"].astype(str).str.strip().str.lower().apply(canonicalize_smiles)

# Statistik
total = len(df)
valid = df["canonical_SMILES"].notna().sum()
invalid = total - valid

print(f"Total SMILES: {total}")
print(f"Berhasil dikonversi: {valid}")
print(f"Gagal dikonversi: {invalid}")
print("Contoh SMILES yang gagal:")
print(df[df["canonical_SMILES"].isna()]["original_SMILES"].head(10).tolist())
