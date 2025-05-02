from rdkit import Chem
from rdkit.Chem import MolToSmiles
import pandas as pd

# Fungsi untuk konversi ke canonical SMILES
def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None
df_genotoxic_clean = pd.read_csv("Genotoxic Smiles.csv")
# Terapkan ke kolom SMILES
df_genotoxic_clean["SMILES"] = df_genotoxic_clean["SMILES"].astype(str).str.strip().apply(canonicalize)

# Simpan hasil
df_genotoxic_clean.dropna(subset=["SMILES"], inplace=True)
df_genotoxic_clean.to_csv("Genotoxic_Smiles_Canonical.csv", index=False)
print("✅ SMILES telah dinormalisasi dan disimpan.")
