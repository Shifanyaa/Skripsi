import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import inchi

print("Memulai proses penggabungan data dengan InChIKey (versi revisi)...")

# 1. Load file
df_genotoxic = pd.read_csv("genotoxic_with_smiles.csv")
with open("molecule_list.json", "r") as file:
    molecule_data = json.load(file)
df_molecules = pd.DataFrame(molecule_data)

# 2. Preprocessing awal
df_genotoxic["SMILES"] = df_genotoxic["SMILES"].astype(str).str.strip().str.lower()
df_molecules["SMILES"] = df_molecules["SMILES"].astype(str).str.strip().str.lower()

# 3. Fungsi konversi ke InChIKey
def smiles_to_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return inchi.MolToInchiKey(mol)
    except:
        return None
    return None

# 4. Generate InChIKey
print("Mengonversi SMILES menjadi InChIKey...")
df_genotoxic["InChIKey"] = df_genotoxic["SMILES"].apply(smiles_to_inchikey)
df_molecules["InChIKey"] = df_molecules["SMILES"].apply(smiles_to_inchikey)

# 5. Hapus duplikasi berdasarkan InChIKey (jaga 1 entry per struktur)
print("Menghapus duplikasi berdasarkan InChIKey di molecule_list.json...")
df_molecules = df_molecules.drop_duplicates(subset="InChIKey", keep="first")

# 6. Merge berdasarkan InChIKey
print("Menggabungkan data berdasarkan InChIKey...")
df_merged = df_genotoxic.merge(df_molecules, on="InChIKey", how="left")

# 7. Cek hasil
print(f"Total data setelah merge: {df_merged.shape[0]} baris")
unmatched_count = df_merged["LogP"].isna().sum()
print(f"Jumlah entri yang tidak cocok (tidak dapat data molekul): {unmatched_count}")

# 8. Hapus baris kosong (opsional)
df_merged_clean = df_merged.dropna()
print(f"Jumlah baris setelah pembersihan: {df_merged_clean.shape[0]}")

# 9. Simpan hasil
df_merged_clean.to_csv("merged_by_inchikey_clean.csv", index=False)
print("Selesai! File tersimpan sebagai merged_by_inchikey_clean.csv")
