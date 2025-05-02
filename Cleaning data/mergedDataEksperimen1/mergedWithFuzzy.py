import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import MolToSmiles
from difflib import get_close_matches

print("Memulai proses penggabungan data...")

# Load file hasil konversi (SMILES + Genotoxicity)
print("Membaca file genotoxic_with_smiles.csv...")
df_genotoxic = pd.read_csv("genotoxic_with_smiles.csv")
print(f"Data genotoxicitas dimuat: {df_genotoxic.shape[0]} baris, {df_genotoxic.shape[1]} kolom")

# Load file fitur molekul (SMILES + LogP, TPSA, dll.)
print("Membaca file molecule_list.json...")
with open("molecule_list.json", "r") as file:
    molecule_data = json.load(file)

df_molecules = pd.DataFrame(molecule_data)
print(f"Data fitur molekul dimuat: {df_molecules.shape[0]} baris, {df_molecules.shape[1]} kolom")

# Hapus duplikasi berdasarkan SMILES untuk menghindari ledakan data
print("Menghapus duplikasi pada molecule_list.json...")
df_molecules = df_molecules.drop_duplicates(subset=["SMILES"], keep="first")
print(f"Total data setelah menghapus duplikasi: {df_molecules.shape[0]} baris")

# Fungsi untuk mengubah SMILES menjadi format kanonik dengan RDKit
def canonicalize_smiles(smiles):
    if pd.isna(smiles) or smiles.strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol:
            return MolToSmiles(mol, canonical=True)
    except:
        pass  # Abaikan error dan lanjutkan
    
    # Coba tanpa sanitasi jika gagal
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol:
            return MolToSmiles(mol, canonical=True)
    except:
        pass  # Jika tetap gagal, kembalikan None
    
    return None  # Jika gagal total, buang data ini

# Normalisasi dan konversi SMILES ke format kanonik
print("Normalisasi dan konversi SMILES ke bentuk kanonik...")
df_genotoxic["SMILES"] = df_genotoxic["SMILES"].astype(str).str.strip().str.lower().apply(canonicalize_smiles)
df_molecules["SMILES"] = df_molecules["SMILES"].astype(str).str.strip().str.lower().apply(canonicalize_smiles)

# Cek jumlah SMILES unik setelah konversi
print(f"Jumlah SMILES unik di genotoxic_with_smiles.csv: {df_genotoxic['SMILES'].nunique()}")
print(f"Jumlah SMILES unik di molecule_list.json: {df_molecules['SMILES'].nunique()}")

# Tampilkan beberapa contoh SMILES setelah konversi
print("Contoh SMILES dari genotoxic_with_smiles.csv setelah konversi:")
print(df_genotoxic["SMILES"].dropna().head(10).tolist())
print("Contoh SMILES dari molecule_list.json setelah konversi:")
print(df_molecules["SMILES"].dropna().head(10).tolist())

# Gunakan fuzzy matching untuk mencari SMILES mirip
print("Mencari SMILES yang tidak cocok menggunakan fuzzy matching...")
mol_smiles_list = df_molecules["SMILES"].dropna().tolist()

def find_best_match(smiles):
    if isinstance(smiles, str): 
        matches = get_close_matches(smiles, mol_smiles_list, n=1, cutoff=0.8)
        return matches[0] if matches else None
    return None

df_genotoxic["Matched_SMILES"] = df_genotoxic["SMILES"].apply(find_best_match)

# Gabungkan berdasarkan kolom SMILES yang sudah dicocokkan
print("Menggabungkan data berdasarkan kolom SMILES...")
df_merged = df_genotoxic.merge(df_molecules, left_on="Matched_SMILES", right_on="SMILES", how="left")
print(f"Total data setelah penggabungan: {df_merged.shape[0]} baris, {df_merged.shape[1]} kolom")

# Cek data yang tidak tergabung
unmatched = df_merged[df_merged.isnull().any(axis=1)]
print(f"{len(unmatched)} senyawa tidak memiliki fitur tambahan.")

# Hapus baris dengan nilai kosong jika perlu
print("Menghapus baris dengan nilai kosong...")
df_merged.dropna(inplace=True)
print(f"Total data setelah pembersihan: {df_merged.shape[0]} baris")

# Simpan hasil gabungan
print("Menyimpan hasil penggabungan ke merged_genotoxic_data.csv...")
df_merged.to_csv("merged_genotoxic_data.csv", index=False)

print("Penggabungan selesai! File tersimpan sebagai merged_genotoxic_data.csv")
