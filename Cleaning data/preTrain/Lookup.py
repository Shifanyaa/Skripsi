import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 1. Load dan filter data
df = pd.read_csv("molecule_list_from_genotoxic.csv")
df = df[~df['Genotoxicity'].isin(['No Data', 'Ambiguous'])].copy()
df['Label'] = df['Genotoxicity'].apply(lambda x: 1 if x.lower() in ['positive', 'yes'] else 0)

# 2. Fitur numerik dan label
features = [
    "LogP", "TPSA", "hbond_acceptors", "hbond_donors",
    "num_atoms", "num_bonds", "rotatable_bonds", "weight"
]
X = df[features]
y = df["Label"]

# 3. Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 5. Buat Lookup Table
df_original_meta = df[["Substance", "Canonical_SMILES", "formula", "Label"]]

# Minoritas asli
df_minority = df[df["Label"] == 1]

# Hitung berapa banyak data minoritas sintetis
repeat_count = y_resampled.tolist().count(1) - len(df_minority)

# Gandakan metadata minoritas
df_minority_aug = pd.concat([df_minority] * ((repeat_count // len(df_minority)) + 1), ignore_index=True).iloc[:repeat_count]

# Gabung dengan mayoritas dan minoritas asli
df_meta_resampled = pd.concat([df[df["Label"] == 0], df_minority, df_minority_aug], ignore_index=True).reset_index(drop=True)

# Tambahkan index untuk pencocokan
df_meta_resampled["index_model_input"] = df_meta_resampled.index

# Simpan Lookup Table
df_meta_resampled.to_csv("lookup_table.csv", index=False)
print("🔍 Lookup table disimpan sebagai lookup_table.csv")
