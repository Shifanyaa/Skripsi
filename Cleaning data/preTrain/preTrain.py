import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("molecule_list_from_genotoxic.csv")

# Filter hanya label Positive dan Negative
df = df[df["Genotoxicity"].isin(["Positive", "Negative"])]

# Hapus duplikat berdasarkan Canonical_SMILES
df.drop_duplicates(subset=["Canonical_SMILES"], inplace=True)

# Tentukan fitur numerik
numeric_features = [
    "LogP", "TPSA", "hbond_acceptors", "hbond_donors",
    "num_atoms", "num_bonds", "rotatable_bonds", "weight"
]

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_features])
y = df["Genotoxicity"].map({"Positive": 1, "Negative": 0})

# Split 60% train, 40% temp
X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
    X_scaled, y, df.index, test_size=0.4, stratify=y, random_state=42
)

# Gabungkan X_temp dan y_temp untuk menjaga indeks
temp_df = pd.DataFrame(X_temp, columns=numeric_features)
temp_df["Label"] = y_temp.values
temp_df["Index"] = idx_temp  # indeks asli dari df

# Split 20% val dan 20% test dari temp
val_df_temp, test_df_temp = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=42
)

# Ekstrak kembali nilai-nilai untuk fungsi build_df()
X_val = val_df_temp[numeric_features].values
y_val = val_df_temp["Label"]
idx_val = val_df_temp["Index"]

X_test = test_df_temp[numeric_features].values
y_test = test_df_temp["Label"]
idx_test = test_df_temp["Index"]

# Fungsi bantu untuk membuat DataFrame akhir
def build_df(X, y, idx):
    return pd.DataFrame(X, columns=numeric_features).assign(
        Label=y.values,
        Substance=df.loc[idx, "Substance"].values,
        Canonical_SMILES=df.loc[idx, "Canonical_SMILES"].values,
        formula=df.loc[idx, "formula"].values
    )

# Bangun DataFrame final
train_df = build_df(X_train, y_train, idx_train)
val_df = build_df(X_val, y_val, idx_val)
test_df = build_df(X_test, y_test, idx_test)

# Simpan ke CSV
train_df.to_csv("train_data.csv", index=False)
val_df.to_csv("val_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Selesai: train_data.csv (60%), val_data.csv (20%), test_data.csv (20%) telah disimpan.")
