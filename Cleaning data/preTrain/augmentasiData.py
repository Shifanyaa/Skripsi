import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 1. Load dan filter data
df = pd.read_csv("molecule_list_from_genotoxic.csv")
df = df[~df['Genotoxicity'].isin(['No Data', 'Ambiguous'])].copy()
df['Label'] = df['Genotoxicity'].str.lower().isin(['positive', 'yes']).astype(int)

# 2. Fitur awal
features = [
    "LogP", "TPSA", "hbond_acceptors", "hbond_donors",
    "num_atoms", "num_bonds", "rotatable_bonds", "weight"
]
X = df[features].copy()
y = df['Label']

# 3. (Opsional) Tangani outliers dengan winsorizing di 1st–99th percentile
for col in features:
    lower, upper = np.percentile(X[col], [1, 99])
    X[col] = X[col].clip(lower=lower, upper=upper)

# 4. Buang fitur variansi rendah (<0.01)
vt = VarianceThreshold(threshold=0.01)
vt.fit(X)
mask = vt.get_support()  # boolean mask fitur yang lulus
X = X.loc[:, mask]
kept_features = X.columns.tolist()
print("Fitur setelah VarianceThreshold:", kept_features)

# 5. Buang fitur highly correlated (|corr| > 0.9)
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
X.drop(columns=to_drop, inplace=True)
print("Fitur yang di-drop karena korelasi tinggi:", to_drop)
print("Fitur akhir:", X.columns.tolist())

# 6. Scaling (MinMax tetap, atau ganti RobustScaler jika suka)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 7. SMOTE pada fitur terpilih
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# 8. Split: Train (70%) + Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_res, y_res,
    test_size=0.30,
    random_state=42,
    stratify=y_res
)

# 9. Split Temp: Validation (15%) & Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# 10. Simpan ke CSV
final_features = X.columns.tolist()
pd.DataFrame(X_train, columns=final_features).assign(Label=y_train)\
  .to_csv("train.csv", index=False)
pd.DataFrame(X_val,   columns=final_features).assign(Label=y_val)\
  .to_csv("val.csv",   index=False)
pd.DataFrame(X_test,  columns=final_features).assign(Label=y_test)\
  .to_csv("test.csv",  index=False)

print("✅ Preprocessing selesai.")
print("Fitur akhir digunakan:", final_features)
print("📁 train.csv (70%), val.csv (15%), test.csv (15%)")
