import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

#Ubah ke csv
with open("molecule_list.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)
df.to_csv("molecule_list.csv", index=False)

#Baca csv
dataMolekul = pd.read_csv("molecule_list.csv")
dataGenotoxic= pd.read_csv("genotoxic_cleaned_final.csv")
dataGenotosixSmiles = pd.read_csv("genotoxic_with_smiles1.csv")
dataMerged1 = pd.read_csv("merged_genotoxic_data.csv")

# #Cek data
# print(dataMolekul.head())
# print(dataGenotoxic.head())
# print(dataGenotosixSmiles.head())
# print(dataMerged1.head())

#cek kolom
print(dataMolekul.shape)
print(dataGenotoxic.shape)
print(dataGenotosixSmiles.shape)
print(dataMerged1.shape)

# # Hapus duplikasi
# dataGenotoxic['Genotoxicity'] = dataGenotoxic["Genotoxicity"].fillna('Unknown')
# priority = {'Positive': 1, 'Negative': 2, 'Unknown': 3}
# dataGenotoxic['priority'] = dataGenotoxic['Genotoxicity'].map(priority)
# dataGenotoxic_sorted = dataGenotoxic.sort_values(by=['Substance', 'priority'])
# dataGenotoxic_cleaned = dataGenotoxic_sorted.drop_duplicates(subset='Substance', keep='first')
# dataGenotoxic_cleaned = dataGenotoxic_cleaned.drop(columns='priority')
# dataGenotoxic_cleaned.to_csv('genotoxic_cleaned_final.csv', index=False)

#Distribusi nilai
# Pie chart
sns.countplot(data=dataGenotoxic, x='Genotoxicity')
plt.title('Distribusi Genotoxicity')
plt.show()
