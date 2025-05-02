import requests
import json

# # Mengambil daftar lengkap SMILES strings
# response = requests.get("https://smilesdb.org/api/smiles/full")
# smiles_list = response.json()

# # Menyimpan data ke file JSON
# with open('smiles_full.json', 'w') as file:
#     json.dump(smiles_list, file, indent=4)

# print("Data telah disimpan ke smiles_full.json")

# # Mengambil 5 SMILES strings acak
# response = requests.get("https://smilesdb.org/api/smiles/random/5")
# random_smiles = response.json()

# # Menyimpan data ke file JSON
# with open('random_smiles.json', 'w') as file:
#     json.dump(random_smiles, file, indent=4)

# print("Data telah disimpan ke random_smiles.json")

# Mengambil daftar lengkap molekul
response = requests.get("https://smilesdb.org/api/full")
molecule_list = response.json()

# Menyimpan data ke file JSON
with open('molecule_list.json', 'w') as file:
    json.dump(molecule_list, file, indent=4)

print("Data telah disimpan ke molecule_list.json")