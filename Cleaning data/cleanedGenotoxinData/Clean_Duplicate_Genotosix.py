import pandas as pd

def clean_genotoxic_data(input_file, output_file):
    # Baca data dari file CSV
    df = pd.read_csv(input_file)
    
    df.columns = ['Substance' ,'Genotoxicity']
    
    remove_status = {"Not determined", "No data", "Other", "(Blanks)", "Ambigous", "Not applicable"}
    
    # Kelompokkan berdasarkan nama senyawa
    grouped = df.groupby('Substance')
    
    # Filter data
    cleaned_data = []
    for Substance, group in grouped:
        statuses = set(group['Genotoxicity'])
        
        # Jika ada "Positive" atau "Negative", hapus status yang tidak relevan
        if "Positive" in statuses or "Negative" in statuses:
            filtered_group = group[~group['Genotoxicity'].isin(remove_status)]
        else:
            filtered_group = group
        
        cleaned_data.append(filtered_group)
    
    # Gabungkan kembali hasil filtering
    cleaned_df = pd.concat(cleaned_data)

    # Hapus duplikasi, simpan hanya kemunculan pertama
    cleaned_df = cleaned_df.drop_duplicates(subset=['Substance'], keep='first')

    # Simpan ke file Excel
    cleaned_df.to_excel(output_file, index=False)
    print(f"Data yang sudah dibersihkan disimpan dalam {output_file}")
    
# Contoh penggunaan
input_file = "Data Genotosix Raw.csv"  # Ganti dengan nama file asli Anda
output_file = "cleaned_genotoxic_data.xlsx"
clean_genotoxic_data(input_file, output_file)
