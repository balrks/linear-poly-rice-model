import pandas as pd
import numpy as np
import os

# Menetapkan direktori dan nama file
directory = 'cleaned_data'
filename = 'cleaned_data_harga.csv'
cleaned_data_path = os.path.join(directory, filename)

# Pastikan direktori ada
os.makedirs(directory, exist_ok=True)

# Sisanya adalah logika pengolahan data Anda...
data_harga = pd.read_csv('data_harga.csv', delimiter=';')
data_harga.replace('-', np.nan, inplace=True)
data_harga['date'] = pd.to_datetime(data_harga['date'], errors='coerce', format='%d/ %m/ %Y')

# Konversi harga dan penyimpanan data
for col in data_harga.columns[1:]:
    data_harga[col] = data_harga[col].apply(lambda x: float(x.replace('Rp', '').replace(',', '')) if pd.notna(x) else x)

data_harga.to_csv(cleaned_data_path, index=False)
print(f"Data telah disimpan di: {cleaned_data_path}")
