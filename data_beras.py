import pandas as pd

# Tentukan path ke file CSV
file_path = 'cleaned_data_harga.csv'
df = pd.read_csv(file_path, delimiter=';') 

print(df.head())
