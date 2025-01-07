import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

# Tentukan path ke file CSV
file_path = 'data_harga_beras_2020_2024.csv'
df = pd.read_csv(file_path) 
df

print(df.head())

# # Load your data
# data_path = 'cleaned_data_harga.csv'
# df = pd.read_csv(data_path)
# import pandas as pd


# # Select only the 'date' and 'Kota Lhokseumawe' columns
# selected_data = df[['date', 'Kota Lhokseumawe']]

# # Display the first few rows to verify the selection
# print(selected_data.head())

# # Check for missing values
# missing_values = selected_data.isnull().sum()

# # Print the number of missing values in each column
# print(missing_values)

# num_rows = selected_data.shape[0]

# # Print the number of rows
# print("Number of rows in the dataset:", num_rows)

# selected_data = df[['date', 'Kota Lhokseumawe']].copy()

# # Check where the missing values were
# missing_before = selected_data[selected_data['Kota Lhokseumawe'].isnull()]

# # Fill missing values using linear interpolation
# selected_data['Kota Lhokseumawe'] = selected_data['Kota Lhokseumawe'].interpolate(method='linear')

# # Display rows where missing values were filled
# missing_after = selected_data.loc[missing_before.index]

# print("Data before interpolation:\n", missing_before)
# print("\nData after interpolation:\n", missing_after)

# # Load your data
# data_path = 'cleaned_data_harga.csv'
# df = pd.read_csv(data_path)

# # 2. Convert 'date' column to datetime
# df['date'] = pd.to_datetime(df['date'])

# # 3. Store the minimum date
# start_date = df['date'].min()

# # 4. Normalize 'date' by subtracting the minimum date to get days since start
# df['date'] = (df['date'] - start_date).dt.days

# # 5. Fill missing values using linear interpolation
# df['Kota Lhokseumawe'] = df['Kota Lhokseumawe'].interpolate(method='linear')

# # 6. Prepare data for regression
# X = df['date'].values.reshape(-1, 1)  # Independent variable (Days since start)
# Y = df['Kota Lhokseumawe'].values     # Dependent variable (Rice price)

# # Split the data into train and test sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Create a linear regression model
# linear_model = LinearRegression()

# # Train the model using the training data
# linear_model.fit(X_train, Y_train)

# # # Print the coefficients and the intercept of the model
# # print("Slope (coefficients of the model):", linear_model.coef_)
# # print("Intercept (bias of the model):", linear_model.intercept_)

# # Polynomial degree
# poly_degree = 2
# poly = PolynomialFeatures(degree=poly_degree)

# # Transform the features for polynomial regression
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)

# # Train a Linear Regression model using polynomial features
# poly_model = LinearRegression()
# poly_model.fit(X_train_poly, Y_train)


# # Print the coefficients and the intercept
# print("Polynomial Regression (degree 2):")
# print("Coefficients:", poly_model.coef_)
# print("Intercept:", poly_model.intercept_)

# # Predict using the linear model
# Y_pred = linear_model.predict(X)

# Konversi kembali kolom 'date' ke format datetime untuk sumbu x
# df['date_actual'] = start_date + pd.to_timedelta(df['date'], unit='D')

# # Buat scatter plot untuk data asli
# scatter_plot = go.Scatter(
#     x=df['date_actual'],  # Gunakan tanggal asli
#     y=df['Kota Lhokseumawe'], 
#     mode='markers', 
#     name='Data Asli', 
#     marker=dict(color='blue')
# )

# # Buat line plot untuk prediksi regresi linear
# line_plot = go.Scatter(
#     x=df['date_actual'],  # Gunakan tanggal asli
#     y=Y_pred, 
#     mode='lines', 
#     name='Regresi Linear', 
#     line=dict(color='red')
# )

# # Gabungkan kedua plot dalam satu layout
# layout = go.Layout(
#     title='Regresi Linear Harga Beras di Kota Lhokseumawe',
#     xaxis=dict(title='Tanggal'),
#     yaxis=dict(title='Harga Beras'),
#     legend=dict(x=0, y=1)
# )

# # Buat figure
# fig = go.Figure(data=[scatter_plot, line_plot], layout=layout)

# # Tampilkan plot
# fig.show()


# # 12. Print the model coefficients and intercept
# print("Slope (Coefficient of the model):", linear_model.coef_[0])
# print("Intercept (Bias of the model):", linear_model.intercept_)


# # Transformasi data uji ke dalam bentuk polinomial
# X_poly = poly.transform(X)  # Apply transformation to the entire dataset
# Y_pred_poly = poly_model.predict(X_poly)

# # Convert 'date' back to actual datetime format for x-axis in plot
# df['date_actual'] = start_date + pd.to_timedelta(df['date'], unit='D')

# # Create scatter plot for actual data
# scatter_plot = go.Scatter(
#     x=df['date_actual'],  # Use actual date
#     y=df['Kota Lhokseumawe'], 
#     mode='markers', 
#     name='Data Asli', 
#     marker=dict(color='blue')
# )

# # Create line plot for polynomial regression predictions
# line_plot = go.Scatter(
#     x=df['date_actual'],  # Use actual date
#     y=Y_pred_poly, 
#     mode='lines', 
#     name='Regresi Polinomial', 
#     line=dict(color='red')
# )

# # Combine both plots in a single layout
# layout = go.Layout(
#     title='Regresi Polinomial Harga Beras di Kota Lhokseumawe',
#     xaxis=dict(title='Tanggal'),
#     yaxis=dict(title='Harga Beras'),
#     legend=dict(x=0, y=1)
# )

# # Create figure
# fig = go.Figure(data=[scatter_plot, line_plot], layout=layout)

# # Display the plot
# fig.show()

# # Tampilkan hasil prediksi
# # print("Prediksi dengan model regresi polinomial:")
# # print(Y_pred_poly[:5])  # Menampilkan 5 prediksi pertama

# # # Jika ingin membandingkan dengan nilai aktual
# # print("\nNilai aktual:")
# # print(Y_test[:5])  # Menampilkan 5 nilai aktual pertama


# # Hitung MAPE
# mape = mean_absolute_percentage_error(Y_test, Y_pred_poly)

# # Cetak hasil
# print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# # Plot nilai aktual dan prediksi
# plt.figure(figsize=(10, 6))
# plt.plot(Y_test, label="Nilai Aktual", marker='o')
# plt.plot(Y_pred_poly, label="Prediksi Polinomial", linestyle='--', marker='x')
# plt.legend()
# plt.title("Perbandingan Nilai Aktual dan Prediksi (Regresi Polinomial)")
# plt.xlabel("Index")
# plt.ylabel("Harga Beras")
# plt.show()

# Prediksi dengan model regresi linear
# Y_pred_linear = linear_model.predict(X_test)

# # Hitung MAPE untuk model regresi linear
# mape_linear = mean_absolute_percentage_error(Y_test, Y_pred_linear)
# print(f"MAPE untuk model regresi linear: {mape_linear * 100:.2f}%")

# # Prediksi dengan model regresi polinomial
# # Transformasi data uji untuk model polinomial
# X_test_poly = poly.transform(X_test)
# Y_pred_poly = poly_model.predict(X_test_poly)

# # # Hitung MAPE untuk model regresi polinomial
# # mape_poly = mean_absolute_percentage_error(Y_test, Y_pred_poly)
# # print(f"MAPE untuk model regresi polinomial: {mape_poly * 100:.2f}%")

# # # Bandingkan model
# # if mape_linear < mape_poly:
# #     print("Model regresi linear memiliki performa lebih baik berdasarkan MAPE.")
# # elif mape_linear > mape_poly:
# #     print("Model regresi polinomial memiliki performa lebih baik berdasarkan MAPE.")
# # else:
# #     print("Kedua model memiliki performa yang sama berdasarkan MAPE.")

# # Plot data aktual
# plt.figure(figsize=(12, 6))
# plt.scatter(X_test, Y_test, color='blue', label='Data Aktual', alpha=0.6)

# # Plot prediksi regresi linear
# plt.plot(X_test, Y_pred_linear, color='red', label='Regresi Linear')

# # Plot prediksi regresi polinomial
# plt.plot(X_test, Y_pred_poly, color='green', label='Regresi Polinomial')

# # Tambahkan informasi pada grafik
# plt.title('Perbandingan Prediksi Regresi Linear dan Polinomial dengan Data Aktual')
# plt.xlabel('Tanggal (Ordinal)')
# plt.ylabel('Harga Beras')
# plt.legend()
# plt.grid(True)

# # Tampilkan grafik
# plt.show()