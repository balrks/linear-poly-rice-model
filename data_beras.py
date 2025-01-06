import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Tentukan path ke file CSV
# file_path = 'cleaned_data_harga.csv'
# df = pd.read_csv(file_path) 

# print(df.head())

# Load your data
# data_path = 'cleaned_data_harga.csv'
# df = pd.read_csv(data_path)

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

# Load your data
data_path = 'cleaned_data_harga.csv'
df = pd.read_csv(data_path)

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Normalize 'date' directly in the same column by subtracting the minimum date and converting to days
df['date'] = (df['date'] - df['date'].min()).dt.days

# Select the normalized 'days_since_start' and 'Kota Lhokseumawe' columns
selected_data = df[['date', 'Kota Lhokseumawe']].copy()

# Fill missing values using linear interpolation
selected_data['Kota Lhokseumawe'] = selected_data['Kota Lhokseumawe'].interpolate(method='linear')

# Creating the independent variable (X) - using normalized days
X = selected_data['date'].values.reshape(-1, 1)  # Reshape for scikit-learn

# Creating the dependent variable (Y)
Y = selected_data['Kota Lhokseumawe'].values

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Create a linear regression model
# linear_model = LinearRegression()

# # Train the model using the training data
# linear_model.fit(X_train, Y_train)

# # Print the coefficients and the intercept of the model
# print("Slope (coefficients of the model):", linear_model.coef_)
# print("Intercept (bias of the model):", linear_model.intercept_)

poly_degree = 2
poly = PolynomialFeatures(degree=poly_degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train a Linear Regression model using polynomial features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)

# Print the coefficients and the intercept
print("Polynomial Regression (degree 2):")
print("Coefficients:", poly_model.coef_)
print("Intercept:", poly_model.intercept_)