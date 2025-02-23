# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 2: Verify the Current Working Directory
print("Current Working Directory:", os.getcwd())

# Step 3: Define File Path
file_path = "../Datasets/Mall_Customers.csv"  # Adjust if needed

# Step 4: Check If File Exists
if os.path.exists(file_path):
    print("✅ File found! Loading dataset...")
else:
    print("❌ File not found. Check the file path.")
    exit()  # Stop execution if file is missing

# Step 5: Load the Dataset
df = pd.read_csv(file_path)

# Step 6: Display Basic Information
print("\nDataset Preview:")
print(df.head())  # Show first few rows

# Step 7: Check for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())  # Check for missing data

# Step 8: Drop Unnecessary Columns
df = df.drop(columns=["CustomerID"])  # Remove CustomerID as it's not needed

# Step 9: Encode Categorical Data (Optional)
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])  # Convert 'Male' to 1, 'Female' to 0

# Step 10: Standardize Numerical Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])

# Convert the scaled data into a DataFrame
df_scaled = pd.DataFrame(scaled_features, columns=["Age", "Annual Income", "Spending Score"])

# Step 11: Include Gender Column (Optional)
df_scaled["Gender"] = df["Gender"]  # Add back encoded gender if needed

# Step 12: Display Preprocessed Data
print("\nPreprocessed Dataset Preview:")
print(df_scaled.head())

# Step 13: Save Preprocessed Data
output_path = "../Datasets/Mall_Customers_Preprocessed.csv"  # Adjust if needed
df_scaled.to_csv(output_path, index=False)

print(f"\n✅ Data Preprocessing Completed! Preprocessed file saved at: {output_path}")
