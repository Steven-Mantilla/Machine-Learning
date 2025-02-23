import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ✅ Step 1: Load the dataset
file_path = "Datasets/Mall_Customers.csv"  # Ensure this path is correct
df = pd.read_csv(file_path)

# ✅ Step 2: Check for missing values
print("🔍 Checking for missing values...\n", df.isnull().sum())

# ✅ Step 3: Handle missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# ✅ Step 4: Encode categorical features (Gender)
if "Gender" in df.columns:
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])  # Male → 1, Female → 0
else:
    print("⚠️ Warning: 'Gender' column not found. Skipping encoding.")

# ✅ Step 5: Scale numerical features
scaler = StandardScaler()
numerical_columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

if all(col in df.columns for col in numerical_columns):
    df_scaled = df.copy()
    df_scaled[numerical_columns] = scaler.fit_transform(df[numerical_columns])
else:
    print("⚠️ Warning: Some numerical columns are missing!")

# ✅ Step 6: Ensure output directory exists
output_dir = "Datasets"
output_path = os.path.join(output_dir, "Mall_Customers_Preprocessed.csv")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📂 Created directory: {output_dir}")

# ✅ Step 7: Save the preprocessed file
df_scaled.to_csv(output_path, index=False)
print(f"✅ Preprocessed file saved at: {output_path}")

# ✅ Step 8: Display dataset preview
print("\n📝 Preprocessed Dataset Preview:")
print(df_scaled.head())
