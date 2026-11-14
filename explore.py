
import pandas as pd

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# 1. Display first 5 records
print("\n--- FIRST 5 RECORDS ---")
print(df.head())

# 2. Display all column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 3. Dataset information
print("\n--- DATASET INFORMATION ---")
print(df.info())

# 4. Statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 5. Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 6. Check diabetes distribution
print("\n--- OUTCOME DISTRIBUTION ---")
print(df["Outcome"].value_counts())

# 7. Dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)


# Check zero values in each column
print("\n--- ZERO VALUE ANALYSIS ---")

columns_to_check = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns_to_check:
    zero_count = (df[column] == 0).sum()
    print(f"{column}: {zero_count} zero values")

