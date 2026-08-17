import pandas as pd

# Load the Kaggle dataset
df = pd.read_csv("data/train.csv")

print("====================================")
print("      ESTATEAI DATASET CHECK")
print("====================================")

print("\nDataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())