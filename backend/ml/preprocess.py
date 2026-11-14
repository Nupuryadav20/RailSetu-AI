
import pandas as pd
from pathlib import Path

# Paths
RAW_PATH = Path("data/raw/indian_railway_predictive_maintenance_100k.csv")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original dataset
df = pd.read_csv(RAW_PATH)

print("Original shape:", df.shape)

# 1. Remove duplicate rows
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# 2. Check target distribution
print("\nTarget distribution:")
print(df["maintenance_required"].value_counts())

# 3. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Save cleaned dataset
output_path = OUTPUT_DIR / "railway_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:", output_path)
print("\nPreprocessing completed successfully!")
