
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

datasets = [
    "indian_railway_failure_detection_maintenance_v2.csv",
    "indian_railway_predictive_maintenance_100k.csv"
]

for filename in datasets:

    file_path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(f"DATASET: {filename}")
    print("=" * 70)

    if not file_path.exists():
        print("File not found:", file_path)
        continue

    df = pd.read_csv(file_path)

    print("\n1. Dataset Shape:")
    print(df.shape)

    print("\n2. Column Names:")
    for col in df.columns:
        print("-", col)

    print("\n3. First 5 Records:")
    print(df.head().to_string())

    print("\n4. Data Types:")
    print(df.dtypes)

    print("\n5. Missing Values:")
    print(df.isnull().sum())

    print("\n6. Duplicate Records:")
    print(df.duplicated().sum())

    print("\n7. Statistical Summary:")
    print(df.describe(include="all").T)

    print("\n8. Possible Target Columns:")
    for col in df.columns:
        if any(word in col.lower() for word in
               ["failure", "maintenance", "defect", "target"]):
            print("-", col)

print("\nDATASET INSPECTION COMPLETED")
