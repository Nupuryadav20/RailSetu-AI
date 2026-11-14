
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_selection import mutual_info_classif

# Load training data only
DATA_DIR = Path("data/processed/model_ready")

X = pd.read_csv(DATA_DIR / "X_train.csv")
y = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

print("\n========== FEATURE AUDIT ==========")

# Identify feature types
numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

# Prepare data for mutual information
X_audit = X.copy()

for col in numeric_cols:
    X_audit[col] = X_audit[col].fillna(
        X_audit[col].median()
    )

for col in categorical_cols:
    X_audit[col] = X_audit[col].fillna(
        X_audit[col].mode()[0]
    )
    X_audit[col] = X_audit[col].astype("category").cat.codes

# Mutual information
discrete_mask = [
    col in categorical_cols
    for col in X_audit.columns
]

mi_scores = mutual_info_classif(
    X_audit,
    y,
    discrete_features=discrete_mask,
    random_state=42
)

mi_df = pd.DataFrame({
    "Feature": X_audit.columns,
    "Mutual Information": mi_scores
})

mi_df = mi_df.sort_values(
    by="Mutual Information",
    ascending=False
)

print("\nTOP 15 FEATURES BY MUTUAL INFORMATION:")
print(mi_df.head(15).to_string(index=False))

# Numerical feature analysis
print("\nNUMERICAL FEATURES BY TARGET:")

original_df = X.copy()
original_df["maintenance_required"] = y.values

for col in numeric_cols:
    summary = original_df.groupby(
        "maintenance_required"
    )[col].agg(["mean", "median"])

    print(f"\n{col}")
    print(summary)

# Categorical feature analysis
print("\nCATEGORICAL MAINTENANCE RATES:")

for col in categorical_cols:
    print(f"\n{col}")

    rates = original_df.groupby(
        col,
        observed=True
    )["maintenance_required"].agg(
        ["count", "mean"]
    )

    print(rates.sort_values("mean", ascending=False))

# Save audit
output_dir = Path("data/processed/eda")
output_dir.mkdir(parents=True, exist_ok=True)

mi_df.to_csv(
    output_dir / "mutual_information.csv",
    index=False
)

print("\nFEATURE AUDIT COMPLETED!")
print("Results saved to:", output_dir)
