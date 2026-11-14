
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/processed/railway_cleaned.csv")

print("Initial dataset:", df.shape)

# Target variable
TARGET = "maintenance_required"

# Exclude identifiers and potentially leakage-prone columns
excluded_columns = [
    "train_id",
    "failure_type",
    "failure_severity",
    "risk_score"
]

X = df.drop(columns=[TARGET] + excluded_columns)
y = df[TARGET]

# Identify feature types
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("\nCategorical features:", categorical_features)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create output directory
output_dir = Path("data/processed/model_ready")
output_dir.mkdir(parents=True, exist_ok=True)

# Save splits
X_train.to_csv(output_dir / "X_train.csv", index=False)
X_test.to_csv(output_dir / "X_test.csv", index=False)

y_train.to_csv(output_dir / "y_train.csv", index=False)
y_test.to_csv(output_dir / "y_test.csv", index=False)

print("\n========== DATA PREPARATION ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))

print("\nExcluded columns:", excluded_columns)
print("\nDATA PREPARATION COMPLETED!")
