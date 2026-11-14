
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
df = pd.read_csv("data/processed/railway_cleaned.csv")

print("\n========== RAILSETU AI: EDA ==========")

# 1. Basic information
print("\nDataset Shape:", df.shape)

print("\nData Types:")
print(df.dtypes)

# 2. Target distribution
print("\nMaintenance Distribution:")
print(df["maintenance_required"].value_counts())

print("\nMaintenance Percentage:")
print(
    df["maintenance_required"]
    .value_counts(normalize=True) * 100
)

# 3. Missing values percentage
print("\nMissing Value Percentage:")
missing = df.isnull().mean() * 100
print(missing[missing > 0].sort_values(ascending=False))

# 4. Maintenance by region
print("\nMaintenance by Region:")
print(
    df.groupby("region")["maintenance_required"]
    .agg(["count", "mean"])
    .sort_values("mean", ascending=False)
)

# 5. Maintenance by train type
print("\nMaintenance by Train Type:")
print(
    df.groupby("train_type")["maintenance_required"]
    .mean()
    .sort_values(ascending=False)
)

# 6. Numeric features vs target
numeric_cols = df.select_dtypes(include="number").columns

print("\nNumeric Features by Maintenance Status:")
print(
    df.groupby("maintenance_required")[numeric_cols]
    .mean()
    .T
)

# 7. Correlation analysis
corr = df[numeric_cols].corr()

print("\nCorrelation with Maintenance Target:")
print(
    corr["maintenance_required"]
    .sort_values(ascending=False)
)

# 8. Create plots
output_dir = Path("data/processed/eda")
output_dir.mkdir(parents=True, exist_ok=True)

# Target distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="maintenance_required")
plt.title("Maintenance Requirement Distribution")
plt.savefig(output_dir / "target_distribution.png")
plt.close()

# Maintenance by region
plt.figure(figsize=(10, 5))
sns.barplot(
    data=df,
    x="region",
    y="maintenance_required",
    errorbar=None
)
plt.title("Maintenance Rate by Railway Region")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(output_dir / "maintenance_by_region.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png")
plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Charts saved inside:", output_dir)
