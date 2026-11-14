
import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# 1. PATHS
# =====================================================

MODEL_PATH = "backend/artifacts/models/xgboost.joblib"
DATA_PATH = "data/processed/model_ready/X_test.csv"

OUTPUT_DIR = "backend/artifacts/explainability"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nLoading trained XGBoost model...")

# =====================================================
# 2. LOAD MODEL AND TEST DATA
# =====================================================

model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(DATA_PATH)

print("Test dataset shape:", X_test.shape)

# =====================================================
# 3. GET PREPROCESSOR AND CLASSIFIER
# =====================================================

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Transform test data
X_transformed = preprocessor.transform(X_test)

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Convert sparse matrix to dense if needed
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed)

print("Transformed shape:", X_transformed.shape)
print("Number of features:", len(feature_names))

# =====================================================
# 4. LIMIT DATA FOR SHAP PERFORMANCE
# =====================================================

# Use a representative sample for explainability.
# This is not model training data.

sample_size = min(1000, X_transformed.shape[0])

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    X_transformed.shape[0],
    size=sample_size,
    replace=False
)

X_sample = X_transformed[sample_indices]

print("SHAP sample size:", X_sample.shape[0])

# =====================================================
# 5. CREATE SHAP EXPLAINER
# =====================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(X_sample)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_values = shap_values[-1]

elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = shap_values[:, :, -1]

print("SHAP calculation completed.")

# =====================================================
# 6. FEATURE IMPORTANCE
# =====================================================

importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": importance
})

importance_df = importance_df.sort_values(
    by="mean_absolute_shap",
    ascending=False
)

importance_path = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.csv"
)

importance_df.to_csv(importance_path, index=False)

print("\nTop 15 important features:")

print(importance_df.head(15).to_string(index=False))

# =====================================================
# 7. GLOBAL FEATURE IMPORTANCE PLOT
# =====================================================

plt.figure(figsize=(10, 7))

top_features = importance_df.head(15).sort_values(
    "mean_absolute_shap"
)

plt.barh(
    top_features["feature"],
    top_features["mean_absolute_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Features")
plt.title("Top 15 Feature Importance - RailSetu AI")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_feature_importance.png"),
    dpi=300
)

plt.close()

# =====================================================
# 8. SHAP SUMMARY PLOT
# =====================================================

print("\nGenerating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_summary_plot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 9. INDIVIDUAL PREDICTION EXPLANATION
# =====================================================

print("\nGenerating individual prediction explanation...")

sample_index = 0

individual_shap = shap_values[sample_index]

individual_df = pd.DataFrame({
    "feature": feature_names,
    "shap_value": individual_shap,
    "feature_value": X_sample[sample_index]
})

individual_df["absolute_shap"] = (
    individual_df["shap_value"].abs()
)

individual_df = individual_df.sort_values(
    "absolute_shap",
    ascending=False
)

individual_path = os.path.join(
    OUTPUT_DIR,
    "individual_prediction_explanation.csv"
)

individual_df.to_csv(individual_path, index=False)

# =====================================================
# 10. SAVE INDIVIDUAL WATERFALL PLOT
# =====================================================

explanation = explainer(X_sample[:1])

# Binary classifier: select the positive-class explanation
if explanation.values.ndim == 3:
    explanation = explanation[:, :, -1]

shap.plots.waterfall(
    explanation[0],
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "individual_waterfall.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 11. FINAL SUMMARY
# =====================================================

print("\n========================================")
print("SHAP EXPLAINABILITY COMPLETED")
print("========================================")

print("\nGenerated files:")

print("- shap_feature_importance.csv")
print("- shap_feature_importance.png")
print("- shap_summary_plot.png")
print("- individual_prediction_explanation.csv")
print("- individual_waterfall.png")

print("\nOutput directory:", OUTPUT_DIR)
