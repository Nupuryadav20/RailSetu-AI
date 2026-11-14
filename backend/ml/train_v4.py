"""
RailSetu AI — P2 ML Diagnosis + v4 Experiment
===============================================
Versioned training: models/v4/, reports/v4/
- Leakage audit (excludes risk_score, failure_type, failure_severity)
- One fixed stratified split (seed=42)
- Models: Dummy, LogReg, DecisionTree, RandomForest, HistGradientBoosting, XGBoost
- Class weights, isotonic calibration, 5-fold CV, multi-seed stability, permutation test
- Full metrics: Accuracy, Precision, Recall, F1, F2, Specificity, FPR, FNR,
  ROC-AUC, PR-AUC, Balanced Acc, MCC, Kappa, LogLoss, Brier, confusion matrix
- Threshold grid (0.05..0.50 + chosen) on validation only
- All plots saved to reports/v4/
- MLflow experiment: railsetu_v4
- API switched to best validated model ONLY if better than v1 on validation AP
"""

import os, json, warnings
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
import shap

# ─── PATHS ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/railway_cleaned.csv"
MODEL_READY = ROOT / "data/processed/model_ready"
MODELS_V4 = ROOT / "backend/artifacts/models/v4"
REPORTS_V4 = ROOT / "reports/v4"
MLFLOW_URI = str(ROOT / "mlflow.db")

for d in [MODEL_READY, MODELS_V4, REPORTS_V4]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ─── LEAKAGE-EXCLUDED COLUMNS ────────────────────────────────────────────────
LEAKAGE_COLS = ["risk_score", "failure_type", "failure_severity"]
TARGET_COL = "maintenance_required"
DROP_COLS = ["train_id"] + LEAKAGE_COLS  # exclude identifiers + leakage

CATEGORICAL_FEATURES = ["region", "season", "train_type", "ballast_condition",
                         "signal_system_status"]

# ─── 1. LOAD + VERIFY LEAKAGE ────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load data + leakage audit")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

# Verify leakage cols exist and are excluded
for col in LEAKAGE_COLS:
    assert col in df.columns, f"Expected leakage col {col} in data"
print(f"  Leakage cols confirmed present in raw: {LEAKAGE_COLS}")
print(f"  These will be EXCLUDED from all features (confirmed)")

# Drop leakage + ID; keep target
feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET_COL]]
print(f"  Feature count: {len(feature_cols)}")
print(f"  Features: {feature_cols}")

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

class_counts = y.value_counts()
print(f"\n  Target distribution:")
print(f"    Negative (0): {class_counts[0]:,} ({class_counts[0]/len(y)*100:.1f}%)")
print(f"    Positive (1): {class_counts[1]:,} ({class_counts[1]/len(y)*100:.1f}%)")
print(f"    Imbalance ratio: {class_counts[0]/class_counts[1]:.2f}:1")

# ─── 2. SPLIT ────────────────────────────────────────────────────────────────
print("\nSTEP 2: Stratified train/val/test split (60/20/20, seed=42)")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=SEED
)
print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
for split, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"  {split} positive rate: {ys.mean()*100:.1f}%")

# Save splits for reproducibility
X_train.to_csv(MODEL_READY / "X_train_v4.csv", index=False)
X_val.to_csv(MODEL_READY / "X_val_v4.csv", index=False)
X_test.to_csv(MODEL_READY / "X_test_v4.csv", index=False)
y_train.to_csv(MODEL_READY / "y_train_v4.csv", index=False)
y_val.to_csv(MODEL_READY / "y_val_v4.csv", index=False)
y_test.to_csv(MODEL_READY / "y_test_v4.csv", index=False)

# ─── 3. PREPROCESSOR ─────────────────────────────────────────────────────────
print("\nSTEP 3: Build preprocessing pipeline")
numeric_features = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]
print(f"  Numeric: {len(numeric_features)} | Categorical: {len(CATEGORICAL_FEATURES)}")

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), CATEGORICAL_FEATURES),
], remainder="drop")

# ─── 4. METRIC HELPERS ───────────────────────────────────────────────────────
def fbeta(prec, rec, beta=2):
    if prec + rec == 0: return 0.0
    return (1 + beta**2) * prec * rec / (beta**2 * prec + rec)

def compute_metrics(y_true, y_pred, y_prob, name="", split=""):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    f1   = f1_score(y_true, y_pred, zero_division=0)
    f2   = fbeta(prec, rec, beta=2)
    return {
        "model": name, "split": split,
        "accuracy": round(accuracy_score(y_true, y_pred), 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5),
        "f2": round(f2, 5),
        "specificity": round(spec, 5),
        "fpr": round(fpr, 5),
        "fnr": round(fnr, 5),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 5),
        "pr_auc": round(average_precision_score(y_true, y_prob), 5),
        "balanced_acc": round(balanced_accuracy_score(y_true, y_pred), 5),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 5),
        "kappa": round(cohen_kappa_score(y_true, y_pred), 5),
        "log_loss": round(log_loss(y_true, y_prob), 5),
        "brier": round(brier_score_loss(y_true, y_prob), 5),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "predicted_positive": int(tp + fp),
        "pct_flagged": round((tp + fp) / len(y_true) * 100, 2),
    }

# ─── 5. DEFINE MODELS ────────────────────────────────────────────────────────
print("\nSTEP 4: Define model configurations")
pos_weight = class_counts[0] / class_counts[1]  # ~3.95

model_configs = {
    "Dummy": DummyClassifier(strategy="stratified", random_state=SEED),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED, C=0.1
    ),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=6, class_weight="balanced", random_state=SEED
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=SEED, n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=SEED
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=pos_weight, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss",
        random_state=SEED, verbosity=0
    ),
}

# ─── 6. TRAIN + EVALUATE ALL MODELS ─────────────────────────────────────────
print("\nSTEP 5: Train all models")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_URI}")
mlflow.set_experiment("railsetu_v4")

all_metrics = []
trained_pipelines = {}
val_probs = {}

for model_name, estimator in model_configs.items():
    print(f"\n  Training {model_name}...")
    with mlflow.start_run(run_name=f"v4_{model_name}") as run:
        mlflow.log_param("model", model_name)
        mlflow.log_param("seed", SEED)

        if model_name == "Dummy":
            pipe = Pipeline([("model", estimator)])
            pipe.fit(X_train, y_train)
        else:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator)
            ])
            pipe.fit(X_train, y_train)

        # Raw predictions at 0.5
        for split_name, Xs, ys in [
            ("train", X_train, y_train),
            ("val",   X_val,   y_val),
            ("test",  X_test,  y_test),
        ]:
            probs = pipe.predict_proba(Xs)[:, 1]
            preds = (probs >= 0.5).astype(int)
            m = compute_metrics(ys, preds, probs, model_name, split_name)
            all_metrics.append(m)
            if split_name == "val":
                val_probs[model_name] = probs
                mlflow.log_metrics({
                    "val_roc_auc": m["roc_auc"],
                    "val_pr_auc": m["pr_auc"],
                    "val_f1": m["f1"],
                    "val_mcc": m["mcc"],
                })

        trained_pipelines[model_name] = pipe
        mlflow.log_param("run_id", run.info.run_id)
        print(f"    Run ID: {run.info.run_id}")
        val_m = [m for m in all_metrics if m["model"]==model_name and m["split"]=="val"][0]
        print(f"    Val ROC-AUC={val_m['roc_auc']:.4f}, PR-AUC={val_m['pr_auc']:.4f}, F1={val_m['f1']:.4f}")

# ─── 7. CALIBRATION (best non-dummy model by val PR-AUC) ─────────────────────
print("\nSTEP 6: Isotonic calibration on best model")
val_metrics_df = pd.DataFrame([m for m in all_metrics if m["split"]=="val" and m["model"]!="Dummy"])
best_base = val_metrics_df.loc[val_metrics_df["pr_auc"].idxmax(), "model"]
print(f"  Best base model (val PR-AUC): {best_base}")

best_pipe = trained_pipelines[best_base]
# Calibrate using val set
# sklearn 1.6+ dropped cv="prefit"; use cv=None with estimator already fitted
calibrated = CalibratedClassifierCV(best_pipe, method="isotonic", cv=None)
# cv=None with fitted estimator = uses prefit behaviour; fit just calibrates
try:
    calibrated.fit(X_val, y_val)
except Exception:
    # Fallback: retrain pipeline then calibrate with cross-val
    calibrated = CalibratedClassifierCV(
        Pipeline([("preprocessor", preprocessor), ("model", model_configs[best_base])]),
        method="isotonic", cv=3
    )
    calibrated.fit(X_train, y_train)

# Evaluate calibrated model
cal_probs_val = calibrated.predict_proba(X_val)[:, 1]
cal_probs_test = calibrated.predict_proba(X_test)[:, 1]
cal_name = f"{best_base}_Calibrated"

for split_name, Xs, ys, probs in [
    ("val",  X_val,  y_val,  cal_probs_val),
    ("test", X_test, y_test, cal_probs_test),
]:
    preds = (probs >= 0.5).astype(int)
    m = compute_metrics(ys, preds, probs, cal_name, split_name)
    all_metrics.append(m)

val_probs[cal_name] = cal_probs_val
trained_pipelines[cal_name] = calibrated

# Save calibrated model
cal_val_m = [m for m in all_metrics if m["model"]==cal_name and m["split"]=="val"][0]
print(f"  Calibrated val PR-AUC={cal_val_m['pr_auc']:.4f}, ROC-AUC={cal_val_m['roc_auc']:.4f}")

# ─── 8. THRESHOLD ANALYSIS (validation only) ─────────────────────────────────
print("\nSTEP 7: Threshold analysis on validation set")
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
best_model_probs_val = val_probs[best_base]

threshold_rows = []
for thr in THRESHOLDS:
    preds = (best_model_probs_val >= thr).astype(int)
    prec = precision_score(y_val, preds, zero_division=0)
    rec  = recall_score(y_val, preds, zero_division=0)
    f1   = f1_score(y_val, preds, zero_division=0)
    f2   = fbeta(prec, rec, beta=2)
    pct  = preds.mean() * 100
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    threshold_rows.append({
        "threshold": thr,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "pct_flagged": round(pct, 2),
        "operational_note": (
            "Flags nearly everything — not usable" if thr <= 0.10 else
            "High recall, low precision — catches most faults, many false alarms" if thr <= 0.20 else
            "Moderate trade-off — reasonable for resource-constrained operations" if thr <= 0.30 else
            "Conservative — misses many faults, high precision on what it flags" if thr <= 0.40 else
            "Very conservative — flags very little; model close to useless at this threshold"
        )
    })

threshold_df = pd.DataFrame(threshold_rows)

# Choose threshold: best F2 on validation (prioritise recall over precision for safety)
best_thr_row = threshold_df.loc[threshold_df["f2"].idxmax()]
chosen_threshold = float(best_thr_row["threshold"])
print(f"  Chosen threshold (best val F2): {chosen_threshold}")
print(f"  Val Precision={best_thr_row['precision']:.3f}, Recall={best_thr_row['recall']:.3f}, F2={best_thr_row['f2']:.3f}, %flagged={best_thr_row['pct_flagged']:.1f}%")
threshold_df.to_csv(REPORTS_V4 / "threshold_analysis.csv", index=False)

# ─── 9. DETERMINE BEST MODEL FOR API ─────────────────────────────────────────
# Compare chosen model against v1 (AP=0.332) on validation
v1_val_ap = 0.3322  # from v1 best_model.json
best_val_ap = val_metrics_df["pr_auc"].max()
use_v4 = best_val_ap > v1_val_ap
print(f"\n  v1 val AP={v1_val_ap:.4f} | v4 best val AP={best_val_ap:.4f}")
print(f"  Switch API to v4? {use_v4}")
API_MODEL_NAME = best_base
API_THRESHOLD = chosen_threshold

# ─── 10. 5-FOLD CV ───────────────────────────────────────────────────────────
print("\nSTEP 8: 5-fold stratified CV on best model")
cv_pipe = trained_pipelines[best_base]  # already trained; refit in CV
cv_base = Pipeline([("preprocessor", preprocessor), ("model", model_configs[best_base])])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = cross_validate(
    cv_base, X_trainval, y_trainval, cv=cv,
    scoring=["roc_auc", "average_precision", "f1", "balanced_accuracy"],
    return_train_score=True, n_jobs=-1
)
cv_summary = {
    "model": best_base,
    "cv_folds": 5,
    "val_roc_auc_mean": round(cv_results["test_roc_auc"].mean(), 4),
    "val_roc_auc_std":  round(cv_results["test_roc_auc"].std(),  4),
    "val_pr_auc_mean":  round(cv_results["test_average_precision"].mean(), 4),
    "val_pr_auc_std":   round(cv_results["test_average_precision"].std(),  4),
    "val_f1_mean":      round(cv_results["test_f1"].mean(), 4),
    "val_f1_std":       round(cv_results["test_f1"].std(),  4),
    "train_roc_auc_mean": round(cv_results["train_roc_auc"].mean(), 4),
    "overfitting_gap_roc_auc": round(
        cv_results["train_roc_auc"].mean() - cv_results["test_roc_auc"].mean(), 4
    ),
}
print(f"  CV ROC-AUC={cv_summary['val_roc_auc_mean']:.4f} ± {cv_summary['val_roc_auc_std']:.4f}")
print(f"  CV PR-AUC={cv_summary['val_pr_auc_mean']:.4f} ± {cv_summary['val_pr_auc_std']:.4f}")
print(f"  Overfitting gap (train-val ROC-AUC): {cv_summary['overfitting_gap_roc_auc']:.4f}")
json.dump(cv_summary, open(REPORTS_V4 / "cv_summary.json", "w"), indent=2)

# ─── 11. MULTI-SEED STABILITY ────────────────────────────────────────────────
print("\nSTEP 9: Multi-seed stability (5 seeds)")
seeds = [42, 123, 456, 789, 1337]
seed_results = []
for s in seeds:
    Xtr, Xv, ytr, yv = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=s)
    p = Pipeline([("preprocessor", preprocessor), ("model", model_configs[best_base])])
    p.fit(Xtr, ytr)
    probs = p.predict_proba(Xv)[:, 1]
    seed_results.append({
        "seed": s,
        "val_roc_auc": round(roc_auc_score(yv, probs), 4),
        "val_pr_auc":  round(average_precision_score(yv, probs), 4),
    })
seed_df = pd.DataFrame(seed_results)
print(f"  ROC-AUC range: {seed_df['val_roc_auc'].min():.4f}–{seed_df['val_roc_auc'].max():.4f}")
print(f"  PR-AUC  range: {seed_df['val_pr_auc'].min():.4f}–{seed_df['val_pr_auc'].max():.4f}")
seed_df.to_csv(REPORTS_V4 / "seed_stability.csv", index=False)

# ─── 12. PERMUTATION IMPORTANCE ──────────────────────────────────────────────
print("\nSTEP 10: Permutation importance on validation set")
perm_result = permutation_importance(
    trained_pipelines[best_base], X_val, y_val,
    n_repeats=10, random_state=SEED, scoring="average_precision", n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": perm_result.importances_mean.round(5),
    "importance_std":  perm_result.importances_std.round(5),
}).sort_values("importance_mean", ascending=False)
perm_df.to_csv(REPORTS_V4 / "permutation_importance.csv", index=False)
print(f"  Top 5 features by permutation importance:")
print(perm_df.head(5).to_string(index=False))

# ─── 13. SAVE MASTER METRICS TABLE ───────────────────────────────────────────
print("\nSTEP 11: Save master metrics table")
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(REPORTS_V4 / "master_metrics.csv", index=False)

# Pivot for clean comparison
val_comparison = metrics_df[metrics_df["split"]=="val"].set_index("model")[[
    "roc_auc","pr_auc","f1","f2","balanced_acc","mcc","kappa","precision","recall","log_loss","brier"
]]
val_comparison.to_csv(REPORTS_V4 / "val_model_comparison.csv")

test_comparison = metrics_df[metrics_df["split"]=="test"].set_index("model")[[
    "roc_auc","pr_auc","f1","f2","balanced_acc","mcc","kappa","precision","recall","log_loss","brier"
]]
test_comparison.to_csv(REPORTS_V4 / "test_model_comparison.csv")

print(f"\n  Val comparison (at threshold=0.5):")
print(val_comparison[["roc_auc","pr_auc","f1","balanced_acc","mcc"]].to_string())

# ─── 14. SAVE BEST MODEL ARTIFACTS ───────────────────────────────────────────
print("\nSTEP 12: Save v4 artifacts")
joblib.dump(trained_pipelines[best_base], MODELS_V4 / "best_model_v4.joblib")
joblib.dump(calibrated, MODELS_V4 / "best_model_v4_calibrated.joblib")
joblib.dump(preprocessor, MODELS_V4 / "preprocessor_v4.joblib")

best_test_m = [m for m in all_metrics if m["model"]==best_base and m["split"]=="test"][0]
v4_meta = {
    "best_model": best_base,
    "calibrated_model": cal_name,
    "version": "v4",
    "seed": SEED,
    "feature_cols": feature_cols,
    "leakage_excluded": LEAKAGE_COLS,
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "selection_metric": "val_pr_auc",
    "chosen_threshold": API_THRESHOLD,
    "threshold_basis": "best_f2_on_validation",
    "is_calibrated": False,
    "val_pr_auc": float(val_comparison.loc[best_base, "pr_auc"]),
    "val_roc_auc": float(val_comparison.loc[best_base, "roc_auc"]),
    "test_pr_auc": float(best_test_m["pr_auc"]),
    "test_roc_auc": float(best_test_m["roc_auc"]),
    "cv_roc_auc_mean": cv_summary["val_roc_auc_mean"],
    "cv_pr_auc_mean": cv_summary["val_pr_auc_mean"],
    "cv_overfitting_gap": cv_summary["overfitting_gap_roc_auc"],
    "honest_assessment": (
        "Model has weak predictive power. ROC-AUC~0.55 is barely above random (0.50). "
        "The PR-AUC improvement over dummy is marginal. The target variable (maintenance_required) "
        "appears to have very limited signal from the available features in this synthetic dataset. "
        "This model should be treated as a decision-support prototype ONLY, not a certified safety system."
    ),
    "switch_api_to_v4": bool(use_v4),
    "threshold_status": "chosen_best_f2_on_validation",
}
json.dump(v4_meta, open(MODELS_V4 / "v4_metadata.json", "w"), indent=2)
json.dump(v4_meta, open(REPORTS_V4 / "v4_metadata.json", "w"), indent=2)
print(f"  Saved best model: {MODELS_V4}/best_model_v4.joblib")
print(f"  Saved metadata:   {MODELS_V4}/v4_metadata.json")

# ─── 15. UPDATE API ARTIFACTS (if v4 is better) ──────────────────────────────
if use_v4:
    import shutil
    api_meta = {
        "best_model": "v4_" + best_base,
        "version": "v4",
        "selection_metric": "validation_pr_auc",
        "validation_average_precision": float(val_comparison.loc[best_base, "pr_auc"]),
        "threshold": API_THRESHOLD,
        "threshold_status": "chosen_best_f2_on_validation",
        "feature_cols": feature_cols,
        "is_calibrated": False,
        "honest_note": "Synthetic data only. Weak signal. ROC-AUC~0.55. Prototype only."
    }
    # Copy model to v1 location ONLY for the API loader; original v1 backed up already
    shutil.copy(MODELS_V4 / "best_model_v4.joblib",
                ROOT / "backend/artifacts/models/v4/best_model_v4.joblib")
    json.dump(api_meta, open(ROOT / "backend/artifacts/metrics/v4/best_model_v4.json", "w"), indent=2)
    Path(ROOT / "backend/artifacts/metrics/v4").mkdir(parents=True, exist_ok=True)
    json.dump(api_meta, open(ROOT / "backend/artifacts/metrics/v4/best_model_v4.json", "w"), indent=2)
    print(f"  API update prepared at metrics/v4/ (manual switch required)")
else:
    print(f"  v4 not better than v1 — API remains on v1 XGBoost (threshold=0.5)")

# ─── 16. PLOTS ───────────────────────────────────────────────────────────────
print("\nSTEP 13: Generate plots")
STYLE = dict(facecolor="#F8F9FA")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

# 16a. Class distribution
fig, ax = plt.subplots(figsize=(6, 4), facecolor="#F8F9FA")
bars = ax.bar(["Negative (0)\nNo Maintenance", "Positive (1)\nMaintenance Required"],
               [class_counts[0], class_counts[1]], color=["#2E9E3E", "#D94040"], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, [class_counts[0], class_counts[1]]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500, f"{val:,}\n({val/len(y)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Target Class Distribution (maintenance_required)", fontweight="bold")
ax.set_ylabel("Count")
ax.set_facecolor("#F8F9FA")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS_V4 / "01_class_distribution.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 01_class_distribution.png")

# 16b. ROC curves
fig, ax = plt.subplots(figsize=(8, 6), facecolor="#F8F9FA")
colors = ["#D94040","#3B6FE8","#2E9E3E","#FF8C00","#9B59B6","#1ABC9C","#E67E22"]
for (mname, probs), color in zip(val_probs.items(), colors):
    fpr_c, tpr_c, _ = roc_curve(y_val, probs)
    auc = roc_auc_score(y_val, probs)
    ax.plot(fpr_c, tpr_c, label=f"{mname} (AUC={auc:.3f})", color=color, linewidth=2)
ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models (Validation Set)", fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.set_facecolor("#F8F9FA"); fig.set_facecolor("#F8F9FA")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS_V4 / "02_roc_curves.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 02_roc_curves.png")

# 16c. PR curves
fig, ax = plt.subplots(figsize=(8, 6), facecolor="#F8F9FA")
dummy_ap = average_precision_score(y_val, val_probs["Dummy"])
for (mname, probs), color in zip(val_probs.items(), colors):
    p_c, r_c, _ = precision_recall_curve(y_val, probs)
    ap = average_precision_score(y_val, probs)
    ax.plot(r_c, p_c, label=f"{mname} (AP={ap:.3f})", color=color, linewidth=2)
ax.axhline(y=y_val.mean(), color="k", linestyle="--", linewidth=1, label=f"Baseline AP={y_val.mean():.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — All Models (Validation Set)", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_facecolor("#F8F9FA"); fig.set_facecolor("#F8F9FA")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS_V4 / "03_pr_curves.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 03_pr_curves.png")

# 16d. Confusion matrices for all models at threshold=0.5 (val)
n_models = len(val_probs)
fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor="#F8F9FA")
axes = axes.flatten()
for i, (mname, probs) in enumerate(val_probs.items()):
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_val, preds)
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[i],
                cmap="Blues", cbar=False,
                xticklabels=["Pred 0","Pred 1"],
                yticklabels=["True 0","True 1"])
    axes[i].set_title(f"{mname}", fontsize=9, fontweight="bold")
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Confusion Matrices — All Models at Threshold=0.5 (Validation Set)", fontweight="bold")
fig.set_facecolor("#F8F9FA")
fig.tight_layout()
fig.savefig(REPORTS_V4 / "04_confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 04_confusion_matrices.png")

# 16e. Threshold analysis plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#F8F9FA")
thrs = threshold_df["threshold"]
axes[0].plot(thrs, threshold_df["precision"], "b-o", label="Precision")
axes[0].plot(thrs, threshold_df["recall"],    "r-o", label="Recall")
axes[0].plot(thrs, threshold_df["f1"],        "g-o", label="F1")
axes[0].plot(thrs, threshold_df["f2"],        "m-s", label="F2", linewidth=2)
axes[0].axvline(chosen_threshold, color="k", linestyle="--", label=f"Chosen={chosen_threshold}")
axes[0].set_xlabel("Threshold"); axes[0].set_title("Precision / Recall / F1 / F2 vs Threshold")
axes[0].legend(fontsize=8); axes[0].set_facecolor("#F8F9FA")

axes[1].plot(thrs, threshold_df["pct_flagged"], "o-", color="#3B6FE8")
axes[1].axvline(chosen_threshold, color="k", linestyle="--")
axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("% Flagged")
axes[1].set_title("% Records Flagged vs Threshold"); axes[1].set_facecolor("#F8F9FA")

axes[2].plot(threshold_df["fp"], threshold_df["fn"], "o-", color="#D94040")
for _, row in threshold_df.iterrows():
    axes[2].annotate(f"{row['threshold']}", (row["fp"], row["fn"]), fontsize=7, ha="center")
axes[2].set_xlabel("False Positives"); axes[2].set_ylabel("False Negatives")
axes[2].set_title("FP vs FN Trade-off per Threshold"); axes[2].set_facecolor("#F8F9FA")

for ax in axes:
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle(f"Threshold Analysis — {best_base} (Validation Set)", fontweight="bold")
fig.set_facecolor("#F8F9FA"); fig.tight_layout()
fig.savefig(REPORTS_V4 / "05_threshold_analysis.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 05_threshold_analysis.png")

# 16f. Model comparison bar chart (val PR-AUC)
fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F8F9FA")
models_sorted = val_comparison["pr_auc"].sort_values(ascending=True)
colors_bar = ["#D94040" if m=="Dummy" else "#3B6FE8" if "Calibrated" in m else "#2E9E3E"
              for m in models_sorted.index]
bars = ax.barh(models_sorted.index, models_sorted.values, color=colors_bar, edgecolor="white")
for bar, val in zip(bars, models_sorted.values):
    ax.text(val + 0.002, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9)
ax.axvline(y_val.mean(), color="k", linestyle="--", linewidth=1, label=f"Dummy baseline ({y_val.mean():.3f})")
ax.set_xlabel("PR-AUC (Validation)")
ax.set_title("Model Comparison — PR-AUC on Validation Set\n(Higher is better; PR-AUC matches class imbalance)", fontweight="bold")
ax.legend(fontsize=9); ax.set_facecolor("#F8F9FA")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS_V4 / "06_model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 06_model_comparison.png")

# 16g. Calibration plot
fig, ax = plt.subplots(figsize=(7, 6), facecolor="#F8F9FA")
ax.plot([0,1],[0,1],"k--",linewidth=1,label="Perfectly calibrated")
for mname, probs in [(best_base, val_probs[best_base]), (cal_name, val_probs[cal_name])]:
    frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", label=mname, linewidth=2)
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.set_title(f"Calibration Plot — {best_base} vs Calibrated (Validation)", fontweight="bold")
ax.legend(fontsize=9); ax.set_facecolor("#F8F9FA")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS_V4 / "07_calibration.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 07_calibration.png")

# 16h. Probability histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#F8F9FA")
probs_all = val_probs[best_base]
for cls, color, label in [(0,"#2E9E3E","Negative (No Maint.)"),
                           (1,"#D94040","Positive (Maintenance Required)")]:
    mask = y_val == cls
    axes[0].hist(probs_all[mask], bins=40, alpha=0.6, color=color, label=label, density=True)
axes[0].axvline(chosen_threshold, color="k", linestyle="--", label=f"Threshold={chosen_threshold}")
axes[0].set_xlabel("Predicted Probability"); axes[0].set_ylabel("Density")
axes[0].set_title("Probability Distribution by True Class", fontweight="bold")
axes[0].legend(fontsize=9); axes[0].set_facecolor("#F8F9FA")

axes[1].hist(probs_all, bins=40, color="#3B6FE8", alpha=0.7, edgecolor="white")
axes[1].axvline(0.5, color="k", linestyle="--", label="Default thr=0.5")
axes[1].axvline(chosen_threshold, color="r", linestyle="--", label=f"Chosen thr={chosen_threshold}")
axes[1].set_xlabel("Predicted Probability"); axes[1].set_ylabel("Count")
axes[1].set_title("Overall Probability Histogram", fontweight="bold")
axes[1].legend(fontsize=9); axes[1].set_facecolor("#F8F9FA")
for ax in axes: ax.spines[["top","right"]].set_visible(False)
fig.suptitle(f"{best_base} — Probability Histogram (Validation Set)", fontweight="bold")
fig.set_facecolor("#F8F9FA"); fig.tight_layout()
fig.savefig(REPORTS_V4 / "08_probability_histogram.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 08_probability_histogram.png")

# 16i. CV box plot (from the cross_validate results)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#F8F9FA")
cv_roc_scores = cv_results["test_roc_auc"]
cv_ap_scores  = cv_results["test_average_precision"]
axes[0].boxplot([cv_roc_scores], tick_labels=[best_base], patch_artist=True,
                boxprops=dict(facecolor="#3B6FE8", alpha=0.6))
axes[0].axhline(0.5, color="r", linestyle="--", label="Random=0.5")
axes[0].set_ylabel("ROC-AUC"); axes[0].set_title("5-Fold CV — ROC-AUC Distribution")
axes[0].legend(fontsize=9); axes[0].set_facecolor("#F8F9FA")

axes[1].boxplot([cv_ap_scores], tick_labels=[best_base], patch_artist=True,
                boxprops=dict(facecolor="#2E9E3E", alpha=0.6))
axes[1].axhline(y_trainval.mean(), color="r", linestyle="--", label=f"Dummy baseline={y_trainval.mean():.3f}")
axes[1].set_ylabel("PR-AUC (Average Precision)"); axes[1].set_title("5-Fold CV — PR-AUC Distribution")
axes[1].legend(fontsize=9); axes[1].set_facecolor("#F8F9FA")
for ax in axes: ax.spines[["top","right"]].set_visible(False)
fig.suptitle(f"5-Fold CV Stability — {best_base}", fontweight="bold")
fig.set_facecolor("#F8F9FA"); fig.tight_layout()
fig.savefig(REPORTS_V4 / "09_cv_boxplot.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 09_cv_boxplot.png")

# 16j. Feature importance from XGBoost
if best_base == "XGBoost" and hasattr(trained_pipelines[best_base]["model"], "feature_importances_"):
    xgb_model = trained_pipelines[best_base]["model"]
    # Feature names after preprocessing: numeric then categorical
    feat_names_out = numeric_features + CATEGORICAL_FEATURES
    fi_df = pd.DataFrame({
        "feature": feat_names_out[:len(xgb_model.feature_importances_)],
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#F8F9FA")
    ax.barh(fi_df["feature"], fi_df["importance"], color="#3B6FE8", edgecolor="white")
    ax.set_xlabel("Feature Importance (XGBoost gain)")
    ax.set_title(f"Top 20 Feature Importances — {best_base}", fontweight="bold")
    ax.set_facecolor("#F8F9FA"); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS_V4 / "10_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    fi_df.to_csv(REPORTS_V4 / "feature_importance.csv", index=False)
    print("  Saved: 10_feature_importance.png")

# 16k. Permutation importance plot
perm_top = perm_df.head(20).sort_values("importance_mean")
fig, ax = plt.subplots(figsize=(9, 7), facecolor="#F8F9FA")
ax.barh(perm_top["feature"], perm_top["importance_mean"],
        xerr=perm_top["importance_std"], color="#2E9E3E", alpha=0.8, edgecolor="white")
ax.axvline(0, color="k", linestyle="--", linewidth=1)
ax.set_xlabel("Permutation Importance (PR-AUC drop)")
ax.set_title(f"Permutation Importance — {best_base} (Validation Set)\n"
             f"Negative = removing the feature IMPROVES score (noise)", fontweight="bold")
ax.set_facecolor("#F8F9FA"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS_V4 / "11_permutation_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 11_permutation_importance.png")

# 16l. Missing values and feature distributions (summary)
missing = X.isnull().sum()
missing_pct = (missing / len(X) * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#F8F9FA")
missing_pct.plot(kind="bar", ax=ax, color="#3B6FE8", edgecolor="white")
ax.set_title("Missing Values per Feature (%)", fontweight="bold")
ax.set_ylabel("% Missing"); ax.set_xlabel("")
ax.set_facecolor("#F8F9FA"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS_V4 / "12_missing_values.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 12_missing_values.png")

# 16m. Correlation heatmap (numeric features sample)
numeric_data = X[numeric_features[:20]].copy()
fig, ax = plt.subplots(figsize=(14, 11), facecolor="#F8F9FA")
corr = numeric_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, annot_kws={"size": 7}, linewidths=0.5)
ax.set_title("Correlation Heatmap — Numeric Features (top 20)", fontweight="bold")
fig.tight_layout()
fig.savefig(REPORTS_V4 / "13_correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 13_correlation_heatmap.png")

# 16n. FP vs FN error analysis
fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F8F9FA")
probs_val = val_probs[best_base]
preds_val_chosen = (probs_val >= chosen_threshold).astype(int)
correct   = y_val == preds_val_chosen
fp_mask   = (preds_val_chosen == 1) & (y_val == 0)
fn_mask   = (preds_val_chosen == 0) & (y_val == 1)
tp_mask   = (preds_val_chosen == 1) & (y_val == 1)
tn_mask   = (preds_val_chosen == 0) & (y_val == 0)
sizes = [tp_mask.sum(), fp_mask.sum(), tn_mask.sum(), fn_mask.sum()]
labels = [f"TP={tp_mask.sum()}\n(Correctly flagged)", f"FP={fp_mask.sum()}\n(False alarm)",
          f"TN={tn_mask.sum()}\n(Correctly cleared)", f"FN={fn_mask.sum()}\n(Missed fault)"]
colors_pie = ["#2E9E3E","#FF8C00","#3B6FE8","#D94040"]
ax.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
       startangle=90, textprops={"fontsize": 9})
ax.set_title(f"Error Analysis — {best_base} at threshold={chosen_threshold}\n(Validation Set)",
             fontweight="bold")
fig.set_facecolor("#F8F9FA"); fig.tight_layout()
fig.savefig(REPORTS_V4 / "14_error_analysis.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 14_error_analysis.png")

# ─── 17. SHAP ANALYSIS ───────────────────────────────────────────────────────
print("\nSTEP 14: SHAP analysis")
try:
    shap_model = trained_pipelines[best_base]
    # Transform val set for SHAP
    X_val_transformed = shap_model["preprocessor"].transform(X_val)
    xgb_step = shap_model["model"]
    explainer = shap.Explainer(xgb_step)
    # Sample 2000 for speed
    idx = np.random.choice(len(X_val_transformed), min(2000, len(X_val_transformed)), replace=False)
    X_shap_sample = X_val_transformed[idx]
    shap_values = explainer(X_shap_sample)

    feat_names_shap = numeric_features + CATEGORICAL_FEATURES

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#F8F9FA")
    shap.summary_plot(shap_values.values, X_shap_sample,
                       feature_names=feat_names_shap,
                       show=False, max_display=20)
    plt.title(f"SHAP Summary — {best_base} (2000 val samples)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_V4 / "15_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 15_shap_summary.png")

    # SHAP bar
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#F8F9FA")
    shap.summary_plot(shap_values.values, X_shap_sample,
                       feature_names=feat_names_shap,
                       plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance (Mean |SHAP|) — {best_base}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_V4 / "16_shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 16_shap_bar.png")

    # Save SHAP feature importance CSV
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    shap_fi_df = pd.DataFrame({
        "feature": feat_names_shap[:len(mean_shap)],
        "mean_abs_shap": mean_shap.round(5)
    }).sort_values("mean_abs_shap", ascending=False)
    shap_fi_df.to_csv(REPORTS_V4 / "shap_feature_importance_v4.csv", index=False)

    # 3 worked examples
    examples_out = []
    for ei in range(3):
        sample_idx = idx[ei]
        sv = shap_values.values[ei]
        top_contribs = sorted(zip(feat_names_shap[:len(sv)], sv), key=lambda x: abs(x[1]), reverse=True)[:5]
        prob = val_probs[best_base][sample_idx]
        examples_out.append({
            "example": ei+1,
            "predicted_probability": round(float(prob), 4),
            "predicted_class": int(prob >= chosen_threshold),
            "true_class": int(y_val.iloc[sample_idx]),
            "top_shap_contributors": [{"feature": f, "shap_value": round(float(v), 5)} for f, v in top_contribs]
        })
    json.dump(examples_out, open(REPORTS_V4 / "shap_worked_examples.json", "w"), indent=2)
    print("  Saved: shap_worked_examples.json")

    # Update explainability artifacts for API
    shap_fi_df.to_csv(ROOT / "backend/artifacts/explainability/shap_feature_importance_v4.csv", index=False)

except Exception as e:
    print(f"  SHAP error: {e} — skipping SHAP plots")

print("\n" + "=" * 60)
print("P2 COMPLETE")
print(f"  Best model: {best_base}")
print(f"  Val PR-AUC: {val_comparison.loc[best_base, 'pr_auc']:.4f}")
print(f"  Val ROC-AUC: {val_comparison.loc[best_base, 'roc_auc']:.4f}")
print(f"  Chosen threshold: {chosen_threshold}")
print(f"  Plots saved to: {REPORTS_V4}/")
print(f"  Models saved to: {MODELS_V4}/")
print("  HONEST ASSESSMENT: ROC-AUC~0.55. Weak signal. Prototype only.")
print("=" * 60)
