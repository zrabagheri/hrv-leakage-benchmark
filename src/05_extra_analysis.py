# -*- coding: utf-8 -*-
"""
Additional analyses for manuscript expansion:
1. ROC curve + confusion matrix for the final held-out Random Forest model.
2. Feature-group ablation (time-domain vs frequency-domain vs non-linear vs all),
   leak-free GroupKFold CV.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import json

RANDOM_STATE = 42
df = pd.read_csv("./data/HRV_stress_dataset.csv")

feature_groups = {
    "time_domain": ["mean_hr", "sdnn", "rmssd", "pnn50", "sdsd", "cvnn"],
    "nonlinear": ["sd1", "sd2", "sd1_sd2_ratio"],
    "frequency_domain": ["lf_power", "hf_power", "lf_hf_ratio", "total_power"],
}
all_features = sum(feature_groups.values(), [])

y = df["label"].values
groups = df["participant"].values

# ---------------------------------------------------------------------
# 1) Feature-group ablation: leak-free 5-fold GroupKFold CV, Random Forest
# ---------------------------------------------------------------------
gkf = GroupKFold(n_splits=5)
ablation_results = {}

combos = {
    "Time-domain only": feature_groups["time_domain"],
    "Frequency-domain only": feature_groups["frequency_domain"],
    "Non-linear only": feature_groups["nonlinear"],
    "Time + Frequency": feature_groups["time_domain"] + feature_groups["frequency_domain"],
    "All 13 features": all_features,
}

print("=" * 70)
print("FEATURE-GROUP ABLATION (leak-free GroupKFold CV, Random Forest)")
print("=" * 70)
for name, cols in combos.items():
    X = df[cols].values
    pipe = Pipeline([("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                                      class_weight="balanced", random_state=RANDOM_STATE))])
    aucs = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring="roc_auc")
    ablation_results[name] = {"n_features": len(cols), "mean_auc": float(aucs.mean()), "std_auc": float(aucs.std())}
    print(f"{name:<25} (n={len(cols):>2} features): AUC = {aucs.mean():.3f} +/- {aucs.std():.3f}")

with open("./results/hrv_ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)

# ---------------------------------------------------------------------
# 2) ROC curve + confusion matrix for final held-out Random Forest model
#    (identical split/model as final_analysis.py)
# ---------------------------------------------------------------------
X_all = df[all_features].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_all, y, groups))
X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

final_model = Pipeline([("clf", RandomForestClassifier(n_estimators=500, max_depth=8,
                                                         class_weight="balanced", random_state=RANDOM_STATE))])
final_model.fit(X_train, y_train)
proba = final_model.predict_proba(X_test)[:, 1]
pred = final_model.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, proba)
auc = roc_auc_score(y_test, proba)
cm = confusion_matrix(y_test, pred)

print(f"\nFinal RF held-out AUC (for ROC plot): {auc:.3f}")
print(f"Confusion matrix:\n{cm}")

# --- ROC curve plot ---
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Random Forest (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance (AUC = 0.500)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Final Held-Out Random Forest Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("./results/roc_curve.png", dpi=200)
plt.close()

# --- Confusion matrix plot ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non-stress", "stress"])
fig, ax = plt.subplots(figsize=(5.5, 5.5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix\nFinal Held-Out Random Forest Model", fontsize=12)
plt.tight_layout()
plt.savefig("./results/confusion_matrix.png", dpi=200)
plt.close()

print("\nSaved: hrv_ablation_results.json, roc_curve.png, confusion_matrix.png")
