# -*- coding: utf-8 -*-
"""
Final, decisive analysis for the HRV stress-classification paper.

Core contribution: quantify how much published-style "naive" (row-level,
non-grouped) evaluation OVERESTIMATES real-world performance compared to a
leak-free, participant-level evaluation. This is the paper's main
methodological hook.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold, KFold,
                                      cross_val_score, StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
import json

RANDOM_STATE = 42
df = pd.read_csv("./data/HRV_stress_dataset.csv")

feature_cols = ["mean_hr", "sdnn", "rmssd", "pnn50", "sdsd", "cvnn",
                 "sd1", "sd2", "sd1_sd2_ratio",
                 "lf_power", "hf_power", "lf_hf_ratio", "total_power"]

X = df[feature_cols].values
y = df["label"].values
groups = df["participant"].values

results = {}

# =====================================================================
# EXPERIMENT 1: "naive" row-level random split (common in published work)
#   -- windows from the SAME participant can appear in both train and test
# =====================================================================
print("=" * 70)
print("EXPERIMENT 1: NAIVE row-level split (participant leakage allowed)")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

naive_models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                        class_weight="balanced", random_state=RANDOM_STATE)),
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                            random_state=RANDOM_STATE)),
    ]),
}

results["naive"] = {}
for name, pipe in naive_models.items():
    aucs = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
    results["naive"][name] = {"mean_auc": float(aucs.mean()), "std_auc": float(aucs.std())}
    print(f"{name}: naive 5-fold row-level CV AUC = {aucs.mean():.3f} +/- {aucs.std():.3f}")

# =====================================================================
# EXPERIMENT 2: leak-free participant-level split (the honest approach)
# =====================================================================
print()
print("=" * 70)
print("EXPERIMENT 2: LEAK-FREE participant-level split (no participant")
print("appears in both train and test)")
print("=" * 70)

gkf = GroupKFold(n_splits=5)
leakfree_models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                        class_weight="balanced", random_state=RANDOM_STATE)),
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                            random_state=RANDOM_STATE)),
    ]),
}

results["leak_free"] = {}
for name, pipe in leakfree_models.items():
    aucs = cross_val_score(pipe, X, y, cv=gkf, groups=groups, scoring="roc_auc")
    results["leak_free"][name] = {"mean_auc": float(aucs.mean()), "std_auc": float(aucs.std())}
    print(f"{name}: leak-free participant-level CV AUC = {aucs.mean():.3f} +/- {aucs.std():.3f}")

# =====================================================================
# SUMMARY TABLE: the inflation gap (the paper's headline finding)
# =====================================================================
print()
print("=" * 70)
print("SUMMARY: AUC inflation from ignoring participant-level leakage")
print("=" * 70)
print(f"{'Model':<22}{'Naive AUC':>12}{'Leak-free AUC':>16}{'Inflation':>12}")
for name in naive_models:
    naive_auc = results["naive"][name]["mean_auc"]
    lf_auc = results["leak_free"][name]["mean_auc"]
    print(f"{name:<22}{naive_auc:>12.3f}{lf_auc:>16.3f}{naive_auc - lf_auc:>+12.3f}")

# =====================================================================
# FINAL MODEL: fit on a proper held-out participant split, report full
# metrics + permutation feature importance (model-agnostic, no SHAP needed)
# =====================================================================
print()
print("=" * 70)
print("FINAL HELD-OUT EVALUATION (participant-level, leak-free)")
print("=" * 70)

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

final_model = Pipeline([
    ("clf", RandomForestClassifier(n_estimators=500, max_depth=8,
                                    class_weight="balanced", random_state=RANDOM_STATE)),
])
final_model.fit(X_train, y_train)
proba = final_model.predict_proba(X_test)[:, 1]
pred = final_model.predict(X_test)

auc = roc_auc_score(y_test, proba)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)
print(f"Final Random Forest -- AUC={auc:.3f}  Accuracy={acc:.3f}  F1={f1:.3f}")
print(classification_report(y_test, pred, target_names=["non-stress", "stress"]))

perm = permutation_importance(final_model, X_test, y_test, n_repeats=30,
                               random_state=RANDOM_STATE, scoring="roc_auc")
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
print("\nPermutation feature importance (drop in AUC when feature is shuffled):")
print(importance_df.to_string(index=False))

results["final_heldout"] = {"auc": float(auc), "accuracy": float(acc), "f1": float(f1)}
results["permutation_importance"] = importance_df.to_dict(orient="records")

with open("./results/hrv_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

importance_df.to_csv("./results/hrv_feature_importance.csv", index=False)

print("\nSaved: hrv_analysis_results.json, hrv_feature_importance.csv")
