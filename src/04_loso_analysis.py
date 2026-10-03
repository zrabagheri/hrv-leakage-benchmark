# -*- coding: utf-8 -*-
"""
Leave-One-Subject-Out (LOSO) cross-validation -- the gold-standard,
subject-independent evaluation protocol for wearable/physiological ML.
Every one of the 34 participants is held out exactly once as the test set.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import json

df = pd.read_csv("./data/HRV_stress_dataset.csv")
feature_cols = ["mean_hr", "sdnn", "rmssd", "pnn50", "sdsd", "cvnn",
                 "sd1", "sd2", "sd1_sd2_ratio",
                 "lf_power", "hf_power", "lf_hf_ratio", "total_power"]
X = df[feature_cols].values
y = df["label"].values
groups = df["participant"].values

logo = LeaveOneGroupOut()

models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                                               class_weight="balanced", random_state=42))]),
}

results = {}
for name, pipe in models.items():
    fold_aucs = []
    fold_records = []
    for train_idx, test_idx in logo.split(X, y, groups):
        y_test = y[test_idx]
        if len(set(y_test)) < 2:
            continue  # can't compute AUC for a subject with only one class
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y_test, proba)
        fold_aucs.append(auc)
        fold_records.append({"participant": int(groups[test_idx][0]), "auc": float(auc),
                              "n_windows": int(len(test_idx))})
    results[name] = {
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "median_auc": float(np.median(fold_aucs)),
        "n_subjects_evaluated": len(fold_aucs),
        "per_subject": fold_records,
    }
    print(f"{name}: LOSO AUC = {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f} "
          f"(median {np.median(fold_aucs):.3f}, n={len(fold_aucs)} subjects)")

with open("./results/hrv_loso_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved hrv_loso_results.json")
