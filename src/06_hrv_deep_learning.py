# -*- coding: utf-8 -*-
"""
Deep learning (1D-CNN and LSTM) on RAW IBI sequences for stress classification.
Works in Google Colab OR Kaggle Notebooks (both have TensorFlow pre-installed --
no pip install needed).

HOW TO USE IN KAGGLE:
  1. kaggle.com -> New Notebook
  2. Right-hand panel -> "Add Input" -> "Upload" -> upload HRV_sequences.npz
  3. Paste this whole script into a cell and run it (Shift+Enter)
     (the script below auto-finds the uploaded file, no path editing needed)
  4. Copy the printed output back to me

HOW TO USE IN COLAB:
  1. colab.research.google.com -> New Notebook
  2. Left sidebar -> Files icon -> drag & drop HRV_sequences.npz into it
  3. Paste this whole script into a cell and run it (Shift+Enter)
  4. Copy the printed output back to me

This compares a 1D-CNN and an LSTM against the classical Random Forest /
Logistic Regression results we already have, using the EXACT SAME windows,
labels, and leak-free participant-level split -- so results are directly
comparable.
"""
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report

tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------
# auto-find HRV_sequences.npz wherever it was uploaded (Kaggle input dir,
# Kaggle working dir, or Colab's /content) so no manual path editing is needed
_candidates = (
    glob.glob("/kaggle/input/**/HRV_sequences.npz", recursive=True)
    + glob.glob("/kaggle/working/HRV_sequences.npz")
    + glob.glob("/content/HRV_sequences.npz")
    + glob.glob("HRV_sequences.npz")
)
if not _candidates:
    raise FileNotFoundError(
        "Could not find HRV_sequences.npz. Make sure you uploaded it "
        "(Kaggle: Add Input -> Upload) and re-run this cell."
    )
_data_path = _candidates[0]
print(f"Loading data from: {_data_path}")
data = np.load(_data_path)
X = data["X"]            # (n_windows, 110) raw IBI values in ms, zero-padded
lengths = data["lengths"]
y = data["y"]
groups = data["groups"]

print(f"X shape: {X.shape}, positive rate: {y.mean():.3f}, participants: {len(set(groups))}")

# normalize (fit scaling stats on train only, done below per-split to avoid leakage)
X = X[..., np.newaxis]  # shape (n, 110, 1) for Conv1D/LSTM input

# ---------------------------------------------------------------------
# leak-free participant-level split (same discipline as the classical-ML part)
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# normalize using TRAIN statistics only (mask out zero-padding when computing stats)
train_vals = X_train[X_train != 0]
mu, sigma = train_vals.mean(), train_vals.std()
X_train_n = np.where(X_train != 0, (X_train - mu) / sigma, 0)
X_test_n = np.where(X_test != 0, (X_test - mu) / sigma, 0)

class_weight = {0: 1.0, 1: (y_train == 0).sum() / (y_train == 1).sum()}
print(f"Train: {len(train_idx)} windows, {len(set(groups[train_idx]))} participants")
print(f"Test:  {len(test_idx)} windows, {len(set(groups[test_idx]))} participants")
print(f"Class weight (to balance stress/non-stress): {class_weight}")


def make_cnn(input_len):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_len, 1)),
        keras.layers.Conv1D(32, 5, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 5, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def make_lstm(input_len):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_len, 1)),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


results = {}
for name, builder in [("1D-CNN", make_cnn), ("LSTM", make_lstm)]:
    print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")
    model = builder(X_train_n.shape[1])
    es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_auc", mode="max")
    model.fit(X_train_n, y_train, validation_split=0.2, epochs=100, batch_size=16,
              class_weight=class_weight, callbacks=[es], verbose=0)

    proba = model.predict(X_test_n, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    results[name] = {"auc": auc, "acc": acc, "f1": f1}
    print(f"{name}: AUC={auc:.3f}  Accuracy={acc:.3f}  F1={f1:.3f}")
    print(classification_report(y_test, pred, target_names=["non-stress", "stress"]))

print("\n" + "=" * 60)
print("COMPARISON: Deep Learning vs classical ML (same test participants)")
print("=" * 60)
print("Classical ML (already computed): Random Forest held-out AUC = 0.638")
for name, r in results.items():
    print(f"{name}: AUC = {r['auc']:.3f}")
