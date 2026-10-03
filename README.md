# Quantifying Participant-Level Data Leakage in Wearable HRV Models

Code and processed data for the paper **"Quantifying Participant-Level Data Leakage in Wearable Heart Rate Variability Models: A Leak-Free Benchmark for Stress Classification"** (Zahra Bagheri, submitted to *Sensors*, MDPI).

## Overview

This repository quantifies how much naive, row-level cross-validation **overestimates** the performance of wearable-HRV stress-classification models compared to a leak-free, participant-level evaluation, using the publicly available [Stress-Predict dataset](https://github.com/italha-d/Stress-Predict-Dataset) (35 participants, Empatica E4 wristband).

It includes:
- Leak-free HRV feature extraction (13 time-domain, frequency-domain, and non-linear features over 60-second windows)
- Naive vs. leak-free 5-fold cross-validation comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Leave-One-Subject-Out (LOSO) benchmark across 34 participants
- Deep learning comparison (1D-CNN and LSTM on raw inter-beat-interval sequences)
- Feature-group ablation (time-domain vs. frequency-domain vs. non-linear)
- Final held-out model ROC curve and confusion matrix

## Repository structure

```
.
├── src/
│   ├── 01_build_hrv_dataset.py     # Extracts the 13 HRV features from raw IBI data → data/HRV_stress_dataset.csv
│   ├── 02_build_sequences.py       # Builds raw IBI sequences for deep learning → data/HRV_sequences.npz
│   ├── 03_final_analysis.py        # Naive vs. leak-free CV comparison + permutation feature importance
│   ├── 04_loso_analysis.py         # Leave-One-Subject-Out benchmark
│   ├── 05_extra_analysis.py        # Feature-group ablation + ROC curve + confusion matrix (final RF model)
│   └── 06_hrv_deep_learning.py     # 1D-CNN and LSTM on raw IBI sequences (Kaggle/Colab compatible)
├── data/
│   ├── HRV_stress_dataset.csv      # Processed, window-level HRV feature dataset (output of script 01)
│   └── HRV_sequences.npz           # Raw IBI sequences for the deep-learning models (output of script 02)
├── results/
│   ├── hrv_analysis_results.json       # Naive vs. leak-free AUC/accuracy/F1 per model
│   ├── hrv_loso_results.json           # Per-participant LOSO results
│   ├── hrv_ablation_results.json       # Feature-group ablation AUCs
│   ├── hrv_feature_importance.csv      # Permutation feature importance
│   ├── roc_curve.png                   # Figure 1 in the manuscript
│   └── confusion_matrix.png            # Figure 2 in the manuscript
├── requirements.txt
└── LICENSE
```

## Data source

The raw data are **not redistributed here**; they are publicly available from the original authors at:
https://github.com/italha-d/Stress-Predict-Dataset

Citation: Iqbal, T.; Simpkin, A.J.; Roshan, D.; Glynn, N.; Killilea, J.; Walsh, J.; Molloy, G.; Ganly, S.; Ryman, H.; Coen, E.; et al. Stress Monitoring Using Wearable Sensors: A Pilot Study and Stress-Predict Dataset. *Sensors* **2022**, *22*, 8135.

To reproduce the pipeline from scratch, clone the Stress-Predict dataset repository above and point `01_build_hrv_dataset.py` / `02_build_sequences.py` at it via the `STRESS_PREDICT_DIR` environment variable (defaults to `./Stress-Predict-Dataset`).

The **processed, de-identified HRV feature dataset** (`data/HRV_stress_dataset.csv`) and raw IBI sequence array (`data/HRV_sequences.npz`) generated from that raw data for this study **are** included here, so the analysis scripts (03–06) can be run directly without re-running the feature-extraction step.

## Reproducing the results

```bash
pip install -r requirements.txt

# (optional — only needed to rebuild data/ from scratch)
git clone https://github.com/italha-d/Stress-Predict-Dataset
python src/01_build_hrv_dataset.py
python src/02_build_sequences.py

# classical ML: naive vs. leak-free comparison, LOSO, ablation, ROC/confusion matrix
python src/03_final_analysis.py
python src/04_loso_analysis.py
python src/05_extra_analysis.py

# deep learning (1D-CNN / LSTM) — run in Kaggle Notebooks or Google Colab,
# or locally if TensorFlow is installed
python src/06_hrv_deep_learning.py
```

## Key results

| Model | Naive CV AUC | Leak-free CV AUC | Inflation |
|---|---|---|---|
| Logistic Regression | 0.629 | 0.624 | +0.004 |
| Random Forest | 0.665 | 0.628 | +0.037 |
| Gradient Boosting | 0.613 | 0.572 | +0.041 |

(See `results/hrv_analysis_results.json` for exact values.)

- **LOSO mean AUC**: 0.66 (Logistic Regression and Random Forest)
- **Final held-out Random Forest**: AUC = 0.638
- **1D-CNN**: AUC = 0.522; **LSTM**: AUC = 0.654
- **Time-domain features alone** matched the full 13-feature set (see `results/hrv_ablation_results.json`)

## Citation

If you use this code, please cite the paper (full citation to be added once published) and the original Stress-Predict dataset above.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The Stress-Predict dataset itself is governed by its own license/terms — see the [original repository](https://github.com/italha-d/Stress-Predict-Dataset).

## Contact

Zahra Bagheri — zra.bagherii@gmail.com — Department of Biomedical Engineering, North Tehran Branch, Islamic Azad University, Tehran, Iran.
