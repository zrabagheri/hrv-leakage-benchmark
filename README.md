# Heart Rate, Personal Calibration and Protocol-Order Confounding in Wrist-Worn Stress Detection

Code and derived data for the manuscript **"Heart Rate, Personal Calibration and Protocol-Order Confounding in Wrist-Worn Stress Detection"** (Zahra Bagheri, under review).

## Overview
A confound-aware re-analysis of the public [Stress-Predict dataset](https://github.com/italha-d/Stress-Predict-Dataset) (34 participants, Empatica E4). Each window is labelled by the **timing of the protocol's stress tasks** (Stroop test, interview, hyperventilation) versus baseline and relaxation periods, using the event tags recorded by the device.

The analyses test:
- Row-level vs participant-level cross-validation (20 repetitions, Nadeau–Bengio corrected resampled t-test)
- Heart rate alone vs HRV features vs variability-only features
- Personal baseline calibration and how much baseline data is needed
- Elapsed session time as a protocol-order shortcut, with detrending and permutation controls
- Multi-signal models (electrodermal activity, skin temperature, movement)
- Leave-one-task-block-out and a bidirectional rest control (each task vs the relaxation before and after it)

## Key results (60 s windows, logistic regression unless noted)
| Analysis | AUC |
|---|---|
| Row-level minus participant-level CV | −0.007 to +0.022 (all 95% intervals include 0) |
| Heart rate only / all HRV features / variability only | 0.655 / 0.634 / 0.508 |
| All features: raw → personally calibrated | 0.651 → 0.785 |
| Calibrated heart rate vs all wrist signals | 0.807 vs 0.811 |
| Elapsed time only (LR / RF) | 0.855 / 0.936 |
| Bidirectional control, heart rate (vs following / preceding rest) | 0.833 / 0.791 |

## Repository structure
```
src/
  build.py      # 60/120/300 s windows and HRV features from the raw data -> data/win*.csv
  multisig.py   # EDA, skin temperature and movement features -> data/win60_multi.csv
  exp.py        # evaluation schemes, feature sets, calibration      (modes: leak, fs, task, norm)
  exp2.py       # calibration statistics, LOSO, controls             (modes: calib, stats, loso, controls)
  exp3.py, exp4.py  # elapsed-time confound and detrending
  leak.py       # repeated row-level vs participant-level comparison (argument: LR, RF or GB)
  ms_exp.py     # multi-signal analyses                              (modes: sets, drift, tm)
  lobo.py       # leave-one-task-block-out
  prerest.py    # bidirectional rest comparison
  figs.py, fig5.py  # Figures 1-5 -> figures/
data/           # derived window-level datasets (output of build.py and multisig.py)
figures/        # manuscript figures
```

## Reproducing
Run all commands from the repository root.
```bash
pip install -r requirements.txt

# optional: rebuild data/ from the raw recordings
git clone https://github.com/italha-d/Stress-Predict-Dataset
python src/build.py
python src/multisig.py

# analyses (examples)
python src/exp.py fs
python src/exp2.py calib
python src/leak.py LR
python src/ms_exp.py sets
python src/lobo.py
python src/prerest.py

# figures
python src/figs.py
python src/fig5.py
```

## Data source
The raw data are not redistributed here. Please cite the original dataset:
Iqbal T, Simpkin AJ, Roshan D, Glynn N, Killilea J, Walsh J, et al. Stress monitoring using wearable sensors: a pilot study and Stress-Predict dataset. *Sensors* 2022;22:8135.

## Note on the first version
An earlier version of this repository accompanied a draft that described the labels as derived from heart rate and reported larger leakage effects from a single cross-validation split. Those scripts remain available in the repository history (commit `4fb18b3`); the current analysis in `src/` supersedes them.

## License
MIT (code). The Stress-Predict dataset is governed by its own terms.

## Contact
Zahra Bagheri — zra.bagherii@gmail.com
