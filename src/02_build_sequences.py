# -*- coding: utf-8 -*-
"""
Build raw IBI SEQUENCES (not hand-crafted features) per 60s window, for a
deep-learning (LSTM / 1D-CNN) model. Same windows/labels as the feature
dataset, so results are directly comparable to the classical-ML results.
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.environ.get("STRESS_PREDICT_DIR", "./Stress-Predict-Dataset")
RAW_DIR = os.path.join(BASE, "Raw_data")
LABELS_CSV = os.path.join(BASE, "Processed_data", "Improved_All_Combined_hr_rsp_binary.csv")
OUT_NPZ = "./data/HRV_sequences.npz"

WINDOW_SEC = 60
MIN_BEATS_PER_WINDOW = 20
MAX_LEN = 110  # pad/truncate all beat sequences to this length (covers ~99th percentile)

def load_ibi(subject_dir):
    path = os.path.join(subject_dir, "IBI.csv")
    with open(path) as f:
        first_line = f.readline().strip()
    start_time = float(first_line.split(",")[0])
    df = pd.read_csv(path, skiprows=1, header=None, names=["offset_sec", "ibi_sec"])
    df["abs_time"] = start_time + df["offset_sec"]
    return df

def main():
    labels_df = pd.read_csv(LABELS_CSV).rename(columns={"Time(sec)": "time_sec"})
    subjects = sorted(d for d in os.listdir(RAW_DIR)
                       if os.path.isdir(os.path.join(RAW_DIR, d)) and d.startswith("S"))

    sequences, lengths, labels, participants, window_ids = [], [], [], [], []

    for subj in subjects:
        pid = int(subj.replace("S", ""))
        subj_labels = labels_df[labels_df["Participant"] == pid]
        if subj_labels.empty:
            continue
        label_lookup = dict(zip(subj_labels["time_sec"].astype(int), subj_labels["Label"].astype(int)))
        t_min, t_max = subj_labels["time_sec"].min(), subj_labels["time_sec"].max()

        ibi_df = load_ibi(os.path.join(RAW_DIR, subj))

        window_start = t_min
        win_idx = 0
        while window_start + WINDOW_SEC <= t_max:
            window_end = window_start + WINDOW_SEC
            sec_range = range(int(window_start), int(window_end))
            window_labels = [label_lookup.get(s) for s in sec_range]
            if any(l is None for l in window_labels) or len(set(window_labels)) != 1:
                window_start += WINDOW_SEC
                win_idx += 1
                continue
            label = window_labels[0]

            mask = (ibi_df["abs_time"] >= window_start) & (ibi_df["abs_time"] < window_end)
            beats = ibi_df.loc[mask]
            ibi_ms = beats["ibi_sec"].values * 1000.0
            valid = (ibi_ms > 300) & (ibi_ms < 2000)
            ibi_ms = ibi_ms[valid]

            if len(ibi_ms) < MIN_BEATS_PER_WINDOW:
                window_start += WINDOW_SEC
                win_idx += 1
                continue

            seq = ibi_ms[:MAX_LEN]
            lengths.append(len(seq))
            padded = np.zeros(MAX_LEN, dtype=np.float32)
            padded[:len(seq)] = seq
            sequences.append(padded)
            labels.append(label)
            participants.append(pid)
            window_ids.append(f"{subj}_w{win_idx}")

            window_start += WINDOW_SEC
            win_idx += 1

    X = np.stack(sequences)           # shape (n_windows, MAX_LEN)
    lengths = np.array(lengths)       # true (unpadded) length of each sequence
    y = np.array(labels)
    groups = np.array(participants)

    np.savez_compressed(OUT_NPZ, X=X, lengths=lengths, y=y, groups=groups,
                         window_ids=np.array(window_ids))

    print(f"Saved {OUT_NPZ}")
    print(f"X shape: {X.shape}  (windows x max_beats_per_window)")
    print(f"Participants: {len(set(groups))}")
    print(f"Label distribution: {np.bincount(y)}")
    print(f"Sequence length stats: min={lengths.min()} max={lengths.max()} "
          f"mean={lengths.mean():.1f} p99={np.percentile(lengths, 99):.0f}")

if __name__ == "__main__":
    main()
