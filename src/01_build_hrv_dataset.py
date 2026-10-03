# -*- coding: utf-8 -*-
"""
Build a leak-free, window-based HRV feature dataset from the Stress-Predict
raw IBI data + the authors' own validated per-second stress/non-stress labels.

Pipeline:
  1. Load per-second ground-truth labels (Improved_All_Combined_hr_rsp_binary.csv)
  2. For each participant, load IBI.csv (beat-to-beat intervals from Empatica E4)
  3. Convert beat offsets to absolute unix time using each file's session start time
  4. Slice into non-overlapping 60s windows; keep only "pure" windows where every
     second in the window carries the same label (drops ambiguous transition windows)
  5. Compute standard time-domain, frequency-domain and non-linear HRV features
     per window
  6. Save the resulting tabular dataset + a data dictionary
"""
import os
import numpy as np
import pandas as pd
from scipy import signal, interpolate

BASE = os.environ.get("STRESS_PREDICT_DIR", "./Stress-Predict-Dataset")
RAW_DIR = os.path.join(BASE, "Raw_data")
LABELS_CSV = os.path.join(BASE, "Processed_data", "Improved_All_Combined_hr_rsp_binary.csv")
OUT_CSV = "./data/HRV_stress_dataset.csv"

WINDOW_SEC = 60          # window length for feature extraction
MIN_BEATS_PER_WINDOW = 20  # discard windows with too few detected beats (poor quality)

def load_ibi(subject_dir):
    """Return (start_unix_time, DataFrame[offset_sec, ibi_sec])."""
    path = os.path.join(subject_dir, "IBI.csv")
    with open(path) as f:
        first_line = f.readline().strip()
    start_time = float(first_line.split(",")[0])
    df = pd.read_csv(path, skiprows=1, header=None, names=["offset_sec", "ibi_sec"])
    df["abs_time"] = start_time + df["offset_sec"]
    return start_time, df

def hrv_time_domain(ibi_ms):
    """ibi_ms: array of IBI values in milliseconds."""
    diffs = np.diff(ibi_ms)
    sdnn = np.std(ibi_ms, ddof=1)
    rmssd = np.sqrt(np.mean(diffs ** 2))
    nn50 = np.sum(np.abs(diffs) > 50)
    pnn50 = 100.0 * nn50 / len(diffs) if len(diffs) > 0 else np.nan
    mean_ibi = np.mean(ibi_ms)
    mean_hr = 60000.0 / mean_ibi
    sdsd = np.std(diffs, ddof=1) if len(diffs) > 1 else np.nan
    cvnn = sdnn / mean_ibi
    return dict(mean_ibi=mean_ibi, sdnn=sdnn, rmssd=rmssd, pnn50=pnn50,
                mean_hr=mean_hr, sdsd=sdsd, cvnn=cvnn)

def hrv_poincare(ibi_ms):
    diffs = np.diff(ibi_ms)
    sd1 = np.sqrt(np.var(diffs, ddof=1) / 2.0) if len(diffs) > 1 else np.nan
    sd2 = np.sqrt(2 * np.var(ibi_ms, ddof=1) - np.var(diffs, ddof=1) / 2.0) if len(diffs) > 1 else np.nan
    sd_ratio = sd1 / sd2 if sd2 not in (0, np.nan) else np.nan
    return dict(sd1=sd1, sd2=sd2, sd1_sd2_ratio=sd_ratio)

def hrv_freq_domain(beat_times_sec, ibi_ms, fs_interp=4.0):
    """Interpolate the IBI series onto an evenly-sampled grid, then Welch PSD."""
    if len(beat_times_sec) < 4:
        return dict(lf_power=np.nan, hf_power=np.nan, lf_hf_ratio=np.nan, total_power=np.nan)
    t = beat_times_sec - beat_times_sec[0]
    # linear interpolation only (no extrapolation) to avoid cubic-spline
    # overshoot artifacts at the window edges
    f_interp = interpolate.interp1d(t, ibi_ms, kind="linear", bounds_error=False,
                                     fill_value=(ibi_ms[0], ibi_ms[-1]))
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs_interp)
    if len(t_uniform) < 8:
        return dict(lf_power=np.nan, hf_power=np.nan, lf_hf_ratio=np.nan, total_power=np.nan)
    ibi_uniform = f_interp(t_uniform)
    ibi_uniform = ibi_uniform - np.mean(ibi_uniform)  # detrend (remove DC)
    freqs, psd = signal.welch(ibi_uniform, fs=fs_interp, nperseg=min(256, len(ibi_uniform)))
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    total_mask = (freqs >= 0.0033) & (freqs < 0.40)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    lf_power = trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else np.nan
    hf_power = trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else np.nan
    total_power = trapz(psd[total_mask], freqs[total_mask]) if total_mask.any() else np.nan
    lf_hf_ratio = lf_power / hf_power if hf_power not in (0, np.nan) and not np.isnan(hf_power) else np.nan
    return dict(lf_power=lf_power, hf_power=hf_power, lf_hf_ratio=lf_hf_ratio, total_power=total_power)

def main():
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df = labels_df.rename(columns={"Time(sec)": "time_sec"})

    subjects = sorted(
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d)) and d.startswith("S")
    )

    rows = []
    skipped_subjects = []

    for subj in subjects:
        pid = int(subj.replace("S", ""))
        subj_labels = labels_df[labels_df["Participant"] == pid]
        if subj_labels.empty:
            skipped_subjects.append(subj)
            continue

        # per-second label lookup for this participant
        label_lookup = dict(zip(subj_labels["time_sec"].astype(int), subj_labels["Label"].astype(int)))
        t_min, t_max = subj_labels["time_sec"].min(), subj_labels["time_sec"].max()

        subj_dir = os.path.join(RAW_DIR, subj)
        try:
            _, ibi_df = load_ibi(subj_dir)
        except Exception as e:
            skipped_subjects.append(subj)
            continue

        # slide non-overlapping windows across the labelled time range
        window_start = t_min
        win_idx = 0
        while window_start + WINDOW_SEC <= t_max:
            window_end = window_start + WINDOW_SEC
            sec_range = range(int(window_start), int(window_end))
            window_labels = [label_lookup.get(s) for s in sec_range]
            if any(l is None for l in window_labels):
                window_start += WINDOW_SEC
                win_idx += 1
                continue
            if len(set(window_labels)) != 1:
                # ambiguous / transition window -> drop (leak-free labeling)
                window_start += WINDOW_SEC
                win_idx += 1
                continue
            label = window_labels[0]

            mask = (ibi_df["abs_time"] >= window_start) & (ibi_df["abs_time"] < window_end)
            beats = ibi_df.loc[mask]
            if len(beats) < MIN_BEATS_PER_WINDOW:
                window_start += WINDOW_SEC
                win_idx += 1
                continue

            ibi_ms = beats["ibi_sec"].values * 1000.0
            beat_times = beats["abs_time"].values

            # physiological plausibility filter (reject clearly bad beat detections)
            valid = (ibi_ms > 300) & (ibi_ms < 2000)
            if valid.sum() < MIN_BEATS_PER_WINDOW:
                window_start += WINDOW_SEC
                win_idx += 1
                continue
            ibi_ms = ibi_ms[valid]
            beat_times = beat_times[valid]

            feats = {}
            feats.update(hrv_time_domain(ibi_ms))
            feats.update(hrv_poincare(ibi_ms))
            feats.update(hrv_freq_domain(beat_times, ibi_ms))
            feats["n_beats"] = len(ibi_ms)
            feats["participant"] = pid
            feats["window_id"] = f"{subj}_w{win_idx}"
            feats["window_start_unix"] = window_start
            feats["label"] = label

            rows.append(feats)
            window_start += WINDOW_SEC
            win_idx += 1

    out_df = pd.DataFrame(rows)
    col_order = ["window_id", "participant", "window_start_unix", "n_beats",
                 "mean_ibi", "mean_hr", "sdnn", "rmssd", "pnn50", "sdsd", "cvnn",
                 "sd1", "sd2", "sd1_sd2_ratio",
                 "lf_power", "hf_power", "lf_hf_ratio", "total_power",
                 "label"]
    out_df = out_df[col_order]
    out_df = out_df.dropna()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print("Subjects processed:", len(subjects) - len(skipped_subjects))
    print("Subjects skipped (no validated labels / read error):", skipped_subjects)
    print("Total windows kept:", len(out_df))
    print("Participants in final dataset:", out_df["participant"].nunique())
    print(out_df["label"].value_counts())
    print(out_df.groupby("participant")["label"].value_counts().unstack(fill_value=0))
    print(out_df.describe().T[["mean", "std", "min", "max"]])

if __name__ == "__main__":
    main()
