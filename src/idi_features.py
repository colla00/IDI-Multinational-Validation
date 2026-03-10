"""
IDI Feature Extraction — npj Multinational Validation
Extracts 9 temporal IDI features from nursing documentation timestamps.
Compatible with both MIMIC-IV and HiRID timestamp formats.
"""

import pandas as pd
import numpy as np
from scipy.stats import variation
from pathlib import Path


def compute_idi_features(events_df: pd.DataFrame,
                          stay_id_col: str = "stay_id",
                          time_col: str = "charttime",
                          window_hours: int = 24) -> pd.DataFrame:
    """
    Compute 9 IDI temporal features from documentation event timestamps.

    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame with columns [stay_id_col, time_col].
    stay_id_col : str
        Column name for ICU stay identifier.
    time_col : str
        Column name for event timestamp (datetime).
    window_hours : int
        Analysis window (default 24 hours from ICU admission).

    Returns
    -------
    pd.DataFrame
        One row per stay_id with 9 IDI features.
    """
    events_df = events_df.copy()
    events_df[time_col] = pd.to_datetime(events_df[time_col])
    events_df = events_df.sort_values([stay_id_col, time_col])

    results = []

    for stay_id, grp in events_df.groupby(stay_id_col):
        times = grp[time_col].values
        n = len(times)

        if n < 2:
            results.append({stay_id_col: stay_id,
                             **{f: np.nan for f in _feature_names()}})
            continue

        # Inter-event intervals in minutes
        deltas = np.diff(times).astype("timedelta64[m]").astype(float)
        deltas = deltas[deltas > 0]  # remove exact duplicates

        if len(deltas) == 0:
            results.append({stay_id_col: stay_id,
                             **{f: np.nan for f in _feature_names()}})
            continue

        # 1. idi_events_24h — total events in window
        idi_events_24h = n

        # 2. idi_events_per_hour — mean hourly rate
        idi_events_per_hour = n / window_hours

        # 3. idi_mean_interevent_min — mean inter-event interval (minutes)
        idi_mean_interevent_min = float(np.mean(deltas))

        # 4. idi_std_interevent_min — std of inter-event intervals
        idi_std_interevent_min = float(np.std(deltas))

        # 5. idi_cv_interevent — coefficient of variation of intervals
        idi_cv_interevent = float(idi_std_interevent_min / idi_mean_interevent_min
                                   if idi_mean_interevent_min > 0 else np.nan)

        # 6. idi_max_gap_min — maximum gap between events
        idi_max_gap_min = float(np.max(deltas))

        # 7. idi_gap_count_60m — number of gaps > 60 minutes
        idi_gap_count_60m = int(np.sum(deltas > 60))

        # 8. idi_gap_count_120m — number of gaps > 120 minutes
        idi_gap_count_120m = int(np.sum(deltas > 120))

        # 9. idi_burstiness — burstiness parameter (Goh & Barabasi 2008)
        # B = (std - mean) / (std + mean), range [-1, 1]
        s, m = float(np.std(deltas)), float(np.mean(deltas))
        idi_burstiness = float((s - m) / (s + m)) if (s + m) > 0 else np.nan

        results.append({
            stay_id_col:               stay_id,
            "idi_events_24h":          idi_events_24h,
            "idi_events_per_hour":     idi_events_per_hour,
            "idi_mean_interevent_min": idi_mean_interevent_min,
            "idi_std_interevent_min":  idi_std_interevent_min,
            "idi_cv_interevent":       idi_cv_interevent,
            "idi_max_gap_min":         idi_max_gap_min,
            "idi_gap_count_60m":       idi_gap_count_60m,
            "idi_gap_count_120m":      idi_gap_count_120m,
            "idi_burstiness":          idi_burstiness,
        })

    return pd.DataFrame(results)


def _feature_names():
    return [
        "idi_events_24h", "idi_events_per_hour", "idi_mean_interevent_min",
        "idi_std_interevent_min", "idi_cv_interevent", "idi_max_gap_min",
        "idi_gap_count_60m", "idi_gap_count_120m", "idi_burstiness",
    ]


def apply_leakage_filter(features_df: pd.DataFrame,
                          outcome_col: str,
                          icu_los_col: str,
                          r_threshold: float = 0.30) -> pd.DataFrame:
    """
    Remove IDI features with |Pearson r| > r_threshold with ICU LOS
    to prevent reverse-causal temporal leakage.
    Non-survivors have longer ICU stays (median 71.3 vs 53.1 h in MIMIC-IV),
    so features correlated with LOS are post-outcome proxies, not predictors.
    """
    idi_cols = [c for c in features_df.columns if c.startswith("idi_")]
    keep = []
    dropped = []
    for col in idi_cols:
        r = features_df[[col, icu_los_col]].dropna().corr().iloc[0, 1]
        if abs(r) <= r_threshold:
            keep.append(col)
        else:
            dropped.append((col, round(r, 3)))

    if dropped:
        print(f"Leakage filter: dropped {len(dropped)} features:")
        for col, r in dropped:
            print(f"  {col}: r={r}")
    print(f"Retained {len(keep)} of {len(idi_cols)} IDI features after leakage filter.")
    return features_df[[c for c in features_df.columns if c not in [d[0] for d in dropped]]]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, help="Path to cohort CSV")
    parser.add_argument("--events", required=True, help="Path to events CSV")
    parser.add_argument("--output", default="data/processed/idi_features.csv")
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort)
    events = pd.read_csv(args.events)
    features = compute_idi_features(events)
    features.to_csv(args.output, index=False)
    print(f"IDI features saved to {args.output} ({len(features)} rows)")
