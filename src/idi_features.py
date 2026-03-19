"""
idi_features.py
---------------
IDI Feature Extraction for the multinational validation study.

Computes 9 IDI temporal features from documentation event timestamps:

  Volume:
    idi_events_24h          - total documentation events in window
    idi_events_per_hour     - event rate per hour

  Surveillance Gap:
    idi_max_gap_min         - maximum inter-event interval (minutes)
    idi_gap_count_60m       - number of gaps > 60 minutes
    idi_gap_count_120m      - number of gaps > 120 minutes

  Rhythm Regularity:
    idi_mean_interevent_min - mean inter-event interval (minutes)
    idi_std_interevent_min  - SD of inter-event intervals (ddof=1)
    idi_cv_interevent       - coefficient of variation (SD / mean)
    idi_burstiness          - burstiness index B = (sigma - mu) / (sigma + mu)

Inter-event intervals are computed in seconds then converted to float minutes
to avoid integer truncation from timedelta64[m]. Sample standard deviation
(ddof=1) is used throughout to match the MIMIC-IV derivation repository.

If intime_series is provided, events are automatically filtered to the first
window_hours from ICU admission. If not provided, the caller must pre-filter
events before passing them to compute_idi_features().

Reference:
  Collier AM, Shalhout SZ. Intensive Documentation Index as an All-Cause
  Mortality Predictor in Critically Ill Patients: A Multi-Center External
  Validation Study. npj Digital Medicine, 2026.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_idi_features(
    events_df: pd.DataFrame,
    stay_id_col: str = "stay_id",
    time_col: str = "charttime",
    window_hours: int = 24,
    intime_series: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute 9 IDI temporal features from documentation event timestamps.

    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame with at minimum [stay_id_col, time_col].
        If intime_series is None, the caller must pre-filter events to the
        desired analysis window before calling this function.
    stay_id_col : str
        Column name for ICU stay identifier.
    time_col : str
        Column name for event timestamp (datetime or parseable string).
    window_hours : int
        Analysis window in hours (default 24). Used to filter events if
        intime_series is provided, and to compute idi_events_per_hour.
    intime_series : pd.Series or None
        If provided, must be indexed by stay_id values and contain the ICU
        admission datetime for each stay. Events outside [0, window_hours)
        from intime will be dropped. If None, no filtering is applied.

    Returns
    -------
    pd.DataFrame
        One row per stay_id with 9 IDI features.
        Stays with fewer than 2 valid intervals receive NaN for all features.
    """
    events_df = events_df.copy()
    events_df[time_col] = pd.to_datetime(events_df[time_col])
    events_df = events_df.sort_values([stay_id_col, time_col])

    results = []

    for stay_id, grp in events_df.groupby(stay_id_col):
        if intime_series is not None:
            intime = pd.to_datetime(intime_series.get(stay_id))
            if pd.isna(intime):
                results.append({stay_id_col: stay_id,
                                 **{f: np.nan for f in _feature_names()}})
                continue
            hours_elapsed = (grp[time_col] - intime).dt.total_seconds() / 3600
            grp = grp[(hours_elapsed >= 0) & (hours_elapsed <= window_hours)]

        times = grp[time_col].values
        n     = len(times)

        if n < 2:
            results.append({stay_id_col: stay_id,
                             **{f: np.nan for f in _feature_names()}})
            continue

        # Compute deltas in seconds then convert to float minutes
        deltas_sec = np.diff(times).astype("timedelta64[s]").astype(float)
        deltas     = deltas_sec / 60.0
        deltas     = deltas[deltas > 0]   # remove exact duplicate timestamps

        if len(deltas) == 0:
            results.append({stay_id_col: stay_id,
                             **{f: np.nan for f in _feature_names()}})
            continue

        mu  = float(np.mean(deltas))
        sig = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

        cv         = (sig / mu)       if mu > 0         else np.nan
        burstiness = ((sig - mu) / (sig + mu)) if (sig + mu) > 0 else np.nan

        results.append({
            stay_id_col:               stay_id,
            "idi_events_24h":          n,
            "idi_events_per_hour":     n / window_hours,
            "idi_mean_interevent_min": mu,
            "idi_std_interevent_min":  sig,
            "idi_cv_interevent":       cv,
            "idi_max_gap_min":         float(np.max(deltas)),
            "idi_gap_count_60m":       int(np.sum(deltas > 60)),
            "idi_gap_count_120m":      int(np.sum(deltas > 120)),
            "idi_burstiness":          burstiness,
        })

    return pd.DataFrame(results)


def _feature_names() -> list:
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
    Remove IDI features with |Pearson r| > r_threshold with ICU LOS.
    Non-survivors have longer ICU stays (median 71.3 h vs 53.1 h in MIMIC-IV),
    so features correlated with LOS are post-outcome proxies, not predictors.
    """
    idi_cols = [c for c in features_df.columns if c.startswith("idi_")]
    keep, dropped = [], []
    for col in idi_cols:
        r = features_df[[col, icu_los_col]].dropna().corr().iloc[0, 1]
        if abs(r) <= r_threshold:
            keep.append(col)
        else:
            dropped.append((col, round(r, 3)))

    if dropped:
        print(f"Leakage filter: dropped {len(dropped)} features:")
        for col, r in sorted(dropped, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {col}: r={r}")
    print(f"Retained {len(keep)} of {len(idi_cols)} IDI features after leakage filter.")
    drop_names = [d[0] for d in dropped]
    return features_df[[c for c in features_df.columns if c not in drop_names]]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Extract IDI features from an events CSV.\n"
            "Provide --cohort to enable automatic 24-hour window filtering.\n"
            "Without --cohort, events must already be filtered to the first "
            "window_hours from ICU admission."
        )
    )
    parser.add_argument("--cohort",  required=False,
                        help="Path to cohort CSV with stay_id and intime columns")
    parser.add_argument("--events",  required=True,
                        help="Path to events CSV with stay_id and charttime columns")
    parser.add_argument("--output",  default="data/processed/idi_features.csv")
    parser.add_argument("--window",  type=int, default=24,
                        help="Analysis window in hours (default 24)")
    args = parser.parse_args()

    events = pd.read_csv(args.events, parse_dates=["charttime"])

    intime_series = None
    if args.cohort:
        cohort = pd.read_csv(args.cohort, parse_dates=["intime"])
        intime_series = cohort.set_index("stay_id")["intime"]
        print(f"Auto-filtering events to first {args.window} h from intime.")
    else:
        print(f"WARNING: no --cohort provided. Events must be pre-filtered "
              f"to the first {args.window} hours from ICU admission.")

    features = compute_idi_features(
        events,
        intime_series=intime_series,
        window_hours=args.window,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"IDI features saved -> {args.output} ({len(features)} rows)")
