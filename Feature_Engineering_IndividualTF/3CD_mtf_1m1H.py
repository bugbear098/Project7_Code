#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine 1H 3CD features with 1m base data (start-oriented) and apply:
- Revised early purge counting (count only at purge, and only if minute != 59)
- Persistence trimming: effective 1H features are zeroed from the purge minute
  forward until a new draw ID appears or the ID clears.

We keep the raw aligned 1H columns intact and write edited copies into *_eff columns.
"""

import os
import pandas as pd
import numpy as np

# =========================
# CONFIG — edit paths
# =========================
INPUT_DIR  = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_Data/Data_Medium/Feature_SandPit/3CD"
OUTPUT_DIR = os.path.join(INPUT_DIR, "Processed")

FILE_1M = "Data_1m_with_3cd.csv"
FILE_1H = "resampled_1H_with_3cd.csv"

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_1m_1H_with_eff.csv")

# =========================
# Helpers
# =========================
def _ensure_dt64(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Ensure datetime is parsed and tz-naive UTC."""
    df[col] = pd.to_datetime(df[col], errors="raise", utc=True)
    df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    return df

# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load 1m
    base = pd.read_csv(os.path.join(INPUT_DIR, FILE_1M))
    base = _ensure_dt64(base, "datetime").sort_values("datetime").reset_index(drop=True)

    # Load 1H (start-oriented)
    tf = pd.read_csv(os.path.join(INPUT_DIR, FILE_1H))
    tf = _ensure_dt64(tf, "datetime").sort_values("datetime").reset_index(drop=True)

    # Columns to carry over from 1H
    cols = [
        "third_candle_draw",
        "third_candle_range",
        "third_candle_body_to_range",
        "anchor_high",
        "anchor_low",
        "draw_started_at",
    ]
    keep = ["datetime"] + [c for c in cols if c in tf.columns]
    tf_keep = tf[keep].copy()

    # Align 1H -> 1m (backward: most recent 1H at-or-before each minute)
    combined = pd.merge_asof(
        base.sort_values("datetime"),
        tf_keep.sort_values("datetime"),
        on="datetime",
        direction="backward",
        suffixes=("", "_1H"),
    )

    # Prefix aligned 1H columns
    rename_map = {c: f"1H_{c}" for c in cols if c in combined.columns}
    combined = combined.rename(columns=rename_map)

    # Dtypes for 1H aligned columns
    combined["1H_third_candle_draw"] = (
        pd.to_numeric(combined.get("1H_third_candle_draw", 0), errors="coerce")
        .fillna(0)
        .astype("int8")
    )
    combined["1H_third_candle_range"] = (
        pd.to_numeric(combined.get("1H_third_candle_range", 0.0), errors="coerce")
        .fillna(0.0)
    )
    combined["1H_third_candle_body_to_range"] = (
        pd.to_numeric(combined.get("1H_third_candle_body_to_range", 0.0), errors="coerce")
        .fillna(0.0)
    )
    combined["1H_anchor_high"] = pd.to_numeric(combined.get("1H_anchor_high", np.nan), errors="coerce")
    combined["1H_anchor_low"]  = pd.to_numeric(combined.get("1H_anchor_low",  np.nan), errors="coerce")
    combined["1H_draw_started_at"] = pd.to_datetime(combined.get("1H_draw_started_at", pd.NaT), errors="coerce")

    # Create EFFECTIVE copies we will edit; raw columns remain untouched
    combined["1H_third_candle_draw_eff"]           = combined["1H_third_candle_draw"].astype("int8")
    combined["1H_third_candle_range_eff"]          = combined["1H_third_candle_range"]
    combined["1H_third_candle_body_to_range_eff"]  = combined["1H_third_candle_body_to_range"]
    combined["1H_anchor_high_eff"]                 = combined["1H_anchor_high"]
    combined["1H_anchor_low_eff"]                  = combined["1H_anchor_low"]
    combined["1H_draw_started_at_eff"]             = combined["1H_draw_started_at"]

    # === Revised counting + trimming ===
    early_purge_count = 0
    total_3CD_events  = 0

    n = len(combined)
    in_draw = False
    current_id = pd.NaT
    current_sign = 0
    latched_ah = np.nan
    latched_al = np.nan

    dt_index = combined["datetime"]
    minute_of_hour = dt_index.dt.minute.to_numpy()  # 0..59

    hi = pd.to_numeric(combined["high"], errors="coerce").to_numpy()
    lo = pd.to_numeric(combined["low"],  errors="coerce").to_numpy()

    draw_series = combined["1H_third_candle_draw"].to_numpy()
    id_series   = combined["1H_draw_started_at"].to_numpy(dtype="datetime64[ns]")
    ah_series   = combined["1H_anchor_high"].to_numpy()
    al_series   = combined["1H_anchor_low"].to_numpy()

    # For counting total 3CD events: increment on first minute a new non-NaT ID appears
    prev_id_seen = pd.NaT

    def zero_eff_span(start_idx: int, end_idx_excl: int):
        num_cols_zero = [
            "1H_third_candle_draw_eff",
            "1H_third_candle_range_eff",
            "1H_third_candle_body_to_range_eff",
            "1H_anchor_high_eff",
            "1H_anchor_low_eff",
        ]
        if start_idx < end_idx_excl:
            combined.loc[start_idx:end_idx_excl-1, num_cols_zero] = 0
            combined.loc[start_idx:end_idx_excl-1, "1H_draw_started_at_eff"] = pd.NaT

    i = 0
    while i < n:
        sign = int(draw_series[i]) if not np.isnan(draw_series[i]) else 0
        did  = pd.NaT if pd.isna(id_series[i]) else pd.to_datetime(id_series[i])

        # --- total_3CD_events: +1 when a new non-NaT ID first appears ---
        if (not pd.isna(did)) and (pd.isna(prev_id_seen) or did != prev_id_seen):
            total_3CD_events += 1
            prev_id_seen = did

        if not in_draw:
            # Activate only when we have a non-zero draw and a non-NaT ID
            if (sign != 0) and (not pd.isna(did)):
                in_draw = True
                current_id = did
                current_sign = sign
                latched_ah = ah_series[i]
                latched_al = al_series[i]
                # NOTE: per new rule, DO NOT increment early_purge_count here
        else:
            # If draw ends without purge (ID goes NaT or changes, or sign becomes 0), reset
            if (sign == 0) or pd.isna(did) or (did != current_id):
                in_draw = False
                current_id = pd.NaT
                current_sign = 0
                latched_ah = np.nan
                latched_al = np.nan
                i += 1
                continue

            # Strict purge checks (NO tolerance):
            if current_sign == 1:
                purged_here = (not np.isnan(latched_ah)) and (hi[i] > latched_ah)
            else:  # current_sign == -1
                purged_here = (not np.isnan(latched_al)) and (lo[i] < latched_al)

            if purged_here:
                # Count early purge ONLY if minute != 59
                if minute_of_hour[i] != 59:
                    early_purge_count += 1

                # Trim effective columns from this minute forward until ID changes/clears or sign==0
                j = i
                while j < n:
                    sign_j = int(draw_series[j]) if not np.isnan(draw_series[j]) else 0
                    did_j  = pd.NaT if pd.isna(id_series[j]) else pd.to_datetime(id_series[j])
                    if (sign_j == 0) or pd.isna(did_j) or (did_j != current_id):
                        break
                    j += 1

                zero_eff_span(i, j)

                # Reset state after purge
                in_draw = False
                current_id = pd.NaT
                current_sign = 0
                latched_ah = np.nan
                latched_al = np.nan

                i += 1
                continue

        i += 1

    # Save
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Wrote combined file: {OUTPUT_FILE}")
    print(f"[INFO] total_3CD_events = {total_3CD_events}")
    print(f"[INFO] early_purge_count = {early_purge_count}")

if __name__ == "__main__":
    main()
