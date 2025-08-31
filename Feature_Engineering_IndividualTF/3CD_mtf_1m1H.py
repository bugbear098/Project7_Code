#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine 1H 3CD features with 1m base data (start-oriented).
Implements simple 'early purge' accounting + trims the persistence of the 1H features
by writing edited copies into *_eff columns.

Rules (as requested):
- Scan minutes; when a non-zero 1H draw appears, add +1 to early_purge_count.
- If a purge is detected on the 59th minute of the hour, subtract 1 (not early).
- On purge, set the *effective* 1H feature columns to 0 (or NaT for the ID) from that minute
  forward until a new draw_started_at appears (or the ID becomes empty).
- If no purge occurs (i.e., the draw ends by invalidation at the 1H close), we do not adjust
  the count (per your current approach).
"""

import os
import pandas as pd
import numpy as np

# =========================
# CONFIG — edit paths
# =========================

INPUT_DIR  = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_Data/Data_Medium/Feature_SandPit/3CD"
OUTPUT_DIR = os.path.join(INPUT_DIR, "Processed")   # or just INPUT_DIR

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
    base = _ensure_dt64(base, "datetime")
    base = base.sort_values("datetime").reset_index(drop=True)

    # Load 1H (start-oriented)
    tf = pd.read_csv(os.path.join(INPUT_DIR, FILE_1H))
    tf = _ensure_dt64(tf, "datetime")
    tf = tf.sort_values("datetime").reset_index(drop=True)

    # Columns to carry over from 1H
    cols = [
        "third_candle_draw",
        "third_candle_range",
        "third_candle_body_to_range",
        "anchor_high",
        "anchor_low",
        "draw_started_at"
    ]
    keep = ["datetime"] + [c for c in cols if c in tf.columns]
    tf_keep = tf[keep].copy()

    # Merge-asof (backward): for each 1m row, get the most recent 1H row <= that minute
    combined = pd.merge_asof(
        base.sort_values("datetime"),
        tf_keep.sort_values("datetime"),
        on="datetime",
        direction="backward",
        suffixes=("", "_1H")
    )

    # Rename the aligned 1H cols with prefix
    rename_map = {c: f"1H_{c}" for c in cols if c in combined.columns}
    combined = combined.rename(columns=rename_map)

    # Ensure types for aligned 1H columns we’ll use
    combined["1H_third_candle_draw"] = pd.to_numeric(combined.get("1H_third_candle_draw", 0), errors="coerce").fillna(0).astype("int8")
    combined["1H_third_candle_range"] = pd.to_numeric(combined.get("1H_third_candle_range", 0.0), errors="coerce").fillna(0.0)
    combined["1H_third_candle_body_to_range"] = pd.to_numeric(combined.get("1H_third_candle_body_to_range", 0.0), errors="coerce").fillna(0.0)
    combined["1H_anchor_high"] = pd.to_numeric(combined.get("1H_anchor_high", np.nan), errors="coerce")
    combined["1H_anchor_low"]  = pd.to_numeric(combined.get("1H_anchor_low", np.nan), errors="coerce")
    combined["1H_draw_started_at"] = pd.to_datetime(combined.get("1H_draw_started_at", pd.NaT), errors="coerce")

    # Create EFFECTIVE copies we will edit (raw columns remain untouched)
    combined["1H_third_candle_draw_eff"]            = combined["1H_third_candle_draw"].astype("int8")
    combined["1H_third_candle_range_eff"]           = combined["1H_third_candle_range"]
    combined["1H_third_candle_body_to_range_eff"]   = combined["1H_third_candle_body_to_range"]
    combined["1H_anchor_high_eff"]                  = combined["1H_anchor_high"]
    combined["1H_anchor_low_eff"]                   = combined["1H_anchor_low"]
    combined["1H_draw_started_at_eff"]              = combined["1H_draw_started_at"]

    # Early purge counting + trimming (per your spec)
    early_purge_count = 0

    n = len(combined)
    in_draw = False
    current_id = pd.NaT
    current_sign = 0
    latched_ah = np.nan
    latched_al = np.nan

    # Arrays for quick access
    dt_index = combined["datetime"]
    minute_of_hour = dt_index.dt.minute.to_numpy()
    hi = pd.to_numeric(combined["high"], errors="coerce").to_numpy()
    lo = pd.to_numeric(combined["low"], errors="coerce").to_numpy()

    draw_series = combined["1H_third_candle_draw"].to_numpy()
    id_series   = combined["1H_draw_started_at"].to_numpy(dtype="datetime64[ns]")
    ah_series   = combined["1H_anchor_high"].to_numpy()
    al_series   = combined["1H_anchor_low"].to_numpy()

    # Helper to zero effective columns on a span [start_idx, end_idx_exclusive)
    def zero_eff_span(start_idx: int, end_idx_excl: int):
        cols_eff_num0 = [
            "1H_third_candle_draw_eff",
            "1H_third_candle_range_eff",
            "1H_third_candle_body_to_range_eff",
            "1H_anchor_high_eff",
            "1H_anchor_low_eff",
        ]
        combined.loc[start_idx:end_idx_excl-1, cols_eff_num0] = 0
        combined.loc[start_idx:end_idx_excl-1, "1H_draw_started_at_eff"] = pd.NaT

    i = 0
    while i < n:
        sign = int(draw_series[i]) if not np.isnan(draw_series[i]) else 0
        did  = pd.NaT if pd.isna(id_series[i]) else pd.to_datetime(id_series[i])

        if not in_draw:
            if sign != 0:
                # New (or continuing) active draw encountered → count it
                in_draw = True
                current_id = did
                current_sign = sign
                latched_ah = ah_series[i]
                latched_al = al_series[i]
                early_purge_count += 1
        else:
            # If ID disappears or changes, or sign becomes 0 → draw ended naturally
            if (sign == 0) or (pd.isna(did)) or (current_id is not pd.NaT and did != current_id):
                in_draw = False
                current_id = pd.NaT
                current_sign = 0
                latched_ah = np.nan
                latched_al = np.nan
                # do not change early_purge_count per your rules
                i += 1
                continue

            # Check for purge against latched anchors
            if current_sign == 1:
                purged_here = (not np.isnan(latched_ah)) and (hi[i] >= latched_ah)
            else:  # current_sign == -1
                purged_here = (not np.isnan(latched_al)) and (lo[i] <= latched_al)

            if purged_here:
                # If purge happens on minute 59, not an early purge → subtract 1
                if minute_of_hour[i] == 59:
                    early_purge_count -= 1

                # Trim effective columns from *this* minute forward
                j = i
                # continue zeroing until the raw 1H draw ID changes or becomes NaT or sign becomes 0
                while j < n:
                    sign_j = int(draw_series[j]) if not np.isnan(draw_series[j]) else 0
                    did_j  = pd.NaT if pd.isna(id_series[j]) else pd.to_datetime(id_series[j])
                    if (sign_j == 0) or (pd.isna(did_j)) or (did_j != current_id):
                        break
                    j += 1

                zero_eff_span(i, j)

                # Reset state
                in_draw = False
                current_id = pd.NaT
                current_sign = 0
                latched_ah = np.nan
                latched_al = np.nan

                # Continue from next row after the trimmed span start (we keep scanning; there might be a new draw later)
                i += 1
                continue

        i += 1

    # Save
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Wrote combined file: {OUTPUT_FILE}")
    print(f"[INFO] early_purge_count (per your counting rule) = {early_purge_count}")

if __name__ == "__main__":
    main()
