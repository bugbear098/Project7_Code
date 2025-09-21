#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTF 3CD combiner (1m base) using OPEN-STATE alignment and strict, ID-aware early-purge logic.

For each TF in {5m, 15m, 1H, 4H, 1D}:
  - Build 1H/TF 'open-state' by shifting close-state features back 1 bar.
  - Align TF open-state onto 1m with merge_asof(direction="backward").
  - Strict purge: bull if 1m.high > anchor_high; bear if 1m.low < anchor_low.
  - Count early purge once per draw ID, only if minute != 59.
  - Zero *_eff columns from purge minute forward until ID changes/clears.
  - Write per-TF CSV with raw open-state and *_eff columns.

Then write a final combined CSV containing 1m base + only the *_eff features for all TFs
(drop anchor_*_eff and draw_started_at_eff in the final combined file).
"""

import os
from typing import Optional, Set, Dict, Tuple, List
import pandas as pd
import numpy as np

# =========================
# CONFIG — edit paths
# =========================
INPUT_DIR  = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_Data/Data_Medium/Feature_SandPit/3CD"
OUTPUT_DIR = os.path.join(INPUT_DIR, "Processed")

FILE_1M = "Data_1m_with_3cd.csv"
TF_FILES: Dict[str, str] = {
    "5m":  "resampled_5m_with_3cd.csv",
    "15m": "resampled_15m_with_3cd.csv",
    "1H":  "resampled_1H_with_3cd.csv",
    "4H":  "resampled_4H_with_3cd.csv",
    "1D":  "resampled_1D_with_3cd.csv",
}

FINAL_COMBINED = os.path.join(OUTPUT_DIR, "mtf_eff_combined_base1m.csv")

# =========================
# Helpers
# =========================
def _ensure_dt64(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="raise", utc=True)
    df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    return df

def build_open_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build TF 'open-state' by shifting close-state columns back one bar.
    Returns: datetime, draw_open, anchor_high_open, anchor_low_open, range_open, ratio_open, draw_id_open
    """
    need = {
        "third_candle_draw",
        "anchor_high", "anchor_low",
        "third_candle_range", "third_candle_body_to_range",
        "draw_started_at",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"TF file missing required columns: {missing}")

    out = df.copy()
    out["draw_open"]            = out["third_candle_draw"].shift(1)
    out["anchor_high_open"]     = out["anchor_high"].shift(1)
    out["anchor_low_open"]      = out["anchor_low"].shift(1)
    out["range_open"]           = out["third_candle_range"].shift(1)
    out["ratio_open"]           = out["third_candle_body_to_range"].shift(1)
    out["draw_id_open"]         = pd.to_datetime(out["draw_started_at"].shift(1), errors="coerce")

    out["draw_open"]  = pd.to_numeric(out["draw_open"], errors="coerce").fillna(0).astype("int8")
    out["range_open"] = pd.to_numeric(out["range_open"], errors="coerce").fillna(0.0)
    out["ratio_open"] = pd.to_numeric(out["ratio_open"], errors="coerce").fillna(0.0)
    out["anchor_high_open"] = pd.to_numeric(out["anchor_high_open"], errors="coerce")
    out["anchor_low_open"]  = pd.to_numeric(out["anchor_low_open"],  errors="coerce")

    return out[[
        "datetime",
        "draw_open", "anchor_high_open", "anchor_low_open",
        "range_open", "ratio_open", "draw_id_open",
    ]]

def id_key(ts: Optional[pd.Timestamp]) -> Optional[int]:
    """Stable integer key (ns since epoch) for a Timestamp, or None for NaT/None."""
    if ts is None or pd.isna(ts):
        return None
    return int(pd.Timestamp(ts).value)

def process_one_tf(
    base: pd.DataFrame,
    tf_path: str,
    tf_label: str,
    out_dir: str
) -> Tuple[pd.DataFrame, int, int]:
    """
    Process a single TF against the 1m base:
      - build open-state for TF
      - align to 1m
      - run strict, ID-aware early-purge detection
      - write per-TF combined CSV
    Returns:
      (combined_df, total_3CD_events, early_purge_count)
    """
    if not os.path.exists(tf_path):
        raise FileNotFoundError(f"{tf_label} file not found: {tf_path}")

    tf = pd.read_csv(tf_path)
    tf = _ensure_dt64(tf, "datetime").sort_values("datetime").reset_index(drop=True)
    tf_open = build_open_state(tf).sort_values("datetime").reset_index(drop=True)

    # Align TF open-state → 1m (backward)
    combined = pd.merge_asof(
        base.sort_values("datetime"),
        tf_open.sort_values("datetime"),
        on="datetime",
        direction="backward",
        suffixes=("", "")
    )

    # Rename aligned columns with TF_ prefix
    combined.rename(columns={
        "draw_open":            f"{tf_label}_open_draw",
        "anchor_high_open":     f"{tf_label}_open_anchor_high",
        "anchor_low_open":      f"{tf_label}_open_anchor_low",
        "range_open":           f"{tf_label}_open_range",
        "ratio_open":           f"{tf_label}_open_ratio",
        "draw_id_open":         f"{tf_label}_open_draw_id",
    }, inplace=True)

    # Types
    combined[f"{tf_label}_open_draw"]    = pd.to_numeric(combined[f"{tf_label}_open_draw"], errors="coerce").fillna(0).astype("int8")
    combined[f"{tf_label}_open_range"]   = pd.to_numeric(combined[f"{tf_label}_open_range"], errors="coerce").fillna(0.0)
    combined[f"{tf_label}_open_ratio"]   = pd.to_numeric(combined[f"{tf_label}_open_ratio"], errors="coerce").fillna(0.0)
    combined[f"{tf_label}_open_anchor_high"] = pd.to_numeric(combined[f"{tf_label}_open_anchor_high"], errors="coerce")
    combined[f"{tf_label}_open_anchor_low"]  = pd.to_numeric(combined[f"{tf_label}_open_anchor_low"],  errors="coerce")
    combined[f"{tf_label}_open_draw_id"]     = pd.to_datetime(combined[f"{tf_label}_open_draw_id"], errors="coerce")

    # Effective copies (edited on purge); raw open-state retained
    combined[f"{tf_label}_third_candle_draw_eff"]          = combined[f"{tf_label}_open_draw"].astype("int8")
    combined[f"{tf_label}_third_candle_range_eff"]         = combined[f"{tf_label}_open_range"]
    combined[f"{tf_label}_third_candle_body_to_range_eff"] = combined[f"{tf_label}_open_ratio"]
    combined[f"{tf_label}_anchor_high_eff"]                = combined[f"{tf_label}_open_anchor_high"]
    combined[f"{tf_label}_anchor_low_eff"]                 = combined[f"{tf_label}_open_anchor_low"]
    combined[f"{tf_label}_draw_started_at_eff"]            = combined[f"{tf_label}_open_draw_id"]

    # === Counting & trimming (STRICT, ID-aware) ===
    early_purge_count = 0
    total_3CD_events  = 0

    n = len(combined)
    in_draw: bool = False
    current_id_ts: Optional[pd.Timestamp] = None
    current_sign: int = 0
    latched_ah = np.nan
    latched_al = np.nan

    dt_index = combined["datetime"]
    minute_of_hour = dt_index.dt.minute.to_numpy()  # 0..59

    hi = pd.to_numeric(combined["high"], errors="coerce").to_numpy()
    lo = pd.to_numeric(combined["low"],  errors="coerce").to_numpy()

    sign_series = combined[f"{tf_label}_open_draw"].to_numpy()
    id_series   = pd.to_datetime(combined[f"{tf_label}_open_draw_id"], errors="coerce")
    ah_series   = combined[f"{tf_label}_open_anchor_high"].to_numpy()
    al_series   = combined[f"{tf_label}_open_anchor_low"].to_numpy()

    # Sets to manage unique events and suppression
    seen_ids: Set[int]   = set()  # for total_3CD_events
    killed_ids: Set[int] = set()  # draw IDs already early-purged (suppress re-activation)

    def zero_eff_span(start_idx: int, end_idx_excl: int):
        """Zero effective TF features on [start_idx, end_idx_excl)."""
        if start_idx >= end_idx_excl:
            return
        num_cols_zero = [
            f"{tf_label}_third_candle_draw_eff",
            f"{tf_label}_third_candle_range_eff",
            f"{tf_label}_third_candle_body_to_range_eff",
            f"{tf_label}_anchor_high_eff",
            f"{tf_label}_anchor_low_eff",
        ]
        combined.loc[start_idx:end_idx_excl-1, num_cols_zero] = 0
        combined.loc[start_idx:end_idx_excl-1, f"{tf_label}_draw_started_at_eff"] = pd.NaT

    i = 0
    while i < n:
        raw_sign = int(sign_series[i]) if not np.isnan(sign_series[i]) else 0
        raw_id_ts = id_series.iloc[i] if i < len(id_series) else pd.NaT
        raw_id_k = id_key(raw_id_ts)

        # Count total unique 3CD events (first time a non-NaT ID appears on 1m grid)
        if raw_id_k is not None and raw_id_k not in seen_ids:
            total_3CD_events += 1
            seen_ids.add(raw_id_k)

        if not in_draw:
            # Activate only if open-state draw is active, ID is valid, and not already killed
            if (raw_sign != 0) and (raw_id_k is not None) and (raw_id_k not in killed_ids):
                in_draw = True
                current_id_ts = raw_id_ts
                current_sign = raw_sign
                latched_ah = ah_series[i]
                latched_al = al_series[i]
        else:
            # End without purge if sign goes 0, ID clears, or ID changes
            cur_k = id_key(current_id_ts)
            if (raw_sign == 0) or (raw_id_k is None) or (raw_id_k != cur_k):
                in_draw = False
                current_id_ts = None
                current_sign = 0
                latched_ah = np.nan
                latched_al = np.nan
                i += 1
                continue

            # STRICT purge checks (no tolerance)
            if current_sign == 1:
                purged = (not np.isnan(latched_ah)) and (hi[i] > latched_ah)
            else:  # current_sign == -1
                purged = (not np.isnan(latched_al)) and (lo[i] < latched_al)

            if purged:
                # Count once per ID and only if minute != 59
                if (raw_id_k not in killed_ids) and (minute_of_hour[i] != 59):
                    early_purge_count += 1

                # Mark ID as killed to suppress re-activation within the same hour
                if raw_id_k is not None:
                    killed_ids.add(raw_id_k)

                # Zero effective features from this minute forward until ID changes/clears or sign==0
                j = i
                while j < n:
                    s_j = int(sign_series[j]) if not np.isnan(sign_series[j]) else 0
                    id_j_k = id_key(id_series.iloc[j]) if j < len(id_series) else None
                    if (s_j == 0) or (id_j_k is None) or (id_j_k != raw_id_k):
                        break
                    j += 1
                zero_eff_span(i, j)

                # Reset state
                in_draw = False
                current_id_ts = None
                current_sign = 0
                latched_ah = np.nan
                latched_al = np.nan

                i += 1
                continue

        i += 1

    # Write per-TF combined file
    out_path = os.path.join(out_dir, f"combined_1m_{tf_label}_openstate_eff.csv")
    combined.to_csv(out_path, index=False)
    print(f"[OK] Wrote {tf_label}: {out_path}")
    print(f"[INFO] {tf_label}: total_3CD_events = {total_3CD_events}")
    print(f"[INFO] {tf_label}: early_purge_count = {early_purge_count}")

    return combined, total_3CD_events, early_purge_count

# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load 1m base
    base = pd.read_csv(os.path.join(INPUT_DIR, FILE_1M))
    base = _ensure_dt64(base, "datetime").sort_values("datetime").reset_index(drop=True)

    # Ensure expected base cols exist (we keep base intact in outputs)
    needed_base = {"datetime", "open", "high", "low", "close", "volume"}
    missing = needed_base - set(base.columns)
    if missing:
        print(f"[WARN] Base file missing columns (continuing): {missing}")

    # Process each TF individually and write per-TF CSVs
    eff_slices: List[pd.DataFrame] = []
    for tf_label, tf_file in TF_FILES.items():
        tf_path = os.path.join(INPUT_DIR, tf_file)
        combined_tf, total_events, early_purges = process_one_tf(base, tf_path, tf_label, OUTPUT_DIR)

        # Prepare a slim slice with only the 3 eff features (for final MTF combined file)
        eff_slice = combined_tf[[
            "datetime",
            f"{tf_label}_third_candle_draw_eff",
            f"{tf_label}_third_candle_range_eff",
            f"{tf_label}_third_candle_body_to_range_eff",
        ]].copy()
        eff_slices.append(eff_slice)

    # Build final combined file: 1m base + only eff features across all TFs
    final = base.copy().sort_values("datetime")
    for sl in eff_slices:
        final = pd.merge_asof(
            final.sort_values("datetime"),
            sl.sort_values("datetime"),
            on="datetime",
            direction="backward",
            suffixes=("", "")
        )

    final.to_csv(FINAL_COMBINED, index=False)
    print(f"[OK] Wrote final MTF eff-only file: {FINAL_COMBINED}")

if __name__ == "__main__":
    main()
