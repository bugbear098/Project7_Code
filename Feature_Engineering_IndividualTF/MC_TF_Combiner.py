#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTF Momentum Candle (MC) combiner (1m base) using OPEN-STATE alignment and strict purge logic.

For each TF in {5m, 15m, 1H, 4H, 1D}:
  - Build TF 'open-state' by shifting close-state MC features back 1 bar.
  - Align TF open-state onto 1m with merge_asof(direction="backward").
  - Purge detection: bull if 1m.high > anchor_high; bear if 1m.low < anchor_low.
  - Zero *_eff columns from purge minute forward until TF state changes.
  - Write per-TF CSV with raw open-state and *_eff columns.

Then write a final combined CSV containing 1m base + only the *_eff features for all TFs
(drop anchor_*_eff in the final combined file).
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List

# =========================
# CONFIG — edit paths
# =========================
INPUT_DIR  = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Feature_SandPit/MC/Processed"
OUTPUT_DIR = os.path.join(INPUT_DIR, "Processed")

FILE_1M = "resampled_1m_with_momentum.csv"
TF_FILES: Dict[str, str] = {
    "5m":  "resampled_5m_with_momentum.csv",
    "15m": "resampled_15m_with_momentum.csv",
    "1H":  "resampled_1H_with_momentum.csv",
    "4H":  "resampled_4H_with_momentum.csv",
    "1D":  "resampled_1D_with_momentum.csv",
}

FINAL_COMBINED = os.path.join(OUTPUT_DIR, "mtf_mc_combined_base1m.csv")

# =========================
# Helpers
# =========================
def _ensure_dt64(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Ensure datetime column is parsed and converted to America/New_York with DST preserved."""
    df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert("America/New_York")
    return df

def build_open_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build TF 'open-state' by shifting close-state columns back one bar.
    Returns: datetime, mc_open, anchor_high_open, anchor_low_open, strength_open, wick_to_body_open, range_open
    """
    need = {
        "momentum_candle_active",
        "momentum_strength", "wick_to_body", "total_range",
        "high", "low"
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"TF file missing required columns: {missing}")

    out = df.copy()
    out["mc_open"]           = out["momentum_candle_active"].shift(1)
    out["strength_open"]     = out["momentum_strength"].shift(1)
    out["wick_to_body_open"] = out["wick_to_body"].shift(1)
    out["range_open"]        = out["total_range"].shift(1)
    out["anchor_high_open"]  = out["high"].shift(1)
    out["anchor_low_open"]   = out["low"].shift(1)

    out["mc_open"]       = pd.to_numeric(out["mc_open"], errors="coerce").fillna(0).astype("int8")
    out["strength_open"] = pd.to_numeric(out["strength_open"], errors="coerce").fillna(0.0)
    out["wick_to_body_open"] = pd.to_numeric(out["wick_to_body_open"], errors="coerce").fillna(0.0)
    out["range_open"]    = pd.to_numeric(out["range_open"], errors="coerce").fillna(0.0)
    out["anchor_high_open"] = pd.to_numeric(out["anchor_high_open"], errors="coerce")
    out["anchor_low_open"]  = pd.to_numeric(out["anchor_low_open"], errors="coerce")

    return out[[
        "datetime",
        "mc_open", "strength_open", "wick_to_body_open", "range_open",
        "anchor_high_open", "anchor_low_open"
    ]]

def process_one_tf(
    base: pd.DataFrame,
    tf_path: str,
    tf_label: str,
    out_dir: str
) -> pd.DataFrame:
    """
    Process a single TF against the 1m base:
      - build open-state for TF
      - align to 1m
      - run strict purge detection
      - write per-TF combined CSV
    Returns combined dataframe
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
        "mc_open":           f"{tf_label}_open_mc",
        "strength_open":     f"{tf_label}_open_strength",
        "wick_to_body_open": f"{tf_label}_open_wick_to_body",
        "range_open":        f"{tf_label}_open_range",
        "anchor_high_open":  f"{tf_label}_open_anchor_high",
        "anchor_low_open":   f"{tf_label}_open_anchor_low",
    }, inplace=True)

    # Effective copies (edited on purge); raw open-state retained
    combined[f"{tf_label}_mc_eff"]           = combined[f"{tf_label}_open_mc"].astype("int8")
    combined[f"{tf_label}_strength_eff"]     = combined[f"{tf_label}_open_strength"]
    combined[f"{tf_label}_wick_to_body_eff"] = combined[f"{tf_label}_open_wick_to_body"]
    combined[f"{tf_label}_range_eff"]        = combined[f"{tf_label}_open_range"]
    combined[f"{tf_label}_anchor_high_eff"]  = combined[f"{tf_label}_open_anchor_high"]
    combined[f"{tf_label}_anchor_low_eff"]   = combined[f"{tf_label}_open_anchor_low"]

    # Purge logic
    hi = pd.to_numeric(combined["high"], errors="coerce").to_numpy()
    lo = pd.to_numeric(combined["low"], errors="coerce").to_numpy()
    mc_state = combined[f"{tf_label}_open_mc"].to_numpy()
    ah = combined[f"{tf_label}_open_anchor_high"].to_numpy()
    al = combined[f"{tf_label}_open_anchor_low"].to_numpy()

    n = len(combined)
    i = 0
    while i < n:
        state = mc_state[i]
        if state == 0:
            i += 1
            continue

        # active bullish
        if state == 1 and not np.isnan(ah[i]) and hi[i] > ah[i]:
            j = i
            while j < n and mc_state[j] == 1:
                combined.loc[j, [
                    f"{tf_label}_mc_eff",
                    f"{tf_label}_strength_eff",
                    f"{tf_label}_wick_to_body_eff",
                    f"{tf_label}_range_eff",
                    f"{tf_label}_anchor_high_eff",
                    f"{tf_label}_anchor_low_eff",
                ]] = 0
                j += 1
            i = j
            continue

        # active bearish
        if state == -1 and not np.isnan(al[i]) and lo[i] < al[i]:
            j = i
            while j < n and mc_state[j] == -1:
                combined.loc[j, [
                    f"{tf_label}_mc_eff",
                    f"{tf_label}_strength_eff",
                    f"{tf_label}_wick_to_body_eff",
                    f"{tf_label}_range_eff",
                    f"{tf_label}_anchor_high_eff",
                    f"{tf_label}_anchor_low_eff",
                ]] = 0
                j += 1
            i = j
            continue

        i += 1

    # Write per-TF combined file
    out_path = os.path.join(out_dir, f"combined_1m_{tf_label}_openstate_eff.csv")
    combined.to_csv(out_path, index=False)
    print(f"[OK] Wrote {tf_label}: {out_path}")

    return combined

# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load 1m base
    base = pd.read_csv(os.path.join(INPUT_DIR, FILE_1M))
    base = _ensure_dt64(base, "datetime").sort_values("datetime").reset_index(drop=True)

    needed_base = {"datetime", "open", "high", "low", "close", "volume"}
    missing = needed_base - set(base.columns)
    if missing:
        print(f"[WARN] Base file missing columns (continuing): {missing}")

    eff_slices: List[pd.DataFrame] = []
    for tf_label, tf_file in TF_FILES.items():
        tf_path = os.path.join(INPUT_DIR, tf_file)
        combined_tf = process_one_tf(base, tf_path, tf_label, OUTPUT_DIR)

        eff_slice = combined_tf[[
            "datetime",
            f"{tf_label}_mc_eff",
            f"{tf_label}_strength_eff",
            f"{tf_label}_wick_to_body_eff",
            f"{tf_label}_range_eff",
        ]].copy()
        eff_slices.append(eff_slice)

    # Build final combined file
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
    print(f"[OK] Wrote final MTF MC eff-only file: {FINAL_COMBINED}")

if __name__ == "__main__":
    main()
