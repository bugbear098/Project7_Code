#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTF Momentum Candle (MC) Master Script

Steps:
1. Load raw resampled OHLCV files (no features).
2. Compute MC features (momentum_candle_active, momentum_strength, wick_to_body, total_range).
3. Build TF open-state (shift MC features back 1 bar).
4. Align TF → 1m base with merge_asof.
5. Apply purge detection:
   - Bull purge if 1m.high > anchor_high
   - Bear purge if 1m.low < anchor_low
   -> Zero *_eff values from purge minute until TF state ends.
6. Save per-TF CSV and final combined eff-only CSV.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List

# =========================
# CONFIG
# =========================
INPUT_DIR  = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Data_Medium_Resampled"
OUTPUT_DIR = os.path.join(INPUT_DIR, "Processed_MC")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_1M = "resampled_1m.csv"
TF_FILES: Dict[str, str] = {
    "5m":  "resampled_5m.csv",
    "15m": "resampled_15m.csv",
    "1H":  "resampled_1H.csv",
    "4H":  "resampled_4H.csv",
    "1D":  "resampled_1D.csv",
}

FINAL_COMBINED = os.path.join(OUTPUT_DIR, "mtf_mc_combined_base1m.csv")

# =========================
# Helpers
# =========================
def _ensure_dt64(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Ensure datetime column parsed and converted to NY time."""
    df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert("America/New_York")
    return df

def compute_mc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Faithful reimplementation of original MC logic:
    - Fresh MC when close > prev_high (bull) or close < prev_low (bear)
    - Persist until invalidation
    - Reset on break
    """

    momentum_active = np.zeros(len(df), dtype="int8")
    momentum_strength = np.zeros(len(df), dtype="float32")
    wick_to_body = np.zeros(len(df), dtype="float32")
    total_range = np.zeros(len(df), dtype="float32")

    active_state = 0
    ref_high, ref_low = None, None
    last_strength = last_wtb = last_range = 0.0

    for i in range(1, len(df)):
        prev_close = df["close"].iloc[i-1]
        prev_high  = df["high"].iloc[i-1]
        prev_low   = df["low"].iloc[i-1]

        open_ = df["open"].iloc[i]
        close = df["close"].iloc[i]
        high  = df["high"].iloc[i]
        low   = df["low"].iloc[i]

        # --- Fresh MC ---
        if close > prev_high:
            momentum_active[i] = 1
            active_state = 1
            ref_high, ref_low = high, low
            last_strength = close - prev_high
            last_wtb = (high - low) / abs(close - open_) if close != open_ else 0.0
            last_range = high - low

        elif close < prev_low:
            momentum_active[i] = -1
            active_state = -1
            ref_high, ref_low = high, low
            last_strength = prev_low - close
            last_wtb = (high - low) / abs(close - open_) if close != open_ else 0.0
            last_range = high - low

        # --- Continuation / Reset ---
        elif active_state == 1:  # bullish active
            if high > ref_high or close < ref_low:
                momentum_active[i] = 0
                active_state = 0
                ref_high = ref_low = None
                last_strength = last_wtb = last_range = 0.0
            else:
                momentum_active[i] = 1

        elif active_state == -1:  # bearish active
            if low < ref_low or close > ref_high:
                momentum_active[i] = 0
                active_state = 0
                ref_high = ref_low = None
                last_strength = last_wtb = last_range = 0.0
            else:
                momentum_active[i] = -1

        # Assign persisted features
        momentum_strength[i] = last_strength if active_state != 0 else 0.0
        wick_to_body[i] = last_wtb if active_state != 0 else 0.0
        total_range[i] = last_range if active_state != 0 else 0.0

    df["momentum_candle_active"] = momentum_active
    df["momentum_strength"] = momentum_strength
    df["wick_to_body"] = wick_to_body
    df["total_range"] = total_range

    return df

def build_open_state(df: pd.DataFrame) -> pd.DataFrame:
    """Shift MC features back 1 bar to create open-state."""
    out = df.copy()
    out["mc_open"]           = out["momentum_candle_active"].shift(1).fillna(0).astype("int8")
    out["strength_open"]     = out["momentum_strength"].shift(1).fillna(0.0)
    out["wick_to_body_open"] = out["wick_to_body"].shift(1).fillna(0.0)
    out["range_open"]        = out["total_range"].shift(1).fillna(0.0)
    out["anchor_high_open"]  = out["high"].shift(1)
    out["anchor_low_open"]   = out["low"].shift(1)

    return out[[
        "datetime",
        "mc_open", "strength_open", "wick_to_body_open", "range_open",
        "anchor_high_open", "anchor_low_open"
    ]]

def process_one_tf(base: pd.DataFrame, tf_path: str, tf_label: str, out_dir: str) -> pd.DataFrame:
    """Generate MC features, align to 1m base, apply purge logic."""
    if not os.path.exists(tf_path):
        raise FileNotFoundError(f"{tf_label} file not found: {tf_path}")

    tf = pd.read_csv(tf_path)
    tf.columns = ["datetime", "open", "high", "low", "close", "volume"]
    tf = _ensure_dt64(tf, "datetime").sort_values("datetime").reset_index(drop=True)

    tf = compute_mc_features(tf)
    tf_open = build_open_state(tf)

    combined = pd.merge_asof(
        base.sort_values("datetime"),
        tf_open.sort_values("datetime"),
        on="datetime",
        direction="backward"
    )

    # Rename columns
    combined.rename(columns={
        "mc_open":           f"{tf_label}_open_mc",
        "strength_open":     f"{tf_label}_open_strength",
        "wick_to_body_open": f"{tf_label}_open_wick_to_body",
        "range_open":        f"{tf_label}_open_range",
        "anchor_high_open":  f"{tf_label}_open_anchor_high",
        "anchor_low_open":   f"{tf_label}_open_anchor_low",
    }, inplace=True)

    # Effective copies
    combined[f"{tf_label}_mc_eff"]           = combined[f"{tf_label}_open_mc"].fillna(0).astype("int8")
    combined[f"{tf_label}_strength_eff"]     = combined[f"{tf_label}_open_strength"].fillna(0.0)
    combined[f"{tf_label}_wick_to_body_eff"] = combined[f"{tf_label}_open_wick_to_body"].fillna(0.0)
    combined[f"{tf_label}_range_eff"]        = combined[f"{tf_label}_open_range"].fillna(0.0)
    combined[f"{tf_label}_anchor_high_eff"]  = combined[f"{tf_label}_open_anchor_high"]
    combined[f"{tf_label}_anchor_low_eff"]   = combined[f"{tf_label}_open_anchor_low"]

    # --- Purge logic ---
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

        if state == 1 and not np.isnan(ah[i]) and hi[i] > ah[i]:  # bullish purge
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

        if state == -1 and not np.isnan(al[i]) and lo[i] < al[i]:  # bearish purge
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

    # Save per-TF file
    out_path = os.path.join(out_dir, f"combined_1m_{tf_label}_openstate_eff.csv")
    combined.to_csv(out_path, index=False)
    print(f"[OK] Wrote {tf_label}: {out_path}")

    return combined

# =========================
# Main
# =========================
def main():
    # Load 1m base
    base_path = os.path.join(INPUT_DIR, FILE_1M)
    base = pd.read_csv(base_path)
    base.columns = ["datetime", "open", "high", "low", "close", "volume"]
    base = _ensure_dt64(base, "datetime").sort_values("datetime").reset_index(drop=True)

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

    # Merge all eff slices into final
    final = base.copy().sort_values("datetime")
    for sl in eff_slices:
        final = pd.merge_asof(
            final.sort_values("datetime"),
            sl.sort_values("datetime"),
            on="datetime",
            direction="backward"
        )

    final.to_csv(FINAL_COMBINED, index=False)
    print(f"[OK] Wrote final MTF MC eff-only file: {FINAL_COMBINED}")

if __name__ == "__main__":
    main()
