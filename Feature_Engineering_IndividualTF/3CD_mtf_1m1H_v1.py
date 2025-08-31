#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine 1H 3CD features with 1m base data.
No early purge detection — 1H features persist until the end of their 1H bar.
"""

import os
import pandas as pd

# =========================
# CONFIG — edit paths
# =========================


script_dir = os.path.dirname(__file__)
INPUT_DIR = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Feature_SandPit','3CD'))
OUTPUT_DIR = INPUT_DIR

FILE_1M = "Data_1m_with_3cd.csv"
FILE_1H = "resampled_1H_with_3cd.csv"

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_1m_1H.csv")

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
    # Load 1m
    base = pd.read_csv(os.path.join(INPUT_DIR, FILE_1M))
    base = _ensure_dt64(base, "datetime")
    base = base.sort_values("datetime").reset_index(drop=True)

    # Load 1H
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

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Wrote combined file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
