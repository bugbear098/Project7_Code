#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTF 3CD Master script with efficient purge span handling (block zeroing) and progress bars.

- Enriches each TF file with 3CD features.
- Aligns higher TF open-state onto 1m base.
- Applies strict purge logic: zero eff features only within the purged draw span.
- Writes per-TF outputs and final combined eff-only file.

Datetime preserved in America/New_York, e.g. "2023-12-10 18:00:00-05:00".
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List
from tqdm import tqdm

# =========================
# CONFIG — edit paths
# =========================
INPUT_DIR  = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Data_Medium_Resampled"
OUTPUT_DIR = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Feature_SandPit/3CD"

TF_FILES: Dict[str, str] = {
    "1m":  "resampled_1m.csv",
    "5m":  "resampled_5m.csv",
    "15m": "resampled_15m.csv",
    "1H":  "resampled_1H.csv",
    "4H":  "resampled_4H.csv",
    "1D":  "resampled_1D.csv",
}

FINAL_COMBINED = os.path.join(OUTPUT_DIR, "mtf_eff_combined_base1m.csv")

EPS = 1e-12

# =========================
# Helpers
# =========================
def _ensure_dt64(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Ensure datetime col is tz-aware America/New_York."""
    df[col] = pd.to_datetime(df[col], errors="raise", utc=True)
    df[col] = df[col].dt.tz_convert("America/New_York")
    return df


def compute_3cd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a TF dataframe with 3CD features."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"datetime", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required cols: {missing}")

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise", utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Init new cols
    df["third_candle_draw"] = 0
    df["third_candle_range"] = 0.0
    df["third_candle_body_to_range"] = 0.0
    df["anchor_high"] = 0.0
    df["anchor_low"] = 0.0
    df["draw_started_at"] = pd.Series([pd.NaT] * len(df), dtype="datetime64[ns]")
    df["draw_started_at"] = df["draw_started_at"].dt.tz_localize("America/New_York")

    # State vars
    draw_active = 0
    draw_high, draw_low = None, None
    persist_range, persist_ratio = 0.0, 0.0
    persist_anchor_high, persist_anchor_low = 0.0, 0.0
    persist_started_at = pd.NaT

    def is_swing_high(i):
        return (df["high"].iloc[i - 1] > df["high"].iloc[i - 2]) and (df["high"].iloc[i - 1] > df["high"].iloc[i])

    def is_swing_low(i):
        return (df["low"].iloc[i - 1] < df["low"].iloc[i - 2]) and (df["low"].iloc[i - 1] < df["low"].iloc[i])

    for i in range(2, len(df)):
        swing_high = is_swing_high(i)
        swing_low = is_swing_low(i)

        if swing_high and not swing_low:
            draw_active = -1
            draw_high = float(df["high"].iloc[i])
            draw_low = float(df["low"].iloc[i])
            persist_range = draw_high - draw_low
            body_i = abs(float(df["open"].iloc[i] - df["close"].iloc[i]))
            persist_ratio = body_i / max(persist_range, EPS)
            persist_anchor_high, persist_anchor_low = draw_high, draw_low
            persist_started_at = df["datetime"].iloc[i]

        elif swing_low and not swing_high:
            draw_active = 1
            draw_high = float(df["high"].iloc[i])
            draw_low = float(df["low"].iloc[i])
            persist_range = draw_high - draw_low
            body_i = abs(float(df["open"].iloc[i] - df["close"].iloc[i]))
            persist_ratio = body_i / max(persist_range, EPS)
            persist_anchor_high, persist_anchor_low = draw_high, draw_low
            persist_started_at = df["datetime"].iloc[i]

        # purge logic inside same TF
        if draw_active == 1:  # bullish
            if (df["high"].iloc[i] > draw_high) or (df["close"].iloc[i] < draw_low):
                draw_active = 0
                persist_range = persist_ratio = 0.0
                persist_anchor_high = persist_anchor_low = 0.0
                persist_started_at = pd.NaT
        elif draw_active == -1:  # bearish
            if (df["low"].iloc[i] < draw_low) or (df["close"].iloc[i] > draw_high):
                draw_active = 0
                persist_range = persist_ratio = 0.0
                persist_anchor_high = persist_anchor_low = 0.0
                persist_started_at = pd.NaT

        df.at[i, "third_candle_draw"] = draw_active
        df.at[i, "third_candle_range"] = persist_range if draw_active != 0 else 0.0
        df.at[i, "third_candle_body_to_range"] = persist_ratio if draw_active != 0 else 0.0
        df.at[i, "anchor_high"] = persist_anchor_high if draw_active != 0 else 0.0
        df.at[i, "anchor_low"] = persist_anchor_low if draw_active != 0 else 0.0
        df.at[i, "draw_started_at"] = persist_started_at if draw_active != 0 else pd.NaT

    return df


def build_open_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["draw_open"] = out["third_candle_draw"].shift(1)
    out["anchor_high_open"] = out["anchor_high"].shift(1)
    out["anchor_low_open"] = out["anchor_low"].shift(1)
    out["range_open"] = out["third_candle_range"].shift(1)
    out["ratio_open"] = out["third_candle_body_to_range"].shift(1)
    out["draw_id_open"] = (
        pd.to_datetime(out["draw_started_at"].shift(1), errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
    )

    out["draw_open"] = pd.to_numeric(out["draw_open"], errors="coerce").fillna(0).astype("int8")
    out["range_open"] = pd.to_numeric(out["range_open"], errors="coerce").fillna(0.0)
    out["ratio_open"] = pd.to_numeric(out["ratio_open"], errors="coerce").fillna(0.0)

    return out[
        ["datetime", "draw_open", "anchor_high_open", "anchor_low_open", "range_open", "ratio_open", "draw_id_open"]
    ]


def process_one_tf(base: pd.DataFrame, tf_path: str, tf_label: str, out_dir: str) -> pd.DataFrame:
    tf = pd.read_csv(tf_path)
    tf = _ensure_dt64(tf, "datetime").sort_values("datetime").reset_index(drop=True)
    tf_enriched = compute_3cd_features(tf)
    tf_open = build_open_state(tf_enriched)

    # align onto 1m
    combined = pd.merge_asof(
        base.sort_values("datetime"),
        tf_open.sort_values("datetime"),
        on="datetime",
        direction="backward",
    )

    # rename cols
    combined.rename(
        columns={
            "draw_open": f"{tf_label}_open_draw",
            "anchor_high_open": f"{tf_label}_open_anchor_high",
            "anchor_low_open": f"{tf_label}_open_anchor_low",
            "range_open": f"{tf_label}_open_range",
            "ratio_open": f"{tf_label}_open_ratio",
            "draw_id_open": f"{tf_label}_open_draw_id",
        },
        inplace=True,
    )

    # effective copies
    combined[f"{tf_label}_third_candle_draw_eff"] = combined[f"{tf_label}_open_draw"].astype("int8")
    combined[f"{tf_label}_third_candle_range_eff"] = combined[f"{tf_label}_open_range"]
    combined[f"{tf_label}_third_candle_body_to_range_eff"] = combined[f"{tf_label}_open_ratio"]

    # purge logic (block zeroing)
    sign = combined[f"{tf_label}_open_draw"].to_numpy()
    ah = combined[f"{tf_label}_open_anchor_high"].to_numpy()
    al = combined[f"{tf_label}_open_anchor_low"].to_numpy()
    hi = combined["high"].to_numpy()
    lo = combined["low"].to_numpy()

    cols_to_zero = [
        f"{tf_label}_third_candle_draw_eff",
        f"{tf_label}_third_candle_range_eff",
        f"{tf_label}_third_candle_body_to_range_eff",
    ]

    n = len(combined)
    i = 0
    for _ in tqdm(range(n), desc=f"Purging {tf_label}"):
        if i >= n:
            break
        s = sign[i]
        if s == 0 or np.isnan(s):
            i += 1
            continue

        draw_sign = int(s)
        j = i
        purged = False

        while j < n and sign[j] == draw_sign:
            if draw_sign == 1 and hi[j] > ah[j]:  # bullish purge
                purged = True
                break
            if draw_sign == -1 and lo[j] < al[j]:  # bearish purge
                purged = True
                break
            j += 1

        if purged:
            k = j
            while k < n and sign[k] == draw_sign:
                k += 1
            combined.loc[j:k, cols_to_zero] = 0
            i = k
        else:
            i = j

    # save per-TF
    out_path = os.path.join(out_dir, f"{tf_label}_with_3cd.csv")
    combined.to_csv(out_path, index=False)
    print(f"[OK] {tf_label} saved → {out_path}")
    return combined


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base = pd.read_csv(os.path.join(INPUT_DIR, TF_FILES["1m"]))
    base = _ensure_dt64(base, "datetime").sort_values("datetime").reset_index(drop=True)

    eff_slices: List[pd.DataFrame] = []
    for tf_label, fname in TF_FILES.items():
        if tf_label == "1m":
            continue
        tf_path = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(tf_path):
            print(f"[WARN] Missing {tf_label}")
            continue
        combined_tf = process_one_tf(base, tf_path, tf_label, OUTPUT_DIR)
        eff_slice = combined_tf[
            [
                "datetime",
                f"{tf_label}_third_candle_draw_eff",
                f"{tf_label}_third_candle_range_eff",
                f"{tf_label}_third_candle_body_to_range_eff",
            ]
        ].copy()
        eff_slices.append(eff_slice)

    final = base.copy().sort_values("datetime")
    for sl in eff_slices:
        final = pd.merge_asof(final, sl.sort_values("datetime"), on="datetime", direction="backward")

    final.to_csv(FINAL_COMBINED, index=False)
    print(f"[OK] Wrote final combined file: {FINAL_COMBINED}")


if __name__ == "__main__":
    main()
