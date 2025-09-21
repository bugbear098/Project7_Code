#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Liquidity purge feature generator (multi-timeframe, base 1m).

Steps:
  1. Load resampled_1m.csv as base clock.
  2. For each higher-TF resampled file (5m, 15m, 1H, 4H, 1D, etc.):
       - Detect purge events (swing high/low sweeps).
       - Map purge events to exact 1m minute where they occurred.
       - Build 1m-aligned features:
           * purge_direction_purges_eff_{3,5,10}
           * purge_timespan_eff_{3,5,10}
           * time_since_purge_eff
       - Write per-TF eff CSV (includes purge_intensity_eff for debug).
  3. Merge into final combined CSV (eff-only, no purge_intensity).
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# =========================
# CONFIG
# =========================
INPUT_DIR  = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Data_Medium_Resampled"
OUTPUT_DIR = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Feature_SandPit/Liq"

BASE_1M_FILE = "resampled_1m.csv"

DECAY = 0.9        # exponential decay factor
EVENT_WINDOWS = [3, 5, 10]  # event counts for direction + timespan features

# =========================
# Helpers
# =========================
def _ensure_dt_tz(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Parse datetime with UTC then convert to America/New_York (DST-preserving)."""
    if col not in df.columns:
        if "DateTime" in df.columns:
            col = "DateTime"
        else:
            raise ValueError("No datetime column found")
    df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert("America/New_York")
    if col != "datetime":
        df.rename(columns={col: "datetime"}, inplace=True)
    return df

def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"open","high","low","close","volume"} and c != lc:
            ren[c] = lc
    if ren:
        df = df.rename(columns=ren)
    needed = {"datetime","open","high","low","close","volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[["datetime","open","high","low","close","volume"]].copy()

def detect_tf_purges(tf: pd.DataFrame) -> List[dict]:
    """Detect purge events at TF level using 3-bar swings, return event dicts with swing level + TF candle window."""
    df = tf.reset_index(drop=True)
    df["swing_high"] = (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
    df["swing_low"]  = (df["low"]  < df["low"].shift(1))  & (df["low"]  < df["low"].shift(-1))

    swing_highs, swing_lows = [], []
    events = []

    n = len(df)
    for i in range(n):
        hi, lo = df.at[i,"high"], df.at[i,"low"]

        if df.at[i,"swing_high"]:
            swing_highs.append((i, hi))
        if df.at[i,"swing_low"]:
            swing_lows.append((i, lo))

        new_highs = []
        for idx, level in swing_highs:
            if hi > level:
                events.append({"tf_idx": i, "tf_dt": df.at[i,"datetime"], "sign": -1, "level": level})
            else:
                new_highs.append((idx, level))
        swing_highs = new_highs

        new_lows = []
        for idx, level in swing_lows:
            if lo < level:
                events.append({"tf_idx": i, "tf_dt": df.at[i,"datetime"], "sign": +1, "level": level})
            else:
                new_lows.append((idx, level))
        swing_lows = new_lows

    # attach candle windows
    for k, ev in enumerate(events):
        i = ev["tf_idx"]
        start = df.at[i,"datetime"]
        if i+1 < n:
            end = df.at[i+1,"datetime"]
        else:
            end = pd.Timestamp.max.tz_localize("UTC").tz_convert("America/New_York")
        events[k]["window_start"] = start
        events[k]["window_end"] = end
    return events

def map_events_to_1m(events: List[dict], base_1m: pd.DataFrame) -> List[Tuple[int,int]]:
    """Map TF purge events to first 1m bar index where breach actually occurred."""
    m1 = base_1m.reset_index(drop=True)
    dt = m1["datetime"]

    mapped = []
    for ev in events:
        ws, we, level, sign = ev["window_start"], ev["window_end"], ev["level"], ev["sign"]
        mask = (dt >= ws) & (dt < we)
        if not mask.any():
            continue
        seg = m1.loc[mask, ["high","low"]]
        seg_idx = np.flatnonzero(mask.values)
        found_idx = None
        if sign == -1:
            hit = np.flatnonzero(seg["high"].values > level)
            if hit.size: found_idx = seg_idx[int(hit[0])]
        else:
            hit = np.flatnonzero(seg["low"].values < level)
            if hit.size: found_idx = seg_idx[int(hit[0])]
        if found_idx is not None:
            mapped.append((int(found_idx), int(sign)))
    mapped.sort(key=lambda t: t[0])
    return mapped

def build_eff_series_from_events(base_1m: pd.DataFrame,
                                 mapped_events: List[Tuple[int,int]],
                                 tf_label: str) -> pd.DataFrame:
    """Build 1m-aligned eff features for this TF."""
    df = base_1m[["datetime"]].copy().reset_index(drop=True)
    n = len(df)

    intensity = np.zeros(n, dtype=np.int16)  # kept in per-TF output only
    dir_series = {k: np.zeros(n, dtype=np.float32) for k in EVENT_WINDOWS}
    timespan   = {k: np.full(n, np.nan, dtype=np.float32) for k in EVENT_WINDOWS}
    time_since = np.full(n, np.nan, dtype=np.float32)

    event_indices, event_signs = [], []
    by_min: Dict[int,List[int]] = {}
    for idx, s in mapped_events:
        by_min.setdefault(idx, []).append(s)

    last_purge_idx = None
    for i in range(n):
        if i in by_min:
            for s in by_min[i]:
                event_indices.append(i)
                event_signs.append(s)
            last_purge_idx = i

        # purge intensity
        intensity[i] = len(by_min.get(i, []))

        # direction over multiple event windows
        for k in EVENT_WINDOWS:
            recent = event_signs[-k:]
            val = 0.0
            for s in recent:
                val = val * DECAY + float(s)
            dir_series[k][i] = val

        # timespans over multiple event windows
        for k in EVENT_WINDOWS:
            if len(event_indices) >= 2:
                last_k = event_indices[-k:] if len(event_indices) >= k else event_indices
                span = last_k[-1] - last_k[0]
                timespan[k][i] = float(span)

        # time since last purge
        if last_purge_idx is not None:
            time_since[i] = float(i - last_purge_idx)

    out = df.copy()
    out[f"{tf_label}_purge_intensity_eff"] = intensity
    for k in EVENT_WINDOWS:
        out[f"{tf_label}_purge_direction_purges_eff_{k}"] = dir_series[k]
        out[f"{tf_label}_purge_timespan_eff_{k}"] = timespan[k]
    out[f"{tf_label}_time_since_purge_eff"] = time_since
    return out

# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # base 1m
    base_path = os.path.join(INPUT_DIR, BASE_1M_FILE)
    base = pd.read_csv(base_path)
    base = _ensure_dt_tz(base, "datetime")
    base = _standardize_ohlcv(base).sort_values("datetime").reset_index(drop=True)

    # higher TF files
    tf_files = []
    for fname in os.listdir(INPUT_DIR):
        if fname.startswith("resampled_") and fname.endswith(".csv") and fname != BASE_1M_FILE:
            tf_label = fname.replace("resampled_","").replace(".csv","")
            tf_files.append((tf_label, os.path.join(INPUT_DIR,fname)))

    eff_slices = []
    for tf_label, tf_path in sorted(tf_files):
        tf = pd.read_csv(tf_path)
        tf = _ensure_dt_tz(tf,"datetime")
        tf = _standardize_ohlcv(tf).sort_values("datetime").reset_index(drop=True)

        events = detect_tf_purges(tf)
        mapped = map_events_to_1m(events, base)
        eff = build_eff_series_from_events(base, mapped, tf_label)

        out_tf = os.path.join(OUTPUT_DIR, f"liq_eff_{tf_label}_base1m.csv")
        eff.to_csv(out_tf, index=False)
        print(f"[OK] Wrote {tf_label} eff → {out_tf}")

        # drop purge_intensity_eff for final combined
        eff_no_intensity = eff.drop(columns=[f"{tf_label}_purge_intensity_eff"])
        eff_slices.append(eff_no_intensity)

    # combine
    final = base[["datetime"]].copy()
    for sl in eff_slices:
        final = pd.merge_asof(final.sort_values("datetime"),
                              sl.sort_values("datetime"),
                              on="datetime",
                              direction="backward")
    final_out = os.path.join(OUTPUT_DIR, "liq_mtf_eff_combined_base1m.csv")
    final.to_csv(final_out, index=False)
    print(f"[OK] Wrote combined eff-only file → {final_out}")

if __name__ == "__main__":
    main()
