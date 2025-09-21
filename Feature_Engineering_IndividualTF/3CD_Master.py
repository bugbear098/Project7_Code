#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Master 3CD pipeline:
  1. Compute 3CD features on each resampled TF file.
  2. Align higher-TF 3CD open-state onto 1m base.
  3. Apply strict purge logic with ID awareness.
  4. Save per-TF and final combined eff features.

Datetime: always tz-aware, America/New_York (e.g. 2023-12-10 18:00:00-05:00).
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Set, Dict, Tuple, List

# =========================
# CONFIG
# =========================
script_dir = os.path.dirname(__file__)
INPUT_DIR  = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Data_Medium_Resampled'))
OUTPUT_DIR = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Feature_SandPit', '3CD', 'Processed'))

TF_FILES = {
    "1m":  "Data_1m.csv",
    "5m":  "resampled_5m.csv",
    "15m": "resampled_15m.csv",
    "1H":  "resampled_1H.csv",
    "4H":  "resampled_4H.csv",
    "1D":  "resampled_1D.csv",
}

FINAL_COMBINED = os.path.join(OUTPUT_DIR, "mtf_3cd_eff_combined_base1m.csv")

EPS = 1e-12

# =========================
# Step 1: Compute 3CD per TF
# =========================
def compute_3cd_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    required = {'datetime','open','high','low','close'}
    if missing := (required - set(df.columns)):
        raise ValueError(f"Missing cols: {missing}")

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values('datetime').reset_index(drop=True)

    df['third_candle_draw'] = 0
    df['third_candle_range'] = 0.0
    df['third_candle_body_to_range'] = 0.0
    df['anchor_high'] = 0.0
    df['anchor_low'] = 0.0
    df['draw_started_at'] = pd.NaT

    draw_active = 0
    draw_high = draw_low = None
    persist_range = persist_ratio = 0.0
    persist_anchor_high = persist_anchor_low = 0.0
    persist_started_at = pd.NaT

    def is_swing_high(i):
        return (df['high'].iloc[i-1] > df['high'].iloc[i-2]) and (df['high'].iloc[i-1] > df['high'].iloc[i])
    def is_swing_low(i):
        return (df['low'].iloc[i-1] < df['low'].iloc[i-2]) and (df['low'].iloc[i-1] < df['low'].iloc[i])

    for i in range(2, len(df)):
        if is_swing_high(i):
            draw_active = -1
            draw_high, draw_low = float(df['high'].iloc[i]), float(df['low'].iloc[i])
            persist_range = draw_high - draw_low
            body_i = abs(float(df['open'].iloc[i]-df['close'].iloc[i]))
            persist_ratio = body_i / max(persist_range, EPS)
            persist_anchor_high, persist_anchor_low = draw_high, draw_low
            persist_started_at = df['datetime'].iloc[i]

        elif is_swing_low(i):
            draw_active = 1
            draw_high, draw_low = float(df['high'].iloc[i]), float(df['low'].iloc[i])
            persist_range = draw_high - draw_low
            body_i = abs(float(df['open'].iloc[i]-df['close'].iloc[i]))
            persist_ratio = body_i / max(persist_range, EPS)
            persist_anchor_high, persist_anchor_low = draw_high, draw_low
            persist_started_at = df['datetime'].iloc[i]

        # Purge/invalid checks
        if draw_active == 1 and ((df['high'].iloc[i] > draw_high) or (df['close'].iloc[i] < draw_low)):
            draw_active = 0; persist_range=0; persist_ratio=0; persist_anchor_high=0; persist_anchor_low=0; persist_started_at=pd.NaT
        elif draw_active == -1 and ((df['low'].iloc[i] < draw_low) or (df['close'].iloc[i] > draw_high)):
            draw_active = 0; persist_range=0; persist_ratio=0; persist_anchor_high=0; persist_anchor_low=0; persist_started_at=pd.NaT

        df.at[i,'third_candle_draw'] = draw_active
        df.at[i,'third_candle_range'] = persist_range if draw_active else 0.0
        df.at[i,'third_candle_body_to_range'] = persist_ratio if draw_active else 0.0
        df.at[i,'anchor_high'] = persist_anchor_high if draw_active else 0.0
        df.at[i,'anchor_low']  = persist_anchor_low if draw_active else 0.0
        df.at[i,'draw_started_at'] = persist_started_at if draw_active else pd.NaT
    return df

# =========================
# Step 2: Build open-state + eff features
# =========================
def build_open_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["draw_open"]        = out["third_candle_draw"].shift(1)
    out["anchor_high_open"] = out["anchor_high"].shift(1)
    out["anchor_low_open"]  = out["anchor_low"].shift(1)
    out["range_open"]       = out["third_candle_range"].shift(1)
    out["ratio_open"]       = out["third_candle_body_to_range"].shift(1)
    out["draw_id_open"]     = pd.to_datetime(out["draw_started_at"].shift(1), errors="coerce")
    return out[["datetime","draw_open","anchor_high_open","anchor_low_open","range_open","ratio_open","draw_id_open"]]

def id_key(ts: Optional[pd.Timestamp]) -> Optional[int]:
    return None if ts is None or pd.isna(ts) else int(pd.Timestamp(ts).value)

def process_one_tf(base: pd.DataFrame, tf: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    tf_open = build_open_state(tf).sort_values("datetime").reset_index(drop=True)
    combined = pd.merge_asof(base.sort_values("datetime"), tf_open.sort_values("datetime"),
                             on="datetime", direction="backward")

    # Rename
    combined.rename(columns={
        "draw_open":f"{tf_label}_open_draw",
        "anchor_high_open":f"{tf_label}_open_anchor_high",
        "anchor_low_open":f"{tf_label}_open_anchor_low",
        "range_open":f"{tf_label}_open_range",
        "ratio_open":f"{tf_label}_open_ratio",
        "draw_id_open":f"{tf_label}_open_draw_id",
    }, inplace=True)

    # Effective copies
    combined[f"{tf_label}_third_candle_draw_eff"]          = combined[f"{tf_label}_open_draw"].fillna(0).astype("int8")
    combined[f"{tf_label}_third_candle_range_eff"]         = combined[f"{tf_label}_open_range"].fillna(0.0)
    combined[f"{tf_label}_third_candle_body_to_range_eff"] = combined[f"{tf_label}_open_ratio"].fillna(0.0)
    combined[f"{tf_label}_anchor_high_eff"]                = combined[f"{tf_label}_open_anchor_high"].fillna(0.0)
    combined[f"{tf_label}_anchor_low_eff"]                 = combined[f"{tf_label}_open_anchor_low"].fillna(0.0)
    combined[f"{tf_label}_draw_started_at_eff"]            = combined[f"{tf_label}_open_draw_id"]

    # Purge zeroing (simplified strict version)
    hi, lo = base["high"].to_numpy(), base["low"].to_numpy()
    sign_series = combined[f"{tf_label}_open_draw"].to_numpy()
    id_series   = pd.to_datetime(combined[f"{tf_label}_open_draw_id"], errors="coerce")

    in_draw=False; current_id=None; current_sign=0; ah=al=np.nan
    for i in range(len(combined)):
        raw_sign = int(sign_series[i]) if not np.isnan(sign_series[i]) else 0
        raw_id = id_series.iloc[i]; raw_idk = id_key(raw_id)
        if not in_draw and raw_sign!=0 and raw_idk is not None:
            in_draw=True; current_id=raw_id; current_sign=raw_sign
            ah=combined[f"{tf_label}_open_anchor_high"].iloc[i]; al=combined[f"{tf_label}_open_anchor_low"].iloc[i]
        elif in_draw:
            curk=id_key(current_id)
            if raw_sign==0 or raw_idk!=curk:
                in_draw=False; continue
            if (current_sign==1 and hi[i]>ah) or (current_sign==-1 and lo[i]<al):
                # zero until draw changes
                j=i
                while j<len(combined) and id_key(id_series.iloc[j])==curk:
                    combined.loc[j,[f"{tf_label}_third_candle_draw_eff",
                                    f"{tf_label}_third_candle_range_eff",
                                    f"{tf_label}_third_candle_body_to_range_eff",
                                    f"{tf_label}_anchor_high_eff",
                                    f"{tf_label}_anchor_low_eff",
                                    f"{tf_label}_draw_started_at_eff"]] = [0,0.0,0.0,0.0,0.0,pd.NaT]
                    j+=1
                in_draw=False
    return combined

# =========================
# Master main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    enriched_files = {}
    # Step 1: compute 3CD features for each TF
    for tf_label, fname in TF_FILES.items():
        in_path = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(in_path):
            print(f"[WARN] Missing file {in_path}"); continue
        df = pd.read_csv(in_path)
        df3 = compute_3cd_features(df)
        out_path = os.path.join(OUTPUT_DIR, fname.replace(".csv","_with_3cd.csv"))
        df3.to_csv(out_path, index=False)
        enriched_files[tf_label]=out_path
        print(f"[OK] {tf_label} 3CD enriched → {out_path}")

    # Step 2: align to 1m and purge eff
    base = pd.read_csv(enriched_files["1m"])
    base['datetime'] = pd.to_datetime(base['datetime'], utc=True).dt.tz_convert("America/New_York")
    base = base.sort_values("datetime").reset_index(drop=True)

    eff_slices=[]
    for tf_label,fpath in enriched_files.items():
        if tf_label=="1m": continue
        tf=pd.read_csv(fpath)
        tf['datetime']=pd.to_datetime(tf['datetime'], utc=True).dt.tz_convert("America/New_York")
        tf=tf.sort_values("datetime").reset_index(drop=True)
        combined=process_one_tf(base,tf,tf_label)
        out_tf=os.path.join(OUTPUT_DIR,f"combined_1m_{tf_label}_eff.csv")
        combined.to_csv(out_tf,index=False)
        eff_slice=combined[["datetime",
            f"{tf_label}_third_candle_draw_eff",
            f"{tf_label}_third_candle_range_eff",
            f"{tf_label}_third_candle_body_to_range_eff"]].copy()
        eff_slices.append(eff_slice)

    # Final combined
    final=base[["datetime"]].copy()
    for sl in eff_slices:
        final=pd.merge_asof(final.sort_values("datetime"), sl.sort_values("datetime"),
                            on="datetime", direction="backward")
    final.to_csv(FINAL_COMBINED,index=False)
    print(f"[OK] Final combined written: {FINAL_COMBINED}")

if __name__=="__main__":
    main()
