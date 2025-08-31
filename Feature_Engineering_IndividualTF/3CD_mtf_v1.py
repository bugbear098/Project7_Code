
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3CD_mtf_v3.py — MTF combiner (bar-start + backward merge + ID-aware early purge)

Goal:
- On a 1-minute grid, present higher-TF (HTF) 3CD features exactly as a live trader sees them:
  * HTF features are the **state at the open** of each HTF bar (no lookahead).
  * If the 1m price **purges** the HTF anchor intra-bar, flip the HTF feature to 0 **immediately**.
  * Do NOT early-invalidate; invalidation remains on the HTF close (naturally captured via next bar's open state = 0).

Key implementation notes:
- We do NOT shift HTF datetimes (we keep **bar-start timestamps**).
- Instead we build explicit **open-state columns** on each HTF file by shifting all relevant features **one bar backward**.
  This ensures the aligned value at time T truly reflects what was known at the HTF **bar open** T.
- We then align those open-state columns onto the 1m base via merge_asof(direction="backward").
"""

import os
import sys
import numpy as np
import pandas as pd

# =========================
# CONFIG — edit these
# =========================

script_dir = os.path.dirname(__file__)
INPUT_DIR = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Feature_SandPit','3CD'))
OUTPUT_DIR = INPUT_DIR

TF_FILES = {
    "1m":  "Data_1m_with_3cd.csv",
    "5m":  "resampled_5m_with_3cd.csv",
    "15m": "resampled_15m_with_3cd.csv",
    "1H":  "resampled_1H_with_3cd.csv",
    "4H":  "resampled_4H_with_3cd.csv",
    "1D":  "resampled_1D_with_3cd.csv",
}

COMBINED_CSV = os.path.join(OUTPUT_DIR, "mtf_3cd_combined_base1m.csv")

# =========================
# Helpers
# =========================
def _ensure_dt64_utc_naive(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Parse datetime with utc=True, then convert to UTC and drop tz (naive ns)."""
    df[col] = pd.to_datetime(df[col], errors="raise", utc=True)
    df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    return df

def load_tf(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = _ensure_dt64_utc_naive(df, "datetime")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def build_open_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    From HTF rows stamped at **bar-start**, build 'state at bar open' by shifting
    what you normally compute at bar close back by 1 bar. This prevents lookahead.

    Produces these columns (all representing state known at the bar open):
      draw_open         : third_candle_draw at bar open (-1/0/+1)
      anchor_high_open  : anchor_high at bar open
      anchor_low_open   : anchor_low at bar open
      range_open        : third_candle_range at bar open
      ratio_open        : third_candle_body_to_range at bar open
      draw_id_open      : draw_started_at at bar open (NaT if none)
    """
    need = {
        'third_candle_draw', 'anchor_high', 'anchor_low',
        'third_candle_range', 'third_candle_body_to_range', 'draw_started_at'
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"HTF file missing required columns: {miss}")

    out = df.copy()
    out['draw_open']        = out['third_candle_draw'].shift(1)
    out['anchor_high_open'] = out['anchor_high'].shift(1)
    out['anchor_low_open']  = out['anchor_low'].shift(1)
    out['range_open']       = out['third_candle_range'].shift(1)
    out['ratio_open']       = out['third_candle_body_to_range'].shift(1)
    out['draw_id_open']     = pd.to_datetime(out['draw_started_at'].shift(1), errors='coerce')

    # Fill NaNs sensibly for the open state:
    out['draw_open']  = pd.to_numeric(out['draw_open'], errors='coerce').fillna(0).astype('int8')
    # Others can remain NaN/0 — the state machine only uses anchors when active
    out['range_open'] = pd.to_numeric(out['range_open'], errors='coerce').fillna(0.0)
    out['ratio_open'] = pd.to_numeric(out['ratio_open'], errors='coerce').fillna(0.0)

    return out[['datetime', 'draw_open', 'anchor_high_open', 'anchor_low_open',
                'range_open', 'ratio_open', 'draw_id_open']]

# =========================
# ID-aware early purge on 1m
# =========================
def apply_early_purge_with_ids(
    base_df: pd.DataFrame,
    draw_col: str,
    id_col: str,
    a_high_col: str,
    a_low_col: str,
    range_col: str,
    ratio_col: str,
    out_prefix: str
) -> pd.DataFrame:
    """
    State machine on the 1m grid using **open-state** aligned higher-TF columns.
    - At most ONE early purge per draw (identified by draw_id_open).
    - Drops effective features to 0 immediately after purge.
    - No early invalidation (HTF close handles that via next bar's open state = 0).
    """
    n = len(base_df)

    draw_raw = pd.to_numeric(base_df[draw_col], errors='coerce').fillna(0).astype('int8').to_numpy()
    id_raw   = pd.to_datetime(base_df[id_col], errors='coerce')

    ah_raw   = base_df[a_high_col].to_numpy()
    al_raw   = base_df[a_low_col].to_numpy()
    rng_raw  = pd.to_numeric(base_df[range_col], errors='coerce').fillna(0.0).to_numpy()
    rat_raw  = pd.to_numeric(base_df[ratio_col], errors='coerce').fillna(0.0).to_numpy()

    highs = base_df['high'].to_numpy()
    lows  = base_df['low'].to_numpy()

    # Outputs
    active_eff = np.zeros(n, dtype='int8')
    purged     = np.zeros(n, dtype='int8')
    purge_time = pd.Series(pd.NaT, index=base_df.index, dtype='datetime64[ns]')

    ah_eff = np.full(n, np.nan)
    al_eff = np.full(n, np.nan)
    rng_eff = np.zeros(n, dtype='float64')
    rat_eff = np.zeros(n, dtype='float64')

    # State
    current_state: int = 0
    current_id: pd.Timestamp | None = None
    killed_id: pd.Timestamp | None = None
    latched_ah = np.nan
    latched_al = np.nan
    latched_rng = 0.0
    latched_rat = 0.0

    for i in range(n):
        raw_state = int(draw_raw[i])
        raw_id = id_raw.iloc[i]  # may be NaT

        # Activation only if currently inactive, open-state says active, and not the killed ID
        if current_state == 0 and raw_state != 0:
            if (killed_id is None) or (pd.isna(raw_id)) or (raw_id != killed_id):
                current_state = raw_state
                current_id = raw_id if not pd.isna(raw_id) else None
                latched_ah = ah_raw[i]
                latched_al = al_raw[i]
                latched_rng = rng_raw[i]
                latched_rat = rat_raw[i]

        # Early purge on 1m bar:
        if current_state == +1:
            if highs[i] > latched_ah:
                purged[i] = 1
                purge_time.iat[i] = base_df['datetime'].iloc[i]
                killed_id = current_id
                current_state = 0
                current_id = None
                latched_ah = np.nan
                latched_al = np.nan
                latched_rng = 0.0
                latched_rat = 0.0

        elif current_state == -1:
            if lows[i] < latched_al:
                purged[i] = 1
                purge_time.iat[i] = base_df['datetime'].iloc[i]
                killed_id = current_id
                current_state = 0
                current_id = None
                latched_ah = np.nan
                latched_al = np.nan
                latched_rng = 0.0
                latched_rat = 0.0

        # Write outputs
        active_eff[i] = current_state
        ah_eff[i] = latched_ah if current_state != 0 else np.nan
        al_eff[i] = latched_al if current_state != 0 else np.nan
        rng_eff[i] = latched_rng if current_state != 0 else 0.0
        rat_eff[i] = latched_rat if current_state != 0 else 0.0

    base_df[f'{out_prefix}_active_effective'] = active_eff
    base_df[f'{out_prefix}_purged_flag']      = purged
    base_df[f'{out_prefix}_purge_time']       = purge_time

    base_df[f'{out_prefix}_anchor_high_eff']  = ah_eff
    base_df[f'{out_prefix}_anchor_low_eff']   = al_eff
    base_df[f'{out_prefix}_third_candle_range_eff']          = rng_eff
    base_df[f'{out_prefix}_third_candle_body_to_range_eff']  = rat_eff

    return base_df

# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load base 1m
    base_path = os.path.join(INPUT_DIR, TF_FILES["1m"])
    if not os.path.exists(base_path):
        print(f"[ERR ] Base file not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    base = load_tf(base_path, label="1m")

    needed = {
        'datetime','open','high','low','close',
        'third_candle_draw','third_candle_range','third_candle_body_to_range',
        'anchor_high','anchor_low','draw_started_at'
    }
    miss = needed - set(base.columns)
    if miss:
        print(f"[ERR ] Base 1m file missing columns: {miss}", file=sys.stderr)
        sys.exit(1)

    # Clarify base column names
    base = base.rename(columns={
        'third_candle_draw':           '1m_third_candle_draw',
        'third_candle_range':          '1m_third_candle_range',
        'third_candle_body_to_range':  '1m_third_candle_body_to_range',
        'anchor_high':                 '1m_anchor_high',
        'anchor_low':                  '1m_anchor_low',
        'draw_started_at':             '1m_draw_started_at',
    })

    mtf = base.copy().sort_values('datetime')

    # 2) For each HTF: build open-state, align (backward), and apply ID-aware early purge
    for tf_label, fname in TF_FILES.items():
        if tf_label == "1m":
            continue

        tf_path = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(tf_path):
            print(f"[WARN] Skipping missing TF file: {tf_label} -> {tf_path}")
            continue

        print(f"[INFO] Aligning {tf_label} (bar-start, using OPEN-STATE) to base 1m: {tf_path}")
        tf_df = load_tf(tf_path, label=tf_label)

        # Build state-at-open
        tf_open = build_open_state(tf_df)  # columns: datetime, draw_open, anchors_open, range_open, ratio_open, draw_id_open

        # Ensure sorted
        tf_open = tf_open.sort_values('datetime')
        mtf = mtf.sort_values('datetime')

        # As-of merge (backward): at each minute, take the most recent HTF open-state at or before that minute
        mtf = pd.merge_asof(
            mtf,
            tf_open,
            on='datetime',
            direction='backward',
            suffixes=('','')
        )

        # Rename to TF-specific names to avoid collisions on next TF
        mtf.rename(columns={
            'draw_open':        f'{tf_label}_open_draw',
            'anchor_high_open': f'{tf_label}_open_anchor_high',
            'anchor_low_open':  f'{tf_label}_open_anchor_low',
            'range_open':       f'{tf_label}_open_range',
            'ratio_open':       f'{tf_label}_open_ratio',
            'draw_id_open':     f'{tf_label}_open_draw_id',
        }, inplace=True)

        # Types
        mtf[f'{tf_label}_open_draw']    = pd.to_numeric(mtf[f'{tf_label}_open_draw'], errors='coerce').fillna(0).astype('int8')
        mtf[f'{tf_label}_open_draw_id'] = pd.to_datetime(mtf[f'{tf_label}_open_draw_id'], errors='coerce')

        # Apply early purge (ID-aware) on the 1m bars
        mtf = apply_early_purge_with_ids(
            mtf,
            draw_col=f'{tf_label}_open_draw',
            id_col=f'{tf_label}_open_draw_id',
            a_high_col=f'{tf_label}_open_anchor_high',
            a_low_col=f'{tf_label}_open_anchor_low',
            range_col=f'{tf_label}_open_range',
            ratio_col=f'{tf_label}_open_ratio',
            out_prefix=tf_label
        )

        # Sanity summary
        purges = int(mtf[f'{tf_label}_purged_flag'].sum())
        unique_draws = mtf[f'{tf_label}_open_draw_id'].dropna().nunique()
        print(f"[INFO] {tf_label}: early purges flagged = {purges} (unique draw IDs at-open = {unique_draws})")

    # 3) Save combined CSV
    mtf.to_csv(COMBINED_CSV, index=False)
    print(f"[OK ] Wrote combined MTF CSV: {COMBINED_CSV}")

if __name__ == "__main__":
    main()
