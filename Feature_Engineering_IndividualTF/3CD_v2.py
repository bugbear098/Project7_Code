import os
import pandas as pd
import numpy as np

# =========================
# CONFIG — edit these
# =========================

script_dir = os.path.dirname(__file__)
INPUT_DIR  = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Data_Medium_Resampled'))  # <-- put all TF CSVs here
OUTPUT_DIR = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Feature_SandPit','3CD'))

# Map timeframe label -> filename in INPUT_DIR
# Edit names to match your files
TF_FILES = {
    "1m":  "Data_1m.csv",
    "5m":  "resampled_5m.csv",
    "15m": "resampled_15m.csv",
    "1H":  "resampled_1H.csv",
    "4H":  "resampled_4H.csv",
    "1D":  "resampled_1D.csv",
}

# Whether to drop the first calendar day (often useful for futures data quirks)
DROP_FIRST_DAY = False

# Small epsilon to avoid divide-by-zero
EPS = 1e-12

# =========================
# Core 3CD computation (THIRD candle = i)
# =========================
def compute_3cd_features(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize columns
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Basic checks
    required = {'datetime', 'open', 'high', 'low', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Time parsing (tz-safe), sort
    df['datetime'] = pd.to_datetime(df['datetime'], errors='raise', utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    # Make index tz-naive UTC for consistent ops
    df.set_index('datetime', inplace=True)
    df.index = df.index.tz_convert('UTC').tz_localize(None)

    # Optional: drop first calendar day
    if DROP_FIRST_DAY and len(df) > 0:
        first_midnight = df.index.normalize()[0]
        df = df[df.index.normalize() > first_midnight].copy()

    df.reset_index(inplace=True)

    # Output columns (persisting)
    df['third_candle_draw'] = 0
    df['third_candle_range'] = 0.0
    df['third_candle_body_to_range'] = 0.0
    df['anchor_high'] = 0.0
    df['anchor_low'] = 0.0
    df['draw_started_at'] = pd.NaT

    # State for persistence
    draw_active = 0        # +1 bullish, -1 bearish, 0 none
    draw_high = None       # anchor_high
    draw_low = None        # anchor_low
    persist_range = 0.0
    persist_ratio = 0.0
    persist_anchor_high = 0.0
    persist_anchor_low = 0.0
    persist_started_at = pd.NaT

    def is_swing_high(i):
        # middle is i-1; third candle is i (anchor uses candle i per your spec)
        return (df['high'].iloc[i-1] > df['high'].iloc[i-2]) and (df['high'].iloc[i-1] > df['high'].iloc[i])

    def is_swing_low(i):
        return (df['low'].iloc[i-1] < df['low'].iloc[i-2]) and (df['low'].iloc[i-1] < df['low'].iloc[i])

    for i in range(2, len(df)):
        swing_high = is_swing_high(i)
        swing_low = is_swing_low(i)

        started_new_draw = False

        if swing_high and swing_low:
            # Rare edge case: ignore
            pass
        elif swing_high:
            # New bearish draw -> anchor to THIRD candle i
            draw_active = -1
            draw_high = float(df['high'].iloc[i])  # anchor_high
            draw_low  = float(df['low'].iloc[i])   # anchor_low

            # Persisting features from candle i
            persist_range = float(df['high'].iloc[i] - df['low'].iloc[i])
            body_i = abs(float(df['open'].iloc[i] - df['close'].iloc[i]))
            persist_ratio = body_i / max(persist_range, EPS)
            persist_anchor_high = draw_high
            persist_anchor_low = draw_low
            persist_started_at = df['datetime'].iloc[i]
            started_new_draw = True

        elif swing_low:
            # New bullish draw -> anchor to THIRD candle i
            draw_active = 1
            draw_high = float(df['high'].iloc[i])  # anchor_high
            draw_low  = float(df['low'].iloc[i])   # anchor_low

            # Persisting features from candle i
            persist_range = float(df['high'].iloc[i] - df['low'].iloc[i])
            body_i = abs(float(df['open'].iloc[i] - df['close'].iloc[i]))
            persist_ratio = body_i / max(persist_range, EPS)
            persist_anchor_high = draw_high
            persist_anchor_low = draw_low
            persist_started_at = df['datetime'].iloc[i]
            started_new_draw = True

        # Same-timeframe purge/invalid (kept exactly as before)
        # Bullish: purge if price takes anchor_high; invalidate if close < anchor_low
        if draw_active == 1:
            if (df['high'].iloc[i] > draw_high) or (df['close'].iloc[i] < draw_low):
                draw_active = 0
                # When draw ends on this bar, reset persisted extras AFTER writing this row’s state below
                persist_range = 0.0
                persist_ratio = 0.0
                persist_anchor_high = 0.0
                persist_anchor_low = 0.0
                persist_started_at = pd.NaT

        # Bearish: purge if price takes anchor_low; invalidate if close > anchor_high
        elif draw_active == -1:
            if (df['low'].iloc[i] < draw_low) or (df['close'].iloc[i] > draw_high):
                draw_active = 0
                persist_range = 0.0
                persist_ratio = 0.0
                persist_anchor_high = 0.0
                persist_anchor_low = 0.0
                persist_started_at = pd.NaT

        # Write row values
        df.at[i, 'third_candle_draw'] = draw_active
        df.at[i, 'third_candle_range'] = persist_range if draw_active != 0 else 0.0
        df.at[i, 'third_candle_body_to_range'] = persist_ratio if draw_active != 0 else 0.0
        df.at[i, 'anchor_high'] = persist_anchor_high if draw_active != 0 else 0.0
        df.at[i, 'anchor_low']  = persist_anchor_low if draw_active != 0 else 0.0
        df.at[i, 'draw_started_at'] = persist_started_at if draw_active != 0 else pd.NaT

    return df

# =========================
# Batch process all TF files
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for tf_label, fname in TF_FILES.items():
        in_path  = os.path.join(INPUT_DIR, fname)
        out_name = os.path.splitext(fname)[0] + "_with_3cd.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if not os.path.exists(in_path):
            print(f"[WARN] Missing file for {tf_label}: {in_path}")
            continue

        print(f"[INFO] Processing {tf_label}: {in_path}")
        df = pd.read_csv(in_path)
        df_enriched = compute_3cd_features(df)
        df_enriched.to_csv(out_path, index=False)
        print(f"[OK] Wrote: {out_path}")

if __name__ == "__main__":
    main()
