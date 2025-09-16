import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------- Parameters --------
INPUT_DIR = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Feature_SandPit/Correct_Code/Correct_Processed_Data"
OUTPUT_DIR = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Feature_SandPit/Liq"

PLOT_CANDLES = 100
ROLLING_WINDOW = 100     # For cumulative_purge_intensity
DECAY = 0.9
TIME_WINDOW = 100        # For purge_direction_time
PURGE_WINDOW = 5         # For purge_direction_purges

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_liquidity_features(df, timeframe_label):
    # --- Fix datetime handling (with timezone) ---
    if 'datetime' in df.columns:
        dt_col = 'datetime'
    elif 'DateTime' in df.columns:
        dt_col = 'DateTime'
    else:
        raise ValueError(f"No datetime column found in {timeframe_label} data")

    # Parse with UTC and convert to US Eastern so offsets show (-05:00 or -04:00)
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_convert("America/New_York")
    df.rename(columns={dt_col: 'datetime'}, inplace=True)

    # Standardize OHLCV column names if needed
    rename_map = {c: c.lower() for c in df.columns if c.lower() in ['open','high','low','close','volume']}
    df.rename(columns=rename_map, inplace=True)

    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    df.reset_index(drop=True, inplace=True)

    # -------- 2. Compute Swing Points --------
    df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
    df['swing_low']  = (df['low']  < df['low'].shift(1))  & (df['low']  < df['low'].shift(-1))

    # -------- 3. Purge Intensity and Direction Calculation --------
    swing_highs, swing_lows, purge_intensity, purge_events = [], [], [], []

    for i in range(len(df)):
        high, low = df.at[i, 'high'], df.at[i, 'low']
        purged = 0

        # Add new swing highs/lows
        if df.at[i, 'swing_high']:
            swing_highs.append((i, high))
        if df.at[i, 'swing_low']:
            swing_lows.append((i, low))

        # Purge highs
        new_highs = []
        for idx, h in swing_highs:
            if high > h:
                purged += 1
                purge_events.append((i, -1))  # Swing high purge = -1
            else:
                new_highs.append((idx, h))
        swing_highs = new_highs

        # Purge lows
        new_lows = []
        for idx, l in swing_lows:
            if low < l:
                purged += 1
                purge_events.append((i, +1))  # Swing low purge = +1
            else:
                new_lows.append((idx, l))
        swing_lows = new_lows

        purge_intensity.append(purged)

    df['purge_intensity'] = purge_intensity

    # -------- 4. Cumulative Purge Intensity --------
    df['cumulative_purge_intensity'] = df['purge_intensity'].rolling(ROLLING_WINDOW, min_periods=1).sum()

    # -------- 5. Decay-Weighted Purge Direction Features --------
    purge_direction_time, purge_direction_purges = [], []

    for i in range(len(df)):
        # --- Time-based decay ---
        net_time = 0.0
        for idx, sign in reversed(purge_events):
            if idx < i - TIME_WINDOW:
                break
            net_time = net_time * DECAY + sign
        purge_direction_time.append(net_time)

        # --- Purge-count-based decay ---
        recent = [sign for idx, sign in reversed(purge_events) if idx <= i][:PURGE_WINDOW]
        net_purge = 0.0
        for sign in recent:
            net_purge = net_purge * DECAY + sign
        purge_direction_purges.append(net_purge)

    df['purge_direction_time'] = purge_direction_time
    df['purge_direction_purges'] = purge_direction_purges

    # -------- 6. Save Output --------
    out_file = os.path.join(OUTPUT_DIR, f"liq_{timeframe_label}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file}")

    # -------- 7. Plot First 100 Candles (for every timeframe) --------
    subset = df.iloc[:PLOT_CANDLES]
    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in subset.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax.plot([row['datetime'], row['datetime']], [row['low'], row['high']], color='black', linewidth=1)
        ax.add_patch(plt.Rectangle(
            (row['datetime'], min(row['open'], row['close'])),
            width=pd.Timedelta(minutes=1),
            height=abs(row['close'] - row['open']),
            color=color,
            alpha=0.8
        ))
        if row['purge_intensity'] > 0:
            ax.text(row['datetime'], row['high'], str(int(row['purge_intensity'])),
                    ha='center', va='bottom', fontsize=8, color='blue')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title(f"First {PLOT_CANDLES} Candles with Purge Indicators ({timeframe_label})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------- Run for all resampled files --------
for file in os.listdir(INPUT_DIR):
    if file.startswith("resampled_") and file.endswith(".csv"):
        filepath = os.path.join(INPUT_DIR, file)
        timeframe = file.replace("resampled_", "").replace(".csv", "")
        df = pd.read_csv(filepath)
        compute_liquidity_features(df, timeframe)
