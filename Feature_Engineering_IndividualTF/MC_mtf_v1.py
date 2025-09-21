import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import datetime
import os

# === Paths ===
INPUT_DIR  = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Data_Medium_Resampled"
OUTPUT_DIR = os.path.join(INPUT_DIR, "Processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Map timeframe to candle width ===
TIMEFRAME_WIDTHS = {
    "1m": datetime.timedelta(minutes=1),
    "5m": datetime.timedelta(minutes=5),
    "15m": datetime.timedelta(minutes=15),
    "1H": datetime.timedelta(hours=1),
    "4H": datetime.timedelta(hours=4),
    "1D": datetime.timedelta(days=1)
}

# === Momentum Candle Logic ===
def process_file(file_path, output_dir, tf_label):
    print(f"Processing {os.path.basename(file_path)} ...")

    # Load CSV
    df = pd.read_csv(file_path, sep=",", header=0)
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    # Parse datetime: standardize to UTC, then convert to New York time
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert("America/New_York")

    # Features
    momentum_active = [0] * len(df)
    momentum_strength = [0.0] * len(df)
    wick_to_body = [0.0] * len(df)
    total_range = [0.0] * len(df)

    active_state = 0
    ref_high = None
    ref_low = None
    last_strength = 0.0
    last_wick_to_body = 0.0
    last_total_range = 0.0

    for i in range(1, len(df)):
        prev_close = df['close'].iloc[i - 1]
        prev_high = df['high'].iloc[i - 1]
        prev_low  = df['low'].iloc[i - 1]

        open_ = df['open'].iloc[i]
        close = df['close'].iloc[i]
        high  = df['high'].iloc[i]
        low   = df['low'].iloc[i]

        # --- Fresh momentum candle ---
        if close > prev_high:
            momentum_active[i] = 1
            active_state = 1
            ref_high = high
            ref_low = low

            # features
            last_strength = close - prev_high
            last_wick_to_body = (high - low) / abs(close - open_) if close != open_ else 0.0
            last_total_range = high - low

        elif close < prev_low:
            momentum_active[i] = -1
            active_state = -1
            ref_high = high
            ref_low = low

            # features
            last_strength = prev_low - close
            last_wick_to_body = (high - low) / abs(close - open_) if close != open_ else 0.0
            last_total_range = high - low

        # --- Continuation / Reset ---
        elif active_state == 1:  # bullish active
            if high > ref_high or close < ref_low:
                momentum_active[i] = 0
                active_state = 0
                ref_high, ref_low = None, None
                last_strength = last_wick_to_body = last_total_range = 0.0
            else:
                momentum_active[i] = 1

        elif active_state == -1:  # bearish active
            if low < ref_low or close > ref_high:
                momentum_active[i] = 0
                active_state = 0
                ref_high, ref_low = None, None
                last_strength = last_wick_to_body = last_total_range = 0.0
            else:
                momentum_active[i] = -1

        # Assign features (persist only while active)
        momentum_strength[i] = last_strength if active_state != 0 else 0.0
        wick_to_body[i] = last_wick_to_body if active_state != 0 else 0.0
        total_range[i] = last_total_range if active_state != 0 else 0.0

    # Add features
    df['momentum_candle_active'] = momentum_active
    df['momentum_strength'] = momentum_strength
    df['wick_to_body'] = wick_to_body
    df['total_range'] = total_range

    # Save processed file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_with_momentum.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Plot first 100 candles
    subset = df.iloc[:100]
    fig, ax = plt.subplots(figsize=(16, 8))

    candle_width = TIMEFRAME_WIDTHS[tf_label]

    for _, row in subset.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'

        # Wick
        ax.plot([row['datetime'], row['datetime']], [row['low'], row['high']], color='black')

        # Body
        rect = Rectangle(
            (row['datetime'], min(row['open'], row['close'])),
            width=candle_width,
            height=abs(row['close'] - row['open']),
            color=color
        )
        ax.add_patch(rect)

        # Label momentum
        ax.text(row['datetime'], row['high'], str(row['momentum_candle_active']),
                ha='center', va='bottom', fontsize=8, color='blue')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    ax.set_title(f"First 100 Candles with Momentum Feature ({base_name})")
    ax.set_xlabel('Time (America/New_York)')
    ax.set_ylabel('Price')
    plt.tight_layout()
    plt.show()


# === Main Loop ===
file_map = {
    "resampled_1m.csv": "1m",
    "resampled_5m.csv": "5m",
    "resampled_15m.csv": "15m",
    "resampled_1H.csv": "1H",
    "resampled_4H.csv": "4H",
    "resampled_1D.csv": "1D"
}

for filename, tf_label in file_map.items():
    file_path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(file_path):
        process_file(file_path, OUTPUT_DIR, tf_label)
    else:
        print(f"⚠️ File not found: {filename}")
