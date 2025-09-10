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

file_path = os.path.join(INPUT_DIR, "Data_1m.csv")

# === Load data ===
df = pd.read_csv(file_path, sep=",", header=0)
df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

# Parse timezone-aware datetime -> convert to UTC
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

# Drop the first day (irregular timestamps)
#first_day = df['datetime'].dt.date.min()
#df = df[df['datetime'].dt.date > first_day].copy()
#df.reset_index(drop=True, inplace=True)

# === Momentum candle feature ===
momentum_active = [0] * len(df)
active_state = 0  # 0 = inactive, 1 = bullish, -1 = bearish
ref_high = None
ref_low = None

for i in range(1, len(df)):
    prev_close = df['close'].iloc[i - 1]
    prev_high = df['high'].iloc[i - 1]
    prev_low = df['low'].iloc[i - 1]

    close = df['close'].iloc[i]
    high = df['high'].iloc[i]
    low = df['low'].iloc[i]

    # --- Priority 1: Fresh momentum candle ---
    if close > prev_high:
        momentum_active[i] = 1
        active_state = 1
        ref_high = high
        ref_low = low
        continue

    if close < prev_low:
        momentum_active[i] = -1
        active_state = -1
        ref_high = high
        ref_low = low
        continue

    # --- Priority 2: Handle continuation or deactivation ---
    if active_state == 0:
        momentum_active[i] = 0
    elif active_state == 1:
        if high > ref_high or close < ref_low:
            momentum_active[i] = 0
            active_state = 0
            ref_high = None
            ref_low = None
        else:
            momentum_active[i] = 1
    elif active_state == -1:
        if low < ref_low or close > ref_high:
            momentum_active[i] = 0
            active_state = 0
            ref_high = None
            ref_low = None
        else:
            momentum_active[i] = -1

df['momentum_candle_active'] = momentum_active

# === Save output ===
output_path = os.path.join(OUTPUT_DIR, "NQ_03-24_with_momentum.csv")
df.to_csv(output_path, index=False)
print("Saved to", output_path)

# === Plot first 100 candles ===
subset = df.iloc[:100]

fig, ax = plt.subplots(figsize=(16, 8))
for _, row in subset.iterrows():
    color = 'green' if row['close'] >= row['open'] else 'red'

    # Wick
    ax.plot([row['datetime'], row['datetime']], [row['low'], row['high']], color='black')

    # Body
    rect = Rectangle(
        (row['datetime'], min(row['open'], row['close'])),
        width=datetime.timedelta(minutes=1),
        height=abs(row['close'] - row['open']),
        color=color
    )
    ax.add_patch(rect)

    # Momentum label
    ax.text(row['datetime'], row['high'], str(row['momentum_candle_active']),
            ha='center', va='bottom', fontsize=8, color='blue')

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)
ax.set_title('First 100 Candles with Momentum Feature')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
plt.tight_layout()
plt.show()
