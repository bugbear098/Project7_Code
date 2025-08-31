import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import datetime

# Load raw data with no headers
df = pd.read_csv('NQ 03-24.txt', sep=';', header=None)

# Assign column names
df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

# Parse datetime string
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')

# (Momentum logic continues here as before...)
# Drop the first day (irregular timestamps)
first_day = df['datetime'].dt.date.min()
df = df[df['datetime'].dt.date > first_day].copy()
df.reset_index(drop=True, inplace=True)


# Initialize output column
momentum_active = [0] * len(df)
active_state = 0  # 0 = inactive, 1 = bullish, -1 = bearish
ref_high = None
ref_low = None

# Loop through each candle
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

# Add to DataFrame
df['momentum_candle_active'] = momentum_active

# Save to CSV
df.to_csv('NQ_03-24_with_momentum.csv', index=False)
print("Saved to NQ_03-24_with_momentum.csv")

# Plotting first 100 candles
subset = df.iloc[:100]

fig, ax = plt.subplots(figsize=(16, 8))
for idx, row in subset.iterrows():
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
