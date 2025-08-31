import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------- Parameters --------
INPUT_FILE = "NQ 03-24.txt"  # Replace with your file path
OUTPUT_FILE = "purge_intensity_1m.csv"
PLOT_CANDLES = 100
ROLLING_WINDOW = 100     # For cumulative_purge_intensity
DECAY = 0.9
TIME_WINDOW = 100        # For purge_direction_time
PURGE_WINDOW = 5         # For purge_direction_purges

# -------- 1. Load Data --------
df = pd.read_csv(INPUT_FILE, sep=';', header=None,
                 names=['DateTime','Open','High','Low','Close','Volume'])
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
df.sort_values('DateTime', inplace=True)
df.set_index('DateTime', inplace=True)

# Drop the first day (irregular timestamps)
first_day = df.index.min().date()
df = df[df.index.date > first_day].copy()
df.reset_index(inplace=True)

# -------- 2. Compute Swing Points --------
df['swing_high'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
df['swing_low']  = (df['Low']  < df['Low'].shift(1))  & (df['Low']  < df['Low'].shift(-1))

# -------- 3. Purge Intensity and Direction Calculation --------
swing_highs = []
swing_lows = []
purge_intensity = []
purge_events = []  # (index, sign)

for i in range(len(df)):
    high = df.at[i, 'High']
    low = df.at[i, 'Low']
    purged = 0

    # Add new swing highs/lows
    if df.at[i, 'swing_high']:
        swing_highs.append((i, df.at[i, 'High']))
    if df.at[i, 'swing_low']:
        swing_lows.append((i, df.at[i, 'Low']))

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
purge_direction_time = []
purge_direction_purges = []

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
df[['DateTime','Open','High','Low','Close','Volume',
    'swing_high','swing_low','purge_intensity',
    'cumulative_purge_intensity','purge_direction_time','purge_direction_purges']
].to_csv(OUTPUT_FILE, index=False)

# -------- 7. Plot First 100 Candles --------
subset = df.iloc[:PLOT_CANDLES]
fig, ax = plt.subplots(figsize=(12, 6))

for idx, row in subset.iterrows():
    color = 'green' if row['Close'] >= row['Open'] else 'red'
    ax.plot([row['DateTime'], row['DateTime']], [row['Low'], row['High']], color='black', linewidth=1)
    ax.add_patch(plt.Rectangle(
        (row['DateTime'], min(row['Open'], row['Close'])),
        width=pd.Timedelta(minutes=1),
        height=abs(row['Close'] - row['Open']),
        color=color,
        alpha=0.8
    ))
    if row['purge_intensity'] > 0:
        ax.text(row['DateTime'], row['High'] + 1, str(int(row['purge_intensity'])),
                ha='center', va='bottom', fontsize=8, color='blue')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_title("First 100 Candles with Purge Indicators")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
