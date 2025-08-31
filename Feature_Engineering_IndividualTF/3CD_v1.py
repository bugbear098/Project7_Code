import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# =========================
# Paths (your pattern)
# =========================
script_dir = os.path.dirname(__file__)
input_file = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Data_Medium_Resampled', 'resampled_5m.csv'))
output_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Feature_SandPit','3CD'))
os.makedirs(output_dir, exist_ok=True)


# =========================
# Load data
# =========================
df = pd.read_csv(input_file)
df.columns = [c.strip().lower() for c in df.columns]

required = {'datetime', 'open', 'high', 'low', 'close'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing one or more required columns in {input_file}: {missing}")

df['datetime'] = pd.to_datetime(df['datetime'], errors='raise', utc=True)
df = df.sort_values('datetime').reset_index(drop=True)

df.set_index('datetime', inplace=True)
df.index = df.index.tz_convert('UTC').tz_localize(None)

# Drop first calendar day
first_midnight = df.index.normalize()[0]
df = df[df.index.normalize() > first_midnight].copy()
df.reset_index(inplace=True)

# =========================
# Third Candle Draw + extra features for candle i (PERSIST while active)
# =========================
df['third_candle_draw'] = 0
df['third_candle_range'] = 0.0
df['third_candle_body_to_range'] = 0.0

draw_active = 0   # 1 bullish, -1 bearish, 0 none
draw_high = None
draw_low  = None
persist_range = 0.0
persist_ratio = 0.0

def is_swing_high(i):
    return (df['high'].iloc[i-1] > df['high'].iloc[i-2]) and (df['high'].iloc[i-1] > df['high'].iloc[i])

def is_swing_low(i):
    return (df['low'].iloc[i-1] < df['low'].iloc[i-2]) and (df['low'].iloc[i-1] < df['low'].iloc[i])

EPS = 1e-12  # guard for zero-range division

for i in range(2, len(df)):
    swing_high = is_swing_high(i)
    swing_low  = is_swing_low(i)

    started_new_draw = False

    if swing_high and swing_low:
        pass
    elif swing_high:
        # New bearish draw anchored to THIRD candle (i)
        draw_active = -1
        draw_high = df['high'].iloc[i]
        draw_low  = df['low'].iloc[i]
        # compute & set persisting features from candle i
        persist_range = float(df['high'].iloc[i] - df['low'].iloc[i])
        body_i = abs(float(df['open'].iloc[i] - df['close'].iloc[i]))
        persist_ratio = body_i / max(persist_range, EPS)
        started_new_draw = True

    elif swing_low:
        # New bullish draw anchored to THIRD candle (i)
        draw_active = 1
        draw_high = df['high'].iloc[i]
        draw_low  = df['low'].iloc[i]
        # compute & set persisting features from candle i
        persist_range = float(df['high'].iloc[i] - df['low'].iloc[i])
        body_i = abs(float(df['open'].iloc[i] - df['close'].iloc[i]))
        persist_ratio = body_i / max(persist_range, EPS)
        started_new_draw = True

    # Purge / invalidation vs the anchored third-candle levels
    if draw_active == 1:
        if (df['high'].iloc[i] > draw_high) or (df['close'].iloc[i] < draw_low):
            draw_active = 0
            persist_range = 0.0
            persist_ratio = 0.0
    elif draw_active == -1:
        if (df['low'].iloc[i] < draw_low) or (df['close'].iloc[i] > draw_high):
            draw_active = 0
            persist_range = 0.0
            persist_ratio = 0.0

    # Set outputs
    df.at[i, 'third_candle_draw'] = draw_active
    # Persist values while draw_active != 0, else 0
    df.at[i, 'third_candle_range'] = persist_range if draw_active != 0 else 0.0
    df.at[i, 'third_candle_body_to_range'] = persist_ratio if draw_active != 0 else 0.0

# =========================
# Save result
# =========================
output_csv = os.path.join(output_dir, 'concatenated_1m_with_third_candle_draw.csv')
df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")

# =========================
# Plotting helpers
# =========================
def infer_bar_width_days(sample_df):
    if len(sample_df) < 2:
        return 60.0 / (24*3600)
    deltas = sample_df['datetime'].diff().dropna()
    return np.median(deltas.dt.total_seconds()) / (24*3600)

def plot_window(window_df, title, outpath):
    data = window_df.copy()
    bar_width_days = infer_bar_width_days(data)
    data['mdates'] = mdates.date2num(data['datetime'])

    fig, ax = plt.subplots(figsize=(16, 8))
    for _, row in data.iterrows():
        x = row['mdates']
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        up = c >= o

        # Wick
        ax.plot([x, x], [l, h], linewidth=1, color='black')

        # Body
        body_bottom = min(o, c)
        body_height = abs(c - o)
        rect = Rectangle((x, body_bottom), width=bar_width_days, height=body_height,
                         edgecolor='black', facecolor=('green' if up else 'red'))
        ax.add_patch(rect)

        # Feature label
        draw_val = row['third_candle_draw']
        if draw_val != 0:
            offset = max((h - l) * 0.02, 0.5)
            label_y = h + offset if draw_val == 1 else l - offset
            ax.text(x + bar_width_days * 0.5, label_y, f'{int(draw_val)}',
                    ha='center', va='bottom' if draw_val == 1 else 'top',
                    fontsize=8, color=('blue' if draw_val == 1 else 'darkred'))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_xlim(data['mdates'].min() - bar_width_days, data['mdates'].max() + bar_width_days*2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)

# =========================
# Create 5×200-bar windows (first + 4 random)
# =========================
WINDOW = 200
n = len(df)
if n < WINDOW:
    raise ValueError(f"Not enough rows to make a {WINDOW}-bar window. Rows: {n}")

windows = [(0, WINDOW)]
rng = np.random.default_rng(42)
candidates = np.arange(1, n - WINDOW + 1)
rand_starts = rng.choice(candidates, size=4, replace=False)
for s in rand_starts:
    windows.append((s, s + WINDOW))

print("Plot windows (start_index, end_index):", windows)

for idx, (s, e) in enumerate(windows, start=1):
    win_df = df.iloc[s:e].copy()
    title = f'Third Candle Draw — Window {idx} ({s}:{e})'
    out_png = os.path.join(output_dir, f'third_candle_draw_window_{idx}_{s}_{e}.png')
    plot_window(win_df, title, out_png)
    print(f"Saved plot: {out_png}")

