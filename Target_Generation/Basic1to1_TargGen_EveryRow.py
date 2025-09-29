import pandas as pd
import os
from tqdm import tqdm

# === Paths ===
INPUT_FILE = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Data_Medium_Resampled/resampled_1m.csv"
OUTPUT_FILE = os.path.join(os.path.dirname(INPUT_FILE), "resampled_1m_with_targets.csv")

# === Parameters ===
Z = 20  # default distance for bullish/bearish targets

# === Load data ===
df = pd.read_csv(INPUT_FILE)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Ensure OHLC columns exist
required_cols = {'open', 'high', 'low', 'close'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing one or more required columns: {required_cols - set(df.columns)}")

# === Generate bullish & bearish thresholds ===
df['bullish_target_price'] = df['open'] + Z
df['bearish_target_price'] = df['open'] - Z

# === Function to check which target gets hit first ===
def get_first_hit(row_idx, df, Z):
    """Return +1 if bullish target is hit first, -1 if bearish target is hit first, None if neither."""
    start_open = df.iloc[row_idx]['open']
    bull_price = start_open + Z
    bear_price = start_open - Z
    
    # iterate forward until one of the targets is reached
    for i in range(row_idx + 1, len(df)):
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        
        if high >= bull_price:
            return 1   # bullish first
        if low <= bear_price:
            return -1  # bearish first
    return None  # neither reached (e.g., end of dataset)

# === Apply target generation with progress bar ===
targets = []
for idx in tqdm(range(len(df)), desc="Generating targets"):
    targets.append(get_first_hit(idx, df, Z))

df['target'] = targets

# === Save ===
df.to_csv(OUTPUT_FILE)
print(f"Saved dataset with targets to: {OUTPUT_FILE}")
