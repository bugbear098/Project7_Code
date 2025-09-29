import pandas as pd
import os
import numpy as np
from numba import njit, prange
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

opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
n = len(df)

# === Numba-accelerated function ===
@njit(parallel=True)
def compute_targets(opens, highs, lows, Z):
    n = len(opens)
    targets = np.full(n, np.nan)
    for i in prange(n):  # parallel outer loop
        bull_price = opens[i] + Z
        bear_price = opens[i] - Z
        for j in range(i+1, n):
            if highs[j] >= bull_price:
                targets[i] = 1
                break
            if lows[j] <= bear_price:
                targets[i] = -1
                break
    return targets

# === Run with progress ===
print("Computing targets with Numba (first run may be slower due to JIT compilation)...")
targets = compute_targets(opens, highs, lows, Z)

df['bullish_target_price'] = opens + Z
df['bearish_target_price'] = opens - Z
df['target'] = targets

# === Save ===
df.to_csv(OUTPUT_FILE)
print(f"Saved dataset with targets to: {OUTPUT_FILE}")
