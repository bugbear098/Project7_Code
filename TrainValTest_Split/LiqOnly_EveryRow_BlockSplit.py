import pandas as pd
import numpy as np
import os
from datetime import timedelta

# === Paths ===
BASE_DIR = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium"
TARGETS_FILE = os.path.join(BASE_DIR, "Data_Medium_Resampled", "resampled_1m_with_targets.csv")
LIQUIDITY_FILE = os.path.join(BASE_DIR, "Feature_SandPit", "Correct_Code", "Liquidity_Sweeps", "Liquidity_Final_Data", "liq_mtf_eff_combined_base1m.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "Processed_Transformer_Chrono")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
df_targets = pd.read_csv(TARGETS_FILE, parse_dates=['datetime'])
df_liq = pd.read_csv(LIQUIDITY_FILE, parse_dates=['datetime'])

# Ensure datetime alignment (floor to minute just in case)
df_targets['datetime'] = pd.to_datetime(df_targets['datetime'], utc=True).dt.floor('min')
df_liq['datetime'] = pd.to_datetime(df_liq['datetime'], utc=True).dt.floor('min')

# === Merge on datetime ===
df = pd.merge(df_liq, df_targets[['datetime', 'target']], on='datetime', how='inner')

# === Drop first 2 weeks of rows ===
first_time = df['datetime'].min()
cutoff_time = first_time + timedelta(days=14)
df = df[df['datetime'] >= cutoff_time].reset_index(drop=True)

# === Create cyclical datetime features ===
minutes_in_day = 24 * 60
df['minutes_since_midnight'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
df['time_of_day_sin'] = np.sin(2 * np.pi * df['minutes_since_midnight'] / minutes_in_day)
df['time_of_day_cos'] = np.cos(2 * np.pi * df['minutes_since_midnight'] / minutes_in_day)

df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# === Final feature/target split ===
X = df.drop(columns=['datetime', 'target', 'minutes_since_midnight', 'day_of_week'])
y = df[['target']]

# === Chronological split ===
n = len(X)
train_end = int(0.7 * n)
val_end = int(0.9 * n)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

# === Save ===
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)

X_val.to_csv(os.path.join(OUTPUT_DIR, "X_val.csv"), index=False)
y_val.to_csv(os.path.join(OUTPUT_DIR, "y_val.csv"), index=False)

X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

print("✅ Chronological split complete. Saved 6 files to:", OUTPUT_DIR)
print(f"Train: {len(X_train)} rows ({100*train_end/n:.1f}%)")
print(f"Val:   {len(X_val)} rows ({100*(val_end-train_end)/n:.1f}%)")
print(f"Test:  {len(X_test)} rows ({100*(n-val_end)/n:.1f}%)")
