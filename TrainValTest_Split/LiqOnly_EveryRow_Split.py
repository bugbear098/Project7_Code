import pandas as pd
import numpy as np
import os
from datetime import timedelta

# === Paths ===
BASE_DIR = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium"
TARGETS_FILE = os.path.join(BASE_DIR, "Data_Medium_Resampled", "resampled_1m_with_targets.csv")
LIQUIDITY_FILE = os.path.join(BASE_DIR, "Feature_SandPit", "Correct_Code", "Liquidity_Sweeps", "Liquidity_Final_Data", "liq_mtf_eff_combined_base1m.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "Processed_Transformer")
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
# Time of day (minutes since midnight)
minutes_in_day = 24 * 60
df['minutes_since_midnight'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
df['time_of_day_sin'] = np.sin(2 * np.pi * df['minutes_since_midnight'] / minutes_in_day)
df['time_of_day_cos'] = np.cos(2 * np.pi * df['minutes_since_midnight'] / minutes_in_day)

# Day of week (0=Mon, 6=Sun)
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# === Final feature/target split ===
X = df.drop(columns=['datetime', 'target', 'minutes_since_midnight', 'day_of_week'])
y = df[['target']]

# === Train/val/test split ===
from sklearn.model_selection import train_test_split

# First split into train+temp (90%) and test (10%)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

# Then split train+val into 70/20
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2222, random_state=42, shuffle=True)
# (0.2222 ≈ 20% of total, since 0.2222 * 0.9 ≈ 0.20)

# === Save to CSV ===
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)

X_val.to_csv(os.path.join(OUTPUT_DIR, "X_val.csv"), index=False)
y_val.to_csv(os.path.join(OUTPUT_DIR, "y_val.csv"), index=False)

X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

print("✅ Data pipeline complete. Saved 6 files to:", OUTPUT_DIR)
