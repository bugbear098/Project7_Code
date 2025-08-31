import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# ----------------------------- #
#         CONFIGURATION         #
# ----------------------------- #

script_dir = os.path.dirname(__file__)
RAW_FOLDER = os.path.abspath(os.path.join(script_dir, '..', 'Raw_1m_Data'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(script_dir, '..', 'Concat_1m_Data'))
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'concatenated_1m.csv')

# Example raw line: "20231207 000100;16017.5;16017.5;16017.5;16017.5;1"
DATETIME_FORMAT = '%Y%m%d %H%M%S'
COLUMN_NAMES = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']

# Timezones: raw is UTC → convert to New York local
SRC_TZ = 'UTC'
DST_TZ = 'America/New_York'

# Mid-range gap thresholds
MIN_GAP = timedelta(minutes=1)
MAX_GAP = timedelta(minutes=20)

# ----------------------------- #
#     HELPER FUNCTIONS          #
# ----------------------------- #

def has_midrange_gaps(day_df, min_gap=MIN_GAP, max_gap=MAX_GAP):
    if day_df.empty:
        return False
    gaps = day_df['Datetime'].diff().dropna()
    return ((gaps > min_gap) & (gaps < max_gap)).any()

def trim_contract_edges(df):
    """Trim start and end days that have midrange gaps (>1m and <20m)."""
    df = df.copy()
    if df.empty:
        return df

    # Trim from start
    unique_days = sorted(df['Datetime'].dt.date.unique())
    for day in unique_days:
        day_df = df[df['Datetime'].dt.date == day]
        if has_midrange_gaps(day_df):
            print(f"[Trim Start] Dropping start day with gaps: {day}")
            df = df[df['Datetime'].dt.date > day]
        else:
            break
    if df.empty:
        return df

    # Trim from end
    unique_days = sorted(df['Datetime'].dt.date.unique(), reverse=True)
    for day in unique_days:
        day_df = df[df['Datetime'].dt.date == day]
        if has_midrange_gaps(day_df):
            print(f"[Trim End] Dropping end day with gaps: {day}")
            df = df[df['Datetime'].dt.date < day]
        else:
            break

    return df

def apply_blackouts_ny(df):
    """
    Remove rows in:
      • Mon–Thu 17:00–18:00 NY (daily maintenance)
      • Fri 17:00 → Sun 18:00 NY (weekend)
    Assumes df['Datetime'] is tz-aware in America/New_York.
    """
    if df.empty:
        return df

    ny = df['Datetime']
    dow = ny.dt.dayofweek                 # Mon=0 ... Sun=6
    minutes = ny.dt.hour * 60 + ny.dt.minute

    # Mon–Thu 17:00–18:00
    daily_break = ((dow >= 0) & (dow <= 3)) & (minutes >= 17*60) & (minutes < 18*60)

    # Fri 17:00 → Sun 18:00
    weekend_break = ((dow == 4) & (minutes >= 17*60)) | (dow == 5) | ((dow == 6) & (minutes < 18*60))

    mask = ~(daily_break | weekend_break)
    removed = int((~mask).sum())
    if removed > 0:
        print(f"[Blackouts] Removed {removed} rows (Mon–Thu 17–18 and Fri 17 → Sun 18).")

    return df.loc[mask].copy()

def maintenance_break_audit(out_df):
    """
    Audit which session window has no prints:
      • Expect ~1.0 for 17:01–17:59 (maintenance)
      • Expect ~0.0 for 16:00–16:59 (active trading)
    Uses CME session date (18:00 prev → 17:00 curr) and ignores boundary minutes.
    """
    if out_df.empty:
        print("[Audit] No data to audit.")
        return

    dfc = out_df.copy()
    # Session date anchored to 17:00 close (so maintenance is within the session end)
    dfc['session_date'] = (dfc['Datetime'] - pd.Timedelta(hours=17)).dt.date
    dfc['hhmm'] = dfc['Datetime'].dt.hour * 100 + dfc['Datetime'].dt.minute

    def any_between(g, a, b):  # inclusive bounds for minutes (hhmm)
        return ((g['hhmm'] >= a) & (g['hhmm'] <= b)).any()

    by_sess = dfc.groupby('session_date', sort=True)

    # No prints during maintenance (ignore 17:00 and 18:00 exact)
    no_prints_17_18 = (~by_sess.apply(lambda g: any_between(g, 1701, 1759))).mean()

    # Normal trading hour 16:00–16:59 should have prints most sessions
    no_prints_16_17 = (~by_sess.apply(lambda g: any_between(g, 1600, 1659))).mean()

    print(f"[Audit] No-prints fraction 17:01–17:59: {no_prints_17_18:.3f} (expect ~1.000)")
    print(f"[Audit] No-prints fraction 16:00–16:59: {no_prints_16_17:.3f} (expect ~0.000)")

# ----------------------------- #
#     LOAD & CLEAN FILES        #
# ----------------------------- #

all_dfs = []

for filename in sorted(os.listdir(RAW_FOLDER)):
    if not filename.endswith('.txt'):
        continue

    file_path = os.path.join(RAW_FOLDER, filename)
    contract_name = os.path.splitext(filename)[0]

    # Load raw
    df = pd.read_csv(file_path, sep=';', header=None, names=COLUMN_NAMES)

    # Parse to datetime, localize as UTC, convert to New York local (tz-aware)
    dt = pd.to_datetime(df['Datetime'], format=DATETIME_FORMAT)
    dt = dt.dt.tz_localize(SRC_TZ).dt.tz_convert(DST_TZ)
    df['Datetime'] = dt

    # Drop the first calendar day (NY local)
    first_day = df['Datetime'].dt.date.min()
    df = df[df['Datetime'].dt.date > first_day].copy()

    # Trim early/late bad days based on midrange gaps
    df = trim_contract_edges(df)
    if df.empty:
        print(f"[SKIP] {filename}: All days removed after trimming due to midrange gaps.")
        continue

    # Apply blackouts (Mon–Thu 17–18, Fri 17 → Sun 18)
    before = len(df)
    df = apply_blackouts_ny(df)
    after = len(df)
    if before != after:
        print(f"[{contract_name}] Blackouts removed {before - after} rows.")

    df['Contract'] = contract_name
    all_dfs.append(df)

if not all_dfs:
    raise ValueError("No valid .txt files found in 'Raw_1m_Data' or all files were excluded due to data issues.")

# ----------------------------- #
#     CONCAT & RESOLVE OVERLAPS #
# ----------------------------- #

combined_df = pd.concat(all_dfs, ignore_index=True)

# Newer contracts win on overlapping timestamps
combined_df.sort_values(by=['Datetime', 'Contract'], inplace=True)
combined_df = combined_df.drop_duplicates(subset='Datetime', keep='last').reset_index(drop=True)

# ----------------------------- #
#     CHECK FOR TIME GAPS       #
# ----------------------------- #

combined_df['Time_Diff'] = combined_df['Datetime'].diff()

# Diagnostic only: huge gaps
large_gaps = combined_df[combined_df['Time_Diff'] > timedelta(days=3)]
print(f"[✓] Final Combined shape: {combined_df.shape}")
print(f"[✓] Large gaps (>3d): {len(large_gaps)}")

# Hard sanity check: mid-range gaps should be none
midrange_gaps = combined_df[(combined_df['Time_Diff'] > MIN_GAP) &
                            (combined_df['Time_Diff'] < MAX_GAP)]
if midrange_gaps.empty:
    print("[✓] Data is gap free!")
else:
    print(f"[!] Found {len(midrange_gaps)} mid-range gaps (>1m and <20m). Showing first 20:")
    print(midrange_gaps[['Datetime', 'Time_Diff']].head(20))
    # Uncomment to enforce hard failure:
    # raise ValueError("Mid-range gaps detected in final output.")

# ----------------------------- #
#          SAVE CLEANED         #
# ----------------------------- #

out_df = combined_df.drop(columns=['Time_Diff', 'Contract'])
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"[✓] Cleaned data saved to: {OUTPUT_FILE}")
print("[i] Output timestamps are America/New_York (tz-aware).")

# ----------------------------- #
#   MAINTENANCE BREAK AUDIT     #
# ----------------------------- #

maintenance_break_audit(out_df)

# ----------------------------- #
#            PLOTS              #
# ----------------------------- #

# Histogram of time differences (seconds)
time_diff_sec = combined_df['Time_Diff'].dt.total_seconds().dropna()
plt.figure(figsize=(10, 4))
plt.hist(time_diff_sec, bins=100, edgecolor='black')
plt.title("Histogram of Time Differences Between Rows (NY time)")
plt.xlabel("Time Difference (seconds)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Timeline of time differences
plt.figure(figsize=(12, 4))
plt.plot(combined_df['Datetime'][1:], time_diff_sec, label='Time Diff (s)', alpha=0.7)
plt.axhline(y=60, linestyle='--', label='Expected 60s')
plt.axhline(y=3*24*3600, linestyle='--', label='3-Day Gap')
plt.title("Consecutive Time Gaps Over Time (NY)")
plt.ylabel("Seconds")
plt.legend()
plt.tight_layout()
plt.show()
