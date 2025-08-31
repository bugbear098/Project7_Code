import pandas as pd
import numpy as np
import os

# --- Paths ---
script_dir = os.path.dirname(__file__)
input_file = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Concat_1m_Data', 'concatenated_1m.csv'))
output_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'Data', 'Data_Medium_Resampled'))
os.makedirs(output_dir, exist_ok=True)

# --- Load 1m data ---
df = pd.read_csv(input_file)
df.columns = [c.lower() for c in df.columns]

# Expect columns: datetime, open, high, low, close, volume
if 'datetime' not in df.columns:
    # adapt if your file uses 'datetime' spelled differently
    raise ValueError(f"Expected a 'datetime' column. Found: {list(df.columns)}")

# Parse to NY timezone
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/New_York')
df.set_index('datetime', inplace=True)
df = df.sort_index()

# Ensure numeric
for c in ['open','high','low','close','volume']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Optional: drop duplicate timestamps, keep last
df = df[~df.index.duplicated(keep='last')]

# --- Resampling helper ---
def resample_ohlc(session_df: pd.DataFrame, rule: str, origin='start_day', offset=None) -> pd.DataFrame:
    """
    label='left' makes the timestamp equal the bar START.
    closed='left' makes bins left-closed/right-open: [start, end)
    """
    agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    rs = session_df.resample(
        rule,
        label='left',
        closed='left',
        origin=origin,
        offset=offset
    ).agg(agg)
    # Drop bars with no data
    rs = rs.dropna(subset=['open','high','low','close'])
    return rs

# --- Timeframes (with anchors) ---
# 5m/15m/1H aligned to :00 automatically with origin='start_day'
# 4H anchored so bars start 02:00, 06:00, 10:00 ... ET via offset='2H'
# 1D anchored to 18:00 ET via offset='18H'
timeframes = {
    '5m':  dict(rule='5min', origin='start_day', offset=None),
    '15m': dict(rule='15min', origin='start_day', offset=None),
    '1H':  dict(rule='1H', origin='start_day', offset=None),
    '4H':  dict(rule='4H', origin='start_day', offset='2H'),
    '1D':  dict(rule='1D', origin='start_day', offset='18H'),
}

# --- Resample all ---
resampled = {}
for name, cfg in timeframes.items():
    resampled[name] = resample_ohlc(df, **cfg)
    out_path = os.path.join(output_dir, f"resampled_{name}.csv")
    resampled[name].to_csv(out_path)
    print(f"Saved {name} -> {out_path}")

# --- Validators ---
def check_hourly_alignment(df_1m: pd.DataFrame, rs_5m: pd.DataFrame, rs_15m: pd.DataFrame, rs_1h: pd.DataFrame, atol=1e-8, examples=5):
    """
    For each exact hour timestamp present in the 1m data, compare:
      open(1m@HH:00) == open(5m@HH:00) == open(15m@HH:00) == open(1H@HH:00)
    (Because we used label='left', each bar is stamped at its start.)
    """
    # Get exact top-of-hour stamps present in 1m data
    one_min_hours = df_1m.index[(df_1m.index.minute == 0) & (df_1m.index.second == 0)]
    common = one_min_hours.intersection(rs_5m.index).intersection(rs_15m.index).intersection(rs_1h.index)

    mismatches = []
    for ts in common:
        v1 = df_1m.at[ts, 'open']
        v5 = rs_5m.at[ts, 'open'] if ts in rs_5m.index else np.nan
        v15 = rs_15m.at[ts, 'open'] if ts in rs_15m.index else np.nan
        v1h = rs_1h.at[ts, 'open'] if ts in rs_1h.index else np.nan

        # Compare within tolerance
        ok = np.isfinite([v1, v5, v15, v1h]).all() and \
             np.allclose([v1, v5, v15, v1h], v1, atol=atol, rtol=0)

        if not ok:
            mismatches.append((ts, v1, v5, v15, v1h))
            if len(mismatches) >= examples:
                break

    print(f"[Check] Hourly open alignment: checked {len(common)} timestamps.")
    if mismatches:
        print(f"[Fail] Found {len(mismatches)} mismatches (showing up to {examples}):")
        for ts, v1, v5, v15, v1h in mismatches:
            print(f"  {ts}  1m:{v1}  5m:{v5}  15m:{v15}  1H:{v1h}")
    else:
        print("[Pass] All matching at top-of-hour.")

def check_anchor_alignment(rs_4h: pd.DataFrame, rs_1d: pd.DataFrame, atol=1e-8, examples=5):
    """
    Ensure 4H bars start at 02:00, 06:00, ... ET
    Ensure 1D bars start at 18:00 ET
    """
    mismatches_4h = [ts for ts in rs_4h.index if not (ts.minute==0 and ts.second==0 and ts.hour%4==2)]
    print(f"[Check] 4H anchor @02:00 ET: {len(rs_4h)} bars, {len(mismatches_4h)} off-anchor.")
    if mismatches_4h:
        for ts in mismatches_4h[:examples]:
            print(f"  Off-anchor 4H start: {ts}")

    mismatches_1d = [ts for ts in rs_1d.index if not (ts.hour==18 and ts.minute==0 and ts.second==0)]
    print(f"[Check] 1D anchor @18:00 ET: {len(rs_1d)} bars, {len(mismatches_1d)} off-anchor.")
    if mismatches_1d:
        for ts in mismatches_1d[:examples]:
            print(f"  Off-anchor 1D start: {ts}")

# Run checks
check_hourly_alignment(
    df_1m=df,
    rs_5m=resampled['5m'],
    rs_15m=resampled['15m'],
    rs_1h=resampled['1H'],
    atol=1e-8
)

check_anchor_alignment(
    rs_4h=resampled['4H'],
    rs_1d=resampled['1D'],
    atol=1e-8
)
