import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# =========================
# CONFIG — edit paths
# =========================
INPUT_DIR  = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Data_Medium_Resampled"
OUTPUT_DIR = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Feature_SandPit/Liq"

# Base + TF files (auto-discovered from INPUT_DIR)
BASE_1M_FILE = "resampled_1m.csv"

# Feature params
DECAY = 0.9        # for purge_direction_purges_eff
PURGE_WINDOW = 5   # number of recent events to decay over
SPAN_EVENTS = 5    # number of last events to measure timespan over (on 1m eff timeline)

# =========================
# Helpers
# =========================
def _ensure_dt_tz(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Parse datetime with UTC then convert to America/New_York (DST-preserving)."""
    if col not in df.columns:
        # tolerate 'DateTime'
        alt = "DateTime"
        if alt in df.columns:
            col = alt
        else:
            raise ValueError("No datetime column found")
    df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert("America/New_York")
    if col != "datetime":
        df.rename(columns={col: "datetime"}, inplace=True)
    return df

def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # standardize OHLCV to lower case names if present
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"open", "high", "low", "close", "volume"} and c != lc:
            ren[c] = lc
    if ren:
        df = df.rename(columns=ren)
    needed = {"datetime", "open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[["datetime", "open", "high", "low", "close", "volume"]].copy()

def detect_tf_purges(tf: pd.DataFrame) -> List[dict]:
    """
    Detect purge events on a TF series (already sorted by datetime).
    Returns a list of dicts with:
      {
        'tf_idx': i,                # index in TF df
        'tf_dt': tf['datetime'][i], # TF bar timestamp
        'sign': +1 for low purge / -1 for high purge,
        'level': price level of purged swing,
        'window_start': tf['datetime'][i],
        'window_end': next TF bar datetime (exclusive) or +inf for last bar
      }
    Swing definition: simple 3-bar pattern using prior detected swings:
      - Keep queues of unbroken swing highs/lows. A purge occurs when
        current high > any stored swing high (bearish purge = -1),
        or current low < any stored swing low (bullish purge = +1).
    """
    df = tf.reset_index(drop=True)
    # swings from simple local extrema at this TF (N=1)
    df["swing_high"] = (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
    df["swing_low"]  = (df["low"]  < df["low"].shift(1))  & (df["low"]  < df["low"].shift(-1))

    swing_highs: List[Tuple[int, float]] = []
    swing_lows:  List[Tuple[int, float]] = []
    events: List[dict] = []

    n = len(df)
    for i in range(n):
        hi = df.at[i, "high"]
        lo = df.at[i, "low"]

        # store new swings
        if bool(df.at[i, "swing_high"]):
            swing_highs.append((i, float(df.at[i, "high"])))
        if bool(df.at[i, "swing_low"]):
            swing_lows.append((i, float(df.at[i, "low"])))

        # check purges against stored swings (consume purged ones)
        new_highs = []
        for idx, level in swing_highs:
            if hi > level:
                events.append({
                    "tf_idx": i,
                    "tf_dt": df.at[i, "datetime"],
                    "sign": -1,           # purge above swing high (bearish sweep)
                    "level": level
                })
            else:
                new_highs.append((idx, level))
        swing_highs = new_highs

        new_lows = []
        for idx, level in swing_lows:
            if lo < level:
                events.append({
                    "tf_idx": i,
                    "tf_dt": df.at[i, "datetime"],
                    "sign": +1,           # purge below swing low (bullish sweep)
                    "level": level
                })
            else:
                new_lows.append((idx, level))
        swing_lows = new_lows

    # attach TF candle windows for mapping to 1m
    for k, ev in enumerate(events):
        i = ev["tf_idx"]
        start = df.at[i, "datetime"]
        if i + 1 < n:
            end = df.at[i + 1, "datetime"]
        else:
            end = pd.Timestamp.max.tz_localize("UTC").tz_convert("America/New_York")
        events[k]["window_start"] = start
        events[k]["window_end"] = end

    return events

def map_events_to_1m(events: List[dict], base_1m: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    Map TF purge events to the exact 1m bar index in base_1m where the level was actually breached.
    Returns a list of tuples (minute_index, sign) in 1m df order.
    Mapping rule:
      - Look within [window_start, window_end) in 1m data
      - For sign = -1 (high purge): first minute where high > level
      - For sign = +1 (low purge):  first minute where low  < level
      - If not found in window, drop the event (conservative).
    """
    m1 = base_1m.reset_index(drop=True)
    # indexes for quick slicing by time window
    dt = m1["datetime"]

    mapped: List[Tuple[int, int]] = []
    for ev in events:
        ws, we = ev["window_start"], ev["window_end"]
        level, sign = ev["level"], ev["sign"]

        # slice 1m rows in window
        mask = (dt >= ws) & (dt < we)
        if not mask.any():
            continue
        seg = m1.loc[mask, ["high", "low"]]
        # we need absolute positional indices
        seg_idx = np.flatnonzero(mask.values)
        found_idx = None
        if sign == -1:
            # need first bar with high > level
            hit = np.flatnonzero((seg["high"].values > level))
            if hit.size:
                found_idx = seg_idx[int(hit[0])]
        else:
            # sign == +1: first bar with low < level
            hit = np.flatnonzero((seg["low"].values < level))
            if hit.size:
                found_idx = seg_idx[int(hit[0])]

        if found_idx is not None:
            mapped.append((int(found_idx), int(sign)))

    # sort by minute index to ensure chronological order
    mapped.sort(key=lambda t: t[0])
    return mapped

def build_eff_series_from_events(base_1m: pd.DataFrame,
                                 mapped_events: List[Tuple[int, int]],
                                 tf_label: str) -> pd.DataFrame:
    """
    From mapped events (minute_index, sign), build per-minute 1m-aligned features for this TF:
      - {tf}_purge_intensity_eff: count of events at that minute
      - {tf}_purge_direction_purges_eff: decay over last PURGE_WINDOW events (by sign)
      - {tf}_purge_timespan_eff: span in 1m candles between oldest and newest among the last up to SPAN_EVENTS events
    """
    df = base_1m[["datetime"]].copy().reset_index(drop=True)
    n = len(df)

    intensity = np.zeros(n, dtype=np.int16)

    # For direction + timespan, we need the event history
    event_indices: List[int] = []
    event_signs: List[int] = []

    dir_series = np.zeros(n, dtype=np.float32)
    span_series = np.full(n, np.nan, dtype=np.float32)

    # group events by minute (multiple events could occur in same minute)
    by_min: Dict[int, List[int]] = {}
    for idx, s in mapped_events:
        by_min.setdefault(idx, []).append(s)

    # walk minute by minute and update
    for i in range(n):
        if i in by_min:
            # record events at this minute
            for s in by_min[i]:
                event_indices.append(i)
                event_signs.append(s)

        # intensity at minute i
        intensity[i] = len(by_min.get(i, []))

        # direction (decay over last PURGE_WINDOW events)
        recent = event_signs[-PURGE_WINDOW:]
        val = 0.0
        for s in recent:
            val = val * DECAY + float(s)
        dir_series[i] = val

        # timespan over last up to SPAN_EVENTS events: difference in minute indices
        if len(event_indices) >= 2:
            # take up to last SPAN_EVENTS events
            last_k = event_indices[-SPAN_EVENTS:] if len(event_indices) >= SPAN_EVENTS else event_indices
            span = last_k[-1] - last_k[0]  # minutes between oldest and newest of that slice
            span_series[i] = float(span)

    out = df.copy()
    out[f"{tf_label}_purge_intensity_eff"] = intensity
    out[f"{tf_label}_purge_direction_purges_eff"] = dir_series
    out[f"{tf_label}_purge_timespan_eff"] = span_series
    return out

# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load base 1m
    base_path = os.path.join(INPUT_DIR, BASE_1M_FILE)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base 1m file not found: {base_path}")

    base = pd.read_csv(base_path)
    base = _ensure_dt_tz(base, "datetime")
    base = _standardize_ohlcv(base).sort_values("datetime").reset_index(drop=True)

    # discover TF files (exclude 1m)
    tf_files = []
    for fname in os.listdir(INPUT_DIR):
        if fname.startswith("resampled_") and fname.endswith(".csv") and fname != BASE_1M_FILE:
            tf_label = fname.replace("resampled_", "").replace(".csv", "")
            tf_files.append((tf_label, os.path.join(INPUT_DIR, fname)))

    eff_slices: List[pd.DataFrame] = []

    for tf_label, tf_path in sorted(tf_files):
        # ---- Load TF data
        tf = pd.read_csv(tf_path)
        tf = _ensure_dt_tz(tf, "datetime")
        tf = _standardize_ohlcv(tf).sort_values("datetime").reset_index(drop=True)

        # ---- Detect TF-level purge events (with TF windows + levels)
        tf_events = detect_tf_purges(tf)

        # ---- Map events to exact 1m minute where breach occurs
        mapped = map_events_to_1m(tf_events, base)

        # ---- Build 1m-aligned eff features (intensity, direction_purges, timespan)
        eff = build_eff_series_from_events(base, mapped, tf_label=tf_label)

        # ---- Save per-TF CSV
        out_tf = os.path.join(OUTPUT_DIR, f"liq_eff_{tf_label}_base1m.csv")
        eff.to_csv(out_tf, index=False)
        print(f"[OK] Wrote {tf_label} eff → {out_tf}")

        eff_slices.append(eff)

    # ---- Build combined eff-only file across TFs
    final = base[["datetime"]].copy()
    for sl in eff_slices:
        final = pd.merge_asof(
            final.sort_values("datetime"),
            sl.sort_values("datetime"),
            on="datetime",
            direction="backward",
            suffixes=("", ""),
        )

    final_out = os.path.join(OUTPUT_DIR, "liq_mtf_eff_combined_base1m.csv")
    final.to_csv(final_out, index=False)
    print(f"[OK] Wrote combined eff-only file → {final_out}")

if __name__ == "__main__":
    main()
