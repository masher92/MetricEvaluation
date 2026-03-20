import pandas as pd
import numpy as np
import os
import io
import pickle
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from contextlib import redirect_stdout

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
parser = argparse.ArgumentParser(description='Rainfall metric processor')
parser.add_argument('--temp-res', type=int, default=5,
                    help='Temporal resolution in minutes (5, 10, 30, or 60)')
args, _ = parser.parse_known_args()

from ClassFunctions import precip_time_series, rainfall_analysis

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR  = '/scratch/hydro4/users/kv25483/MetricEvaluation/Data/'
TEMP_RES  = 60  #args.temp_res   # 5=native detection; 10/30/60=inherit from 5-min pickle
THRESHOLD = '11h'
N_WORKERS = 4

PICKLE_DIR = os.path.join(BASE_DIR, 'DanishRainDataPickles')
OUTPUT_DIR = os.path.join(BASE_DIR, 'DanishRainData_Outputs', f'{TEMP_RES}mins_new')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_directory(filename):
    """Determine input data directory from filename."""
    return 'DanishRainData_SVK' if 'svk' in filename else 'DanishRainData'


def get_pending_files():
    """
    Build the list of files that still need processing at this resolution.

    For 5-min: iterates raw CSV files directly.
    For coarser: iterates existing 5-min pickles (which define what's available).
    """
    if TEMP_RES == 5:
        # Collect all CSV files across both data directories
        files = []
        for directory in ['DanishRainData_SVK', 'DanishRainData']:
            input_dir = os.path.join(BASE_DIR, directory)
            if os.path.isdir(input_dir):
                files += [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        pending = [
            f for f in files
            if not os.path.exists(os.path.join(OUTPUT_DIR, f'All_events_{f}'))
        ]
    else:
        # Use existing 5-min pickles as the source of truth
        pickle_files = os.listdir(PICKLE_DIR)
        pending = [
            p.replace('.pkl', '') for p in pickle_files
            if not os.path.exists(
                os.path.join(OUTPUT_DIR, f'All_events_{p.replace(".pkl", "")}'))
        ]

    return sorted(pending)

# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(filename):
    """
    Process one gauge file at TEMP_RES resolution.

    At 5-min: detects events natively from the raw CSV.
    At coarser resolutions: inherits events from the 5-min pickle.

    All internal print output is captured and returned as a log string
    so that parallel workers don't produce interleaved console output.

    Returns
    -------
    (filename, status, log)
    """
    directory   = get_directory(filename)
    input_path  = os.path.join(BASE_DIR, directory, filename)
    output_path = os.path.join(OUTPUT_DIR, f'All_events_{filename}')
    pickle_path = os.path.join(PICKLE_DIR, f'{filename}.pkl')

    if os.path.exists(output_path):
        return filename, 'skipped', ''

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):

            # ----------------------------------------------------------
            # For coarser resolutions, check the 5-min pickle has events
            # before doing anything else
            # ----------------------------------------------------------
            if TEMP_RES != 5:
                if not os.path.exists(pickle_path):
                    return filename, 'no 5-min pickle found', buf.getvalue()
                with open(pickle_path, 'rb') as f:
                    five_min_ts = pickle.load(f)
                if not five_min_ts.events:
                    return filename, 'skipped (no 5-min events)', buf.getvalue()

            # ----------------------------------------------------------
            # Basic data check
            # ----------------------------------------------------------
            if not os.path.exists(input_path):
                return filename, f'error: input file not found ({input_path})', buf.getvalue()

            if pd.read_csv(input_path).empty:
                return filename, 'empty file', buf.getvalue()

            # ----------------------------------------------------------
            # Build time series
            # ----------------------------------------------------------
            reference_pickle = pickle_path if TEMP_RES != 5 else None

            ts = precip_time_series(
                input_path,
                temp_res=TEMP_RES,
                reference_pickle=reference_pickle
            )

            if ts.data['precipitation (mm/min)'].lt(0).any():
                return filename, 'negative values', buf.getvalue()

            ts.pad_and_resample()

            # Check for missing timesteps
            dt_index   = ts.data.index
            full_range = pd.date_range(
                start=dt_index.min(), end=dt_index.max(), freq=f'{TEMP_RES}T')
            missing = full_range.difference(dt_index)
            if len(missing) > 0:
                print(f"Warning: {len(missing)} missing timesteps after resampling")

            # ----------------------------------------------------------
            # Event detection / inheritance happens inside rainfall_analysis
            # ----------------------------------------------------------
            analysis = rainfall_analysis(THRESHOLD, ts)

            if not ts.events:
                return filename, 'no events found', buf.getvalue()

            # ----------------------------------------------------------
            # Save 5-min pickle (only in native mode)
            # ----------------------------------------------------------
            if TEMP_RES == 5:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(ts, f, protocol=4)

            # ----------------------------------------------------------
            # Compute and save metrics
            # ----------------------------------------------------------
            analysis.get_metrics()
            df = pd.DataFrame(analysis.metrics)
            df['gauge_num']  = filename.split('_')[0]
            df['start_time'] = [e[0] for e in ts.events]
            df['end_time']   = [e[1] for e in ts.events]

            # Preserve link back to original 5-min event index
            if ts.original_event_indices is not None:
                df['event_num'] = ts.original_event_indices

            df.to_csv(output_path, index=False)
            return filename, f'success ({len(df)} events)', buf.getvalue()

    except Exception as e:
        return filename, f'error: {e}', buf.getvalue()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

pending = get_pending_files()
mode    = 'native (5-min detection)' if TEMP_RES == 5 else f'inherited from 5-min pickle'
print(f"Resolution : {TEMP_RES} min  [{mode}]")
print(f"Files to process: {len(pending)}")

with Pool(processes=N_WORKERS) as pool:
    results = list(tqdm(
        pool.imap_unordered(process_file, pending),
        total=len(pending),
        desc='Processing'
    ))

# ---------------------------------------------------------------------------
# Summary — printed cleanly once all workers are done
# ---------------------------------------------------------------------------

errors  = [(f, s, log) for f, s, log in results if s.startswith('error')]
skipped = [(f, s, log) for f, s, log in results if s == 'skipped']
success = [(f, s, log) for f, s, log in results if s.startswith('success')]

print(f"\nSuccess: {len(success)}  |  Skipped: {len(skipped)}  |  Errors: {len(errors)}")
print()

# Per-file logs grouped neatly
for filename, status, log in sorted(results, key=lambda x: x[0]):
    if status == 'skipped':
        continue
    print(f"--- {filename} : {status} ---")
    if log.strip():
        for line in log.strip().splitlines():
            print(f"    {line}")
    print()

# Errors highlighted at the end
if errors:
    print("ERRORS:")
    for filename, status, log in errors:
        print(f"  {filename}: {status}")
        if log.strip():
            for line in log.strip().splitlines():
                print(f"      {line}")
