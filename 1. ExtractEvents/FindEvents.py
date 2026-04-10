import pandas as pd
import numpy as np
import os
import io
import warnings
from tqdm import tqdm
from contextlib import redirect_stdout

warnings.simplefilter(action='ignore', category=FutureWarning)

from ClassFunctions_new import precip_time_series, rainfall_analysis

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR       = '/scratch/hydro4/users/kv25483/MetricEvaluation/Data/'
THRESHOLD      = '11h'
RESOLUTIONS    = [5, 10, 30, 60]   # 5-min is always processed first (native detection)

OUTPUT_DIRS = {
    res: os.path.join(BASE_DIR, 'DanishRainData_Outputs', f'{res}mins_new_new')
    for res in RESOLUTIONS
}
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_directory(filename):
    """Determine input data directory from filename."""
    return 'DanishRainData_SVK' if 'svk' in filename else 'DanishRainData'


def get_pending_files():
    """
    Return files where at least one resolution output is still missing.
    Uses the raw CSV files as the source of truth.
    """
    files = []
    for directory in ['DanishRainData_SVK', 'DanishRainData']:
        input_dir = os.path.join(BASE_DIR, directory)
        if os.path.isdir(input_dir):
            files += [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    # Only include a file if at least one resolution output is missing
    pending = [
        f for f in files
        if any(
            not os.path.exists(os.path.join(OUTPUT_DIRS[res], f'All_events_{f}'))
            for res in RESOLUTIONS
        )
    ]
    return sorted(pending)


def build_and_save_metrics(ts: precip_time_series, filename: str, res: int,
                            all_five_min_indices: list = None):
    """
    Run rainfall_analysis on an already-initialised ts object and save the
    resulting metrics CSV to the appropriate output directory.

    For inherited (coarse) resolutions, flagged events get NaN metrics but
    still appear as rows so that event_num is cross-comparable across all
    resolution CSVs.

    Parameters
    ----------
    ts                   : precip_time_series  (events already detected/inherited)
    filename             : str                 original CSV filename
    res                  : int                 resolution in minutes
    all_five_min_indices : list or None
        Full list of original 5-min event indices. If provided, any indices
        missing from ts (due to flagging) are added as all-NaN rows so the
        output always has the same set of event_num values.
    """
    output_path = os.path.join(OUTPUT_DIRS[res], f'All_events_{filename}')

    # Identify which events are clean (can have metrics computed)
    flags        = ts.event_flags if ts.event_flags is not None else ['' for _ in ts.events]
    clean_mask   = [f == '' for f in flags]
    clean_events = [e for e, ok in zip(ts.events, clean_mask) if ok]
    clean_indices = [idx for idx, ok in zip(ts.original_event_indices, clean_mask) if ok]

    # Build a temporary ts-like object with only clean events for metric computation
    # We do this by temporarily overriding the events on ts, running analysis,
    # then restoring — avoids duplicating the whole class
    all_events    = ts.events
    all_indices   = ts.original_event_indices
    all_flags     = flags
    all_raw       = ts.raw_events
    all_norm      = ts.normalised_events
    all_dblnorm   = ts.double_normalised_events
    all_dmcs      = ts.DMCs
    all_dmcs_100  = ts.DMCs_100

    ts.events                = clean_events
    ts.original_event_indices = clean_indices
    ts.event_flags           = ['' for _ in clean_events]
    ts.raw_events             = None
    ts.normalised_events      = None
    ts.double_normalised_events = None
    ts.DMCs                   = None
    ts.DMCs_100               = None

    analysis = rainfall_analysis(THRESHOLD, ts)
    analysis.get_metrics()
    df_clean = pd.DataFrame(analysis.metrics)
    df_clean['event_num']    = clean_indices
    df_clean['start_time']   = [e[0] for e in clean_events]
    df_clean['end_time']     = [e[1] for e in clean_events]
    df_clean['mapping_flag'] = ''

    # Restore ts to its full state
    ts.events                = all_events
    ts.original_event_indices = all_indices
    ts.event_flags           = all_flags
    ts.raw_events             = all_raw
    ts.normalised_events      = all_norm
    ts.double_normalised_events = all_dblnorm
    ts.DMCs                   = all_dmcs
    ts.DMCs_100               = all_dmcs_100

    # Build NaN rows for flagged events
    flagged_rows = []
    for event, idx, flag in zip(all_events, all_indices, all_flags):
        if flag != '':
            row = {col: np.nan for col in df_clean.columns}
            row['event_num']    = idx
            row['start_time']   = event[0]
            row['end_time']     = event[1]
            row['mapping_flag'] = flag
            flagged_rows.append(row)

    df = pd.concat(
        [df_clean] + ([pd.DataFrame(flagged_rows)] if flagged_rows else []),
        ignore_index=True
    ).sort_values('event_num').reset_index(drop=True)

    df['gauge_num'] = filename.split('_')[0]
    df.to_csv(output_path, index=False)
    n_flagged = len(flagged_rows)
    print(f"  Saved {len(df)} events ({n_flagged} flagged) → {output_path}")

# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(filename):
    """
    Process one gauge file at all resolutions.

    Workflow
    --------
    1. Detect events natively at 5-min resolution.
    2. Save 5-min metrics CSV.
    3. For each coarser resolution (10, 30, 60 min):
       a. Build a new precip_time_series at that resolution.
       b. Inherit event boundaries from the 5-min ts object.
       c. Save metrics CSV.
    4. Move to the next gauge.

    All print output is captured and returned as a log string so that
    parallel workers do not produce interleaved console output.

    Returns
    -------
    (filename, status_dict, log)
      status_dict maps each resolution to its outcome string.
    """
    directory  = get_directory(filename)
    input_path = os.path.join(BASE_DIR, directory, filename)

    statuses = {res: 'pending' for res in RESOLUTIONS}
    buf = io.StringIO()

    try:
        with redirect_stdout(buf):

            if not os.path.exists(input_path):
                for res in RESOLUTIONS:
                    statuses[res] = f'error: input file not found ({input_path})'
                return filename, statuses, buf.getvalue()

            if pd.read_csv(input_path).empty:
                for res in RESOLUTIONS:
                    statuses[res] = 'empty file'
                return filename, statuses, buf.getvalue()

            # ----------------------------------------------------------
            # Step 1 — 5-min native detection
            # ----------------------------------------------------------
            res_5 = 5
            out_5 = os.path.join(OUTPUT_DIRS[res_5], f'All_events_{filename}')

            ts_5min = precip_time_series(input_path, temp_res=res_5)

            if ts_5min.data['precipitation (mm/min)'].lt(0).any():
                for res in RESOLUTIONS:
                    statuses[res] = 'negative values in raw data'
                return filename, statuses, buf.getvalue()

            ts_5min.pad_and_resample()
            ts_5min.get_events(threshold=THRESHOLD)

            if not ts_5min.events:
                for res in RESOLUTIONS:
                    statuses[res] = 'no events found at 5-min'
                return filename, statuses, buf.getvalue()

            if not os.path.exists(out_5):
                build_and_save_metrics(ts_5min, filename, res_5)
                statuses[res_5] = f'success ({len(ts_5min.events)} events)'
            else:
                statuses[res_5] = 'skipped (already exists)'
                print(f"  5-min output already exists, skipping save")

            # ----------------------------------------------------------
            # Step 2 — coarser resolutions inherited from 5-min ts
            # ----------------------------------------------------------
            for res in [r for r in RESOLUTIONS if r != 5]:
                out_path = os.path.join(OUTPUT_DIRS[res], f'All_events_{filename}')

                if os.path.exists(out_path):
                    statuses[res] = 'skipped (already exists)'
                    print(f"  {res}-min output already exists, skipping")
                    continue

                print(f"\n  --- {res}-min ---")
                ts_coarse = precip_time_series(input_path, temp_res=res)
                ts_coarse.pad_and_resample()
                ts_coarse.inherit_events_from(ts_5min)

                build_and_save_metrics(ts_coarse, filename, res)
                n_flagged = sum(1 for f in ts_coarse.event_flags if f != '')
                statuses[res] = (f'success ({len(ts_coarse.events)} events, '
                                 f'{n_flagged} flagged)')

    except Exception as e:
        msg = f'error: {e}'
        for res in RESOLUTIONS:
            if statuses[res] == 'pending':
                statuses[res] = msg
        return filename, statuses, buf.getvalue()

    return filename, statuses, buf.getvalue()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
pending = get_pending_files()
print(f"Resolutions : {RESOLUTIONS}")
print(f"Files to process: {len(pending)}")

#  --- Test mode: run first 3 files sequentially ---
# test_files = pending[5:10]
# results = []
# for f in test_files:
#     print(f"\n{'='*60}")
#     print(f"Processing: {f}")
#     filename, statuses, log = process_file(f)
#     results.append((filename, statuses, log))

#     # Print captured log
#     if log.strip():
#         for line in log.strip().splitlines():
#             print(f"  {line}")

#     # Print per-resolution outcomes
#     for res, status in statuses.items():
#         print(f"  [{res}-min] {status}")

# # --- Uncomment to run all files in parallel ---
from multiprocessing import Pool
N_WORKERS = 4
with Pool(processes=N_WORKERS) as pool:
    results = list(tqdm(
        pool.imap_unordered(process_file, pending),
        total=len(pending),
        desc='Processing'
    ))
    
    
# --- Summary ---
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for filename, statuses, _ in results:
    print(f"\n{filename}")
    for res, status in statuses.items():
        print(f"  [{res}-min] {status}")
    