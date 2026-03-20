import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
import datetime
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

home_dir = '/scratch/hydro4/users/kv25483/MetricEvaluation/Data/'

# ---------------------------------------------------------------------------
# RainfallEvent: thin wrapper so each event knows its own type
# ---------------------------------------------------------------------------

class RainfallEvent:
    """
    Wraps a DataFrame of rainfall values and records whether the data are
    raw (mm/timestep), DMC-incremental (dimensionless, already diffed), or
    double-normalised-incremental (dimensionless, already diffed).

    All event types store INCREMENTAL values, never cumulative.
    The index is:
      - raw      : DatetimeIndex
      - dblnorm  : float 0→1  (normalised time)
      - dmc      : float 0→1  (normalised time)
    """

    RAW     = 'raw'
    DBLNORM = 'dblnorm'
    DMC     = 'dmc'

    def __init__(self, df: pd.DataFrame, kind: str, temp_res_minutes: float = None):
        assert kind in (self.RAW, self.DBLNORM, self.DMC), f"Unknown kind: {kind}"
        self.df = df
        self.kind = kind
        self.temp_res_minutes = temp_res_minutes

    @property
    def values(self) -> np.ndarray:
        return self.df.iloc[:, 0].to_numpy()

    @property
    def index(self):
        return self.df.index

    def __len__(self):
        return len(self.df)

    @property
    def is_raw(self):
        return self.kind == self.RAW

    @property
    def is_dimensionless(self):
        return self.kind in (self.DBLNORM, self.DMC)

    @property
    def duration_hours(self) -> float:
        if self.is_raw:
            duration = (self.df.index[-1] - self.df.index[0]).total_seconds() / 3600
            return duration if duration > 0 else np.nan
        return 1.0

    @property
    def timestep_hours(self) -> float:
        if self.is_raw and self.temp_res_minutes:
            return self.temp_res_minutes / 60.0
        return 1.0 / max(len(self) - 1, 1)

    @property
    def peak_position_ratio(self) -> float:
        """Position of peak intensity as fraction of event duration (0–1)."""
        peak_idx = np.argmax(self.values)
        return peak_idx / max(len(self.values) - 1, 1)

    @property
    def time_to_peak(self):
        """
        For raw events: minutes to peak.
        For dimensionless events: normalised position 0–1.
        """
        if self.is_raw:
            peak_ts = self.df.iloc[:, 0].idxmax()
            return (peak_ts - self.df.index[0]).total_seconds() / 60
        return self.peak_position_ratio

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.values)

    @property
    def cumulative_normalised(self) -> np.ndarray:
        cum = self.cumulative
        total = cum[-1]
        return cum / total if total > 0 else cum


# ---------------------------------------------------------------------------
# precip_time_series
# ---------------------------------------------------------------------------

class precip_time_series:
    """
    Loads and processes a precipitation time series.

    Parameters
    ----------
    data_path        : str
        Path to the raw CSV file.
    temp_res         : int
        Temporal resolution in minutes for resampling.
    reference_pickle : str or None
        If provided, events are inherited from this pre-existing 5-min pickle
        (matched to the coarser resolution) rather than detected from scratch.
        Pass the full path to the .pkl file.
    """

    def __init__(self, data_path: str, temp_res: int, reference_pickle: str = None):
        self.temp_res = temp_res
        self.reference_pickle = reference_pickle

        self.data, self.statid = self._read_raw_data(data_path)

        self.padded = False
        self.events = None
        self.original_event_indices = None  # only populated in inherited mode

        self.raw_events:              list[RainfallEvent] = None
        self.normalised_events:       list[RainfallEvent] = None
        self.double_normalised_events:list[RainfallEvent] = None
        self.DMCs:                    list[RainfallEvent] = None
        self.DMCs_100:                list[RainfallEvent] = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _read_raw_data(self, raw_data_file_path: str):
        precip = pd.read_csv(raw_data_file_path, encoding='ISO-8859-1', index_col=0)
        precip.rename(columns={'precip_past1min': 'precipitation (mm/min)'}, inplace=True)
        precip = precip[precip['precipitation (mm/min)'] != -9999]
        precip = precip[precip['precipitation (mm/min)'] != -999999.0]
        precip.reset_index(inplace=True)
        precip['timeobs'] = precip['timeobs'].apply(
            lambda x: datetime.datetime.fromtimestamp(x))
        precip.set_index('timeobs', inplace=True)

        filename  = raw_data_file_path.split('/')[-1]
        station_id = filename.split('_')[0]
        print(station_id)

        precip = self._resolve_duplicates(precip)
        precip = self._fill_in_missing_vals(precip)

        qc = pd.read_csv(home_dir + 'Other/station_exclusion_periods_from_climadb.csv')
        qc.rename(columns={'the_date': 'start_time', 'hour': 'end_time'}, inplace=True)
        qc_this_station = qc[qc['statid'] == int(station_id)]
        precip = self._run_quality_control(precip, qc_this_station)
        precip.loc[precip['QC_fail'], 'precipitation (mm/min)'] = np.nan
        precip.index = pd.to_datetime(precip.index)
        del precip['QC_fail']

        return precip, station_id

    def _resolve_duplicates(self, df):
        dup_df  = df[df.index.duplicated(keep=False)]
        grouped = dup_df.groupby(dup_df.index)
        differ_count = sum(
            1 for _, group in grouped
            if len(group) > 1 and not group.eq(group.iloc[0]).all().all()
        )
        print(f"Differing duplicate rows: {differ_count}")
        return df[~df.index.duplicated(keep='first')]

    def _fill_in_missing_vals(self, df):
        full_range = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq='1T')
        df.index = pd.to_datetime(df.index).floor('T')
        return df.reindex(full_range).fillna(0)

    def _run_quality_control(self, df, qc_this_station):
        df_filtered = df.copy()
        df_filtered['QC_fail'] = False
        for i in range(len(qc_this_station)):
            start_time      = pd.Timestamp(qc_this_station.iloc[i]['start_time'])
            end_time        = qc_this_station.iloc[i]['end_time']
            extra_start     = start_time - pd.Timedelta(hours=1)
            time_list       = pd.to_datetime(
                pd.date_range(start=extra_start, end=end_time, freq='T').to_list())
            df_filtered.loc[df_filtered.index.isin(time_list), 'QC_fail'] = True
        return df_filtered

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def pad_and_resample(self, pad_value=0):
        freq       = f'{self.temp_res}min'
        self.data  = self.data.resample(freq).sum().fillna(pad_value)
        self.padded = True

    # ------------------------------------------------------------------
    # Event detection — public entry point
    # ------------------------------------------------------------------

    def get_events(self, threshold, min_duration=None, min_precip=1):
        """
        Detect or inherit rainfall events.

        In native mode (reference_pickle=None):
            Detects events directly from the data using a rolling-window
            dry-period approach. Uses `threshold` as the inter-event dry period.

        In inherited mode (reference_pickle provided):
            Loads events from the 5-min pickle and maps their boundaries
            onto the coarser-resolution data by flooring start times and
            ceiling end times.

        Parameters
        ----------
        threshold    : str  e.g. '11h' — only used in native mode
        min_duration : int (minutes) — defaults to 3 timesteps if not set
        min_precip   : float (mm)
        """
        if min_duration is None:
            min_duration = self.temp_res * 3

        if not self.padded:
            self.pad_and_resample()

        if self.reference_pickle:
            self._init_events_from_pickle()
            self._trim_leading_trailing_zeroes()
        else:
            self._init_events_from_data(threshold)

        self._filter_events_by_length(min_duration)
        self._filter_events_by_amount(min_precip)

    # ------------------------------------------------------------------
    # Native event detection
    # ------------------------------------------------------------------

    def _init_events_from_data(self, threshold):
        """Detect events directly using rolling-window dry-period method."""
        precip = self.data
        time_delta        = precip.index[1] - precip.index[0]
        threshold_minutes = pd.to_timedelta(threshold).total_seconds() / 60
        window_size       = int(threshold_minutes // (time_delta.total_seconds() / 60))

        precip_sum = precip.rolling(window=window_size, min_periods=1).sum()
        precip_sum['precipitation (mm/min)'] = precip_sum[
            'precipitation (mm/min)'].apply(lambda x: 0 if abs(x) < 1e-10 else x)
        valid_count = precip.rolling(window_size).count()
        is_dry      = (precip_sum == 0) & (valid_count == window_size)

        events       = []
        in_event     = False
        current_start = None

        for i in range(1, len(precip)):
            prev_dry = is_dry.iloc[i - 1, 0]
            curr_dry = is_dry.iloc[i, 0]
            if not curr_dry and prev_dry and not in_event:
                current_start = precip.index[i]
                in_event = True
            elif curr_dry and not prev_dry and in_event:
                current_end = precip.index[i] - pd.to_timedelta(threshold)
                events.append((current_start, current_end))
                in_event = False

        if in_event:
            for i in reversed(range(len(precip))):
                if precip.iloc[i, 0] != 0:
                    events.append((current_start, precip.index[i]))
                    break

        self.events = events
        # No original_event_indices in native mode
        self.original_event_indices = list(range(len(events)))
        print(f"Detected {len(events)} events (native mode, threshold={threshold})")

    # ------------------------------------------------------------------
    # Inherited event detection (coarse resolution from 5-min pickle)
    # ------------------------------------------------------------------

    def _init_events_from_pickle(self):
        """
        Load events from the 5-min reference pickle and map their boundaries
        onto the coarser-resolution data (floor start, ceil end).
        """
        with open(self.reference_pickle, 'rb') as f:
            five_min_pickle = pickle.load(f)
        five_min_events = five_min_pickle.events
        print(f"Loaded {len(five_min_events)} events from 5-min pickle")

        precip = self.data.sort_index()
        events           = []
        original_indices = []

        for i, (start_5m, end_5m) in enumerate(five_min_events):
            # Floor start to nearest previous coarse timestep
            pos_start = precip.index.get_indexer([start_5m], method='pad')[0]
            if pos_start == -1:
                continue
            bin_start = precip.index[pos_start]

            # Ceil end to nearest following coarse timestep
            pos_end = precip.index.get_indexer([end_5m], method='backfill')[0]
            if pos_end == -1:
                continue
            bin_end = precip.index[pos_end]

            events.append((bin_start, bin_end))
            original_indices.append(i)

        self.events                = events
        self.original_event_indices = original_indices
        print(f"Matched {len(events)} events at {self.temp_res}-min resolution")

    # ------------------------------------------------------------------
    # Event filtering (shared by both modes)
    # ------------------------------------------------------------------

    def _filter_events_by_length(self, min_duration):
        filtered_events  = []
        filtered_indices = []
        for event, idx in zip(self.events, self.original_event_indices):
            duration = (event[1] - event[0]).total_seconds() / 60
            if duration >= min_duration:
                filtered_events.append(event)
                filtered_indices.append(idx)
        self.events                = filtered_events
        self.original_event_indices = filtered_indices

    def _filter_events_by_amount(self, min_precip):
        filtered_events  = []
        filtered_indices = []
        for event, idx in zip(self.events, self.original_event_indices):
            total = self.data.loc[event[0]:event[1]]['precipitation (mm/min)'].sum()
            if total >= min_precip:
                filtered_events.append(event)
                filtered_indices.append(idx)
        self.events                = filtered_events
        self.original_event_indices = filtered_indices

    def _trim_leading_trailing_zeroes(self, min_length=3):
        """
        Remove leading and trailing zero-precipitation timesteps from each
        event. Only used in inherited mode, where coarse boundaries may
        overshoot the actual rainfall.
        """
        print("Trimming leading/trailing zeros...")
        initial_count  = len(self.events)
        trimmed_events  = []
        trimmed_indices = []
        skipped_dry     = 0
        skipped_short   = 0
        shortened       = 0

        for i, (start, end) in enumerate(self.events):
            event_data    = self.data.loc[start:end]
            precip_values = event_data['precipitation (mm/min)'].values
            non_zero      = np.nonzero(precip_values)[0]

            if len(non_zero) == 0:
                skipped_dry += 1
                continue

            start_idx = non_zero[0]
            end_idx   = non_zero[-1] + 1
            trimmed   = event_data.iloc[start_idx:end_idx]

            if len(trimmed) < min_length:
                skipped_short += 1
                continue

            if start_idx > 0 or end_idx < len(event_data):
                shortened += 1

            trimmed_events.append((trimmed.index[0], trimmed.index[-1]))
            trimmed_indices.append(self.original_event_indices[i])

        self.events                = trimmed_events
        self.original_event_indices = trimmed_indices

        print(f"  Before: {initial_count}  |  "
              f"Skipped (dry): {skipped_dry}  |  "
              f"Skipped (short): {skipped_short}  |  "
              f"Shortened: {shortened}  |  "
              f"Remaining: {len(trimmed_events)}")

    # ------------------------------------------------------------------
    # Build typed event lists
    # ------------------------------------------------------------------

    def create_raw_events(self, threshold=None):
        if self.events is None:
            self.get_events(threshold=threshold)
        self.raw_events = [
            RainfallEvent(
                self.data.loc[e[0]:e[1]],
                RainfallEvent.RAW,
                temp_res_minutes=self.temp_res
            )
            for e in self.events
        ]

    def create_double_normalised_events(self, threshold=None):
        if self.events is None:
            self.get_events(threshold=threshold)
        self.double_normalised_events = [
            RainfallEvent(
                self._make_dblnorm_incremental(self.data.loc[e[0]:e[1]].values),
                RainfallEvent.DBLNORM
            )
            for e in self.events
        ]

    def create_normalised_events(self, threshold=None):
        if self.events is None:
            self.get_events(threshold=threshold)
        self.normalised_events = [
            RainfallEvent(
                self._make_norm_incremental(self.data.loc[e[0]:e[1]].values),
                RainfallEvent.DBLNORM
            )
            for e in self.events
        ]

    def create_DMCs(self, threshold=None):
        if self.events is None:
            self.get_events(threshold=threshold)
        if self.double_normalised_events is None:
            self.create_double_normalised_events(threshold)
        self.DMCs     = [self._make_dmc(e, n_bins=10)  for e in self.double_normalised_events]
        self.DMCs_100 = [self._make_dmc(e, n_bins=100) for e in self.double_normalised_events]

    # ------------------------------------------------------------------
    # Helpers for building event DataFrames
    # ------------------------------------------------------------------

    def _make_dblnorm_incremental(self, series: np.ndarray) -> pd.DataFrame:
        series    = series.flatten()
        cum       = np.cumsum(series)
        total     = cum[-1]
        norm_cum  = cum / total if total > 0 else np.zeros_like(cum, dtype=float)
        incremental = np.diff(norm_cum, prepend=0)
        time_norm   = np.linspace(0, 1, len(incremental))
        return pd.DataFrame({'normalised_rainfall': incremental}, index=time_norm)

    def _make_norm_incremental(self, series: np.ndarray) -> pd.DataFrame:
        series    = series.flatten()
        cum       = np.cumsum(series)
        total     = cum[-1]
        norm_cum  = cum / total if total > 0 else np.zeros_like(cum, dtype=float)
        incremental = np.diff(norm_cum, prepend=0)
        time_norm   = np.linspace(0, 1, len(incremental))
        return pd.DataFrame({'normalised_rainfall': incremental}, index=time_norm)

    def _make_dmc(self, event: 'RainfallEvent', n_bins: int) -> 'RainfallEvent':
        cum          = np.cumsum(event.values)
        target_pts   = np.linspace(0, 1, n_bins)
        time_norm    = np.linspace(0, 1, len(cum))
        interp_func  = interp1d(time_norm, cum, kind='linear', fill_value='extrapolate')
        interp_cum   = np.round(interp_func(target_pts), 6)
        incremental  = np.diff(interp_cum, prepend=0)
        df = pd.DataFrame({'DMC': incremental}, index=target_pts)
        return RainfallEvent(df, RainfallEvent.DMC)

    @staticmethod
    def interpolate_rainfall(dim_less_curve, bin_count):
        target_pts  = np.linspace(0, 1, bin_count + 1)
        norm_time   = np.linspace(0, 1, len(dim_less_curve))
        interp_func = interp1d(norm_time, dim_less_curve,
                               kind='linear', fill_value='extrapolate')
        return interp_func(target_pts)

    # ------------------------------------------------------------------
    # Return helpers
    # ------------------------------------------------------------------

    def return_specific_event(self, event_idx):
        return self.data.loc[self.events[event_idx][0]:self.events[event_idx][1]]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_all_events(self):
        plt.figure(figsize=(20, 10))
        plt.plot(self.data.index, self.data.values)
        for dates in self.events:
            plt.vlines(dates[0], colors='green', linestyles='--', ymin=0, ymax=3)
            plt.vlines(dates[1], colors='red',   linestyles='--', ymin=0, ymax=3)
        plt.legend(['Precipitation', 'Event start', 'Event end'])
        plt.ylabel('[mm]')
        plt.title('Precip data with events')

    def plot_specific_event(self, event_idx):
        time_delta         = self.data.index[1] - self.data.index[0]
        time_delta_minutes = time_delta.seconds / 60
        event = self.data.loc[self.events[event_idx][0]:self.events[event_idx][1]]
        plt.figure()
        plt.bar(event.index - time_delta, event.values[:, 0],
                width=pd.Timedelta(minutes=time_delta_minutes), align='edge')
        plt.legend(['Precipitation'])
        plt.title(f'Event {event_idx}')


# ---------------------------------------------------------------------------
# rainfall_analysis
# ---------------------------------------------------------------------------

class rainfall_analysis:

    def __init__(self, threshold, ts: precip_time_series):
        self.ts       = ts
        self.temp_res = ts.temp_res
        self.metrics  = {}

        if not ts.padded:
            ts.pad_and_resample()
        if ts.events is None:
            ts.get_events(threshold=threshold)
        if ts.raw_events is None:
            ts.create_raw_events(threshold)
        if ts.normalised_events is None:
            ts.create_normalised_events(threshold)
        if ts.double_normalised_events is None:
            ts.create_double_normalised_events(threshold)
        if ts.DMCs is None:
            ts.create_DMCs(threshold)

    # ==================================================================
    # Public entry point
    # ==================================================================

    def get_metrics(self):
        self._compute_raw_only_metrics()
        event_sets = {
            'raw':     self.ts.raw_events,
            'dmc':     self.ts.DMCs,
            'dblnorm': self.ts.double_normalised_events,
        }
        for label, events in event_sets.items():
            print(f"Computing metrics: {label}")
            self._compute_intensity_metrics(label, events)
            self._compute_timing_metrics(label, events)
            self._compute_shape_metrics(label, events)

    # ==================================================================
    # Raw-only metrics
    # ==================================================================

    def _compute_raw_only_metrics(self):
        raw = self.ts.raw_events
        self.metrics['total_precip'] = np.array([e.values.sum() for e in raw])

        if self.temp_res > 30:
            self.metrics['I30'] = np.full(len(raw), np.nan)
        else:
            window_size = int(30 / self.temp_res)
            self.metrics['I30'] = np.array([
                e.df.rolling(window=window_size, min_periods=1).sum().max().values[0] / 0.5
                for e in raw
            ])

    # ==================================================================
    # Intensity metrics
    # ==================================================================

    def _compute_intensity_metrics(self, label: str, events: list):
        s     = f'_{label}'
        res_h = self._res_hours(events[0])
        vals  = [e.values for e in events]

        self.metrics[f'max_intensity{s}']  = np.array([v.max()  / res_h for v in vals])
        self.metrics[f'min_intensity{s}']  = np.array([v.min()  / res_h for v in vals])
        self.metrics[f'mean_intensity{s}'] = np.array([
            e.values.sum() / e.duration_hours for e in events])
        self.metrics[f'std{s}']      = np.array([v.std()  / res_h for v in vals])
        self.metrics[f'skewness{s}'] = np.array([skew(v, bias=False)     for v in vals])
        self.metrics[f'kurtosis{s}'] = np.array([kurtosis(v, bias=False)  for v in vals])

        if label == 'raw':
            self.metrics[f'cv{s}'] = (
                self.metrics[f'std{s}'] / self.metrics[f'mean_intensity{s}'])

        self.metrics[f'relative_amp{s}'] = (
            (self.metrics[f'max_intensity{s}'] - self.metrics[f'min_intensity{s}'])
            / self.metrics[f'mean_intensity{s}'])
        self.metrics[f'peak_mean_ratio{s}'] = (
            self.metrics[f'max_intensity{s}'] / self.metrics[f'mean_intensity{s}'])
        self.metrics[f'ni{s}'] = self.metrics[f'peak_mean_ratio{s}']

        self.metrics[f'gini{s}']            = np.array([self._gini_coef(e.values)        for e in events])
        self.metrics[f'lorenz_asymmetry{s}'] = np.array([self._lorentz_asymmetry(e.values) for e in events])
        self.metrics[f'PCI{s}']             = np.array([self._calculate_pci(e.values)    for e in events])

        temp = np.array([self._high_low_zone_indicators(e) for e in events])
        self.metrics[f'% time HIZ{s}']         = temp[:, 0]
        self.metrics[f'% time LIZ{s}']         = temp[:, 1]
        self.metrics[f'% rain HIZ{s}']         = temp[:, 2]
        self.metrics[f'Mean Intensity HIZ{s}'] = temp[:, 3]

    # ==================================================================
    # Timing metrics
    # ==================================================================

    def _compute_timing_metrics(self, label: str, events: list):
        s = f'_{label}'

        self.metrics[f'time_to_peak{s}']       = np.array([e.time_to_peak        for e in events], dtype='float64')
        self.metrics[f'peak_position_ratio{s}'] = np.array([e.peak_position_ratio for e in events], dtype='float64')
        self.metrics[f'duration{s}']            = np.array([
            len(e) * self.temp_res if e.is_raw else len(e)
            for e in events])

        ppr = self.metrics[f'peak_position_ratio{s}']

        self.metrics[f'third_ppr{s}']  = np.select(
            [ppr < 0.4, (ppr >= 0.4) & (ppr <= 0.6), ppr > 0.6], [0, 1, 2])
        self.metrics[f'3rd_w_peak{s}'] = np.select(
            [ppr < 0.33, (ppr >= 0.33) & (ppr <= 0.66), ppr > 0.66], [0, 1, 2])
        self.metrics[f'4th_w_peak{s}'] = np.select(
            [ppr < 0.25,
             (ppr >= 0.25) & (ppr < 0.5),
             (ppr >= 0.5)  & (ppr < 0.75),
             ppr >= 0.75], [0, 1, 2, 3])
        self.metrics[f'5th_w_peak{s}'] = np.select(
            [ppr < 0.2,
             (ppr >= 0.2) & (ppr < 0.4),
             (ppr >= 0.4) & (ppr < 0.6),
             (ppr >= 0.6) & (ppr < 0.8),
             ppr >= 0.8], [0, 1, 2, 3, 4])

        for pct, name in [(0.25, 'T25'), (0.50, 'T50'), (0.75, 'T75'), (0.50, 'D50')]:
            self.metrics[f'{name}{s}'] = np.array([
                self._calc_dX(e.values, pct, cumulative=e.cumulative_normalised)
                for e in events])

        self.metrics[f'time_skewness{s}'] = self._compute_time_based_skewness(events)
        self.metrics[f'time_kurtosis{s}'] = self._compute_time_based_kurtosis(events)
        self.metrics[f'time_std{s}']      = self._compute_time_based_std(events)

    # ==================================================================
    # Shape metrics
    # ==================================================================

    def _compute_shape_metrics(self, label: str, events: list):
        s = f'_{label}'

        self.metrics[f'TCI{s}']          = np.array([self._calculate_tci(e.values)            for e in events])
        self.metrics[f'asymm_d{s}']      = np.array([self._calculate_event_asymmetry(e.values) for e in events])
        self.metrics[f'Event Loading{s}'] = np.array([self._calculate_event_loading(e)          for e in events])
        self.metrics[f'NRMSE_P{s}']      = np.array([self._calculate_nrmse_peak(e)             for e in events])
        self.metrics[f'skewp{s}']        = np.array([self._calculate_skew_p(e)                 for e in events])

        temp = np.array([self._find_heaviest_run_half(e.values) for e in events])
        self.metrics[f'heaviest_half{s}'] = temp[:, 0]

        self.metrics[f'intermittency{s}']   = np.array([self._compute_intermittency(e.values) for e in events])
        self.metrics[f'event_dry_ratio{s}'] = np.array([self._event_dry_ratio(e.values)       for e in events])

        self.metrics[f'centre_gravity{s}']             = np.array([self._compute_rcg(e.values)             for e in events])
        self.metrics[f'centre_gravity_interpolated{s}'] = np.array([self._compute_rcg_interpolated(e.values) for e in events])

        temp = np.array([self._compute_mass_dist_indicators(e.values, use_interpolation=False) for e in events])
        for i, name in enumerate(['m1', 'm2', 'm3', 'm4', 'm5']):
            self.metrics[f'{name}{s}'] = temp[:, i]

        temp = np.array([self._compute_mass_dist_indicators(e.values, use_interpolation=True) for e in events])
        for i, name in enumerate(['m1_wi', 'm2_wi', 'm3_wi', 'm4_wi', 'm5_wi']):
            self.metrics[f'{name}{s}'] = temp[:, i]

        temp = np.array([self._frac_in_quarters(e.values, interpolate=True) for e in events])
        for i, name in enumerate(['frac_q1_wi', 'frac_q2_wi', 'frac_q3_wi', 'frac_q4_wi']):
            self.metrics[f'{name}{s}'] = temp[:, i]

        self.metrics[f'3rd_w_most{s}'] = np.array([self._nth_with_most(e.values, n=3) for e in events])
        self.metrics[f'4th_w_most{s}'] = np.array([self._nth_with_most(e.values, n=4) for e in events])
        self.metrics[f'5th_w_most{s}'] = np.array([self._nth_with_most(e.values, n=5) for e in events])

        self.metrics[f'3rd_ARR{s}'] = np.array([self._calc_ARR_thirds(e.values)  for e in events])
        self.metrics[f'3rd_rcg{s}'] = np.array([self._thirds_rcg(e.values)       for e in events])

        temp = np.array([self._classify_BSC(e.values) for e in events])
        self.metrics[f'BSC{s}']       = temp[:, 0]
        self.metrics[f'BSC_Index{s}'] = temp[:, 1].astype(int)

    # ==================================================================
    # Metric implementation — all operate on incremental numpy arrays
    # ==================================================================

    @staticmethod
    def _res_hours(event: RainfallEvent) -> float:
        if event.is_raw and event.temp_res_minutes:
            return event.temp_res_minutes / 60.0
        return 1.0

    @staticmethod
    def _compute_intermittency(series: np.ndarray) -> float:
        series = np.asarray(series).flatten()
        wet    = series > 0
        return (wet[:-1] != wet[1:]).sum() / len(series)

    @staticmethod
    def _event_dry_ratio(series: np.ndarray) -> float:
        series = np.round(np.asarray(series).flatten(), 6)
        return np.count_nonzero(series == 0) / len(series) * 100

    @staticmethod
    def _compute_rcg(series: np.ndarray) -> float:
        series = np.asarray(series).flatten()
        total  = series.sum()
        if total == 0:
            return np.nan
        positions = np.arange(len(series))
        return int(np.round(np.sum(positions * series) / total)) / len(series)

    @staticmethod
    def _compute_rcg_interpolated(series: np.ndarray) -> float:
        series = np.asarray(series).flatten()
        n      = len(series)
        if n < 2:
            return np.nan
        total = series.sum()
        if total == 0:
            return np.nan
        return np.sum(np.arange(n) * series) / total / (n - 1)

    @staticmethod
    def _calculate_pci(series: np.ndarray) -> float:
        series = np.asarray(series).flatten()
        total  = series.sum()
        return 0.0 if total == 0 else (np.sum(series ** 2) / total ** 2) * 100

    @staticmethod
    def _gini_coef(series: np.ndarray) -> float:
        series = np.asarray(series).flatten()
        n      = len(series)
        if n == 0 or np.all(series == 0):
            return 0.0
        mean_val = np.mean(series)
        if mean_val == 0:
            return 0.0
        # O(n log n) sorted version — avoids the O(n²) matrix approach
        s = np.sort(series)
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * s) / (n * s.sum())) - (n + 1) / n

    @staticmethod
    def _lorentz_asymmetry(series: np.ndarray) -> float:
        series = np.round(np.asarray(series).flatten(), 6)
        if np.all(series == series[0]):
            return np.nan
        n    = len(series)
        mean = np.mean(series)
        lower = series[series < mean]
        upper = series[series > mean]
        if len(lower) == 0 or len(upper) == 0:
            return np.nan
        m    = len(lower)
        x_m  = lower.max()
        x_m1 = upper.min()
        delta  = (mean - x_m) / (x_m1 - x_m)
        return (m + delta) / n + (lower.mean() + delta * x_m1) / np.sum(series)

    def _high_low_zone_indicators(self, event: RainfallEvent) -> np.ndarray:
        series   = event.values / self._res_hours(event)
        mean_i   = series.mean()
        above    = np.where(series > mean_i)[0]
        below    = np.where(series < mean_i)[0]
        frac_hi  = len(above) / len(series) * 100
        frac_lo  = len(below) / len(series) * 100
        rain_hi  = series[above].sum() / series.sum() * 100 if series.sum() > 0 else 0
        mean_hi  = series[above].mean() if len(above) > 0 else 0
        return np.array([frac_hi, frac_lo, rain_hi, mean_hi])

    def _compute_time_based_skewness(self, events: list) -> np.ndarray:
        result = []
        for e in events:
            v, pos = self._time_positions(e)
            total  = v.sum()
            if total == 0:
                result.append(np.nan)
                continue
            t_cg  = np.sum(pos * v) / total
            sigma = np.sqrt(np.sum(((pos - t_cg) ** 2) * v) / total)
            if sigma == 0:
                result.append(np.nan)
                continue
            result.append(np.sum(((pos - t_cg) ** 3) * v) / (total * sigma ** 3))
        return np.array(result)

    def _compute_time_based_kurtosis(self, events: list) -> np.ndarray:
        result = []
        for e in events:
            v, pos = self._time_positions(e)
            total  = v.sum()
            if total == 0 or np.any(np.isnan(v)):
                result.append(np.nan)
                continue
            t_cg    = np.sum(pos * v) / total
            sigma_sq = np.sum(((pos - t_cg) ** 2) * v) / total
            if sigma_sq == 0 or np.isnan(sigma_sq):
                result.append(np.nan)
                continue
            sigma = np.sqrt(sigma_sq)
            result.append(np.sum(((pos - t_cg) ** 4) * v) / (total * sigma ** 4))
        return np.array(result)

    def _compute_time_based_std(self, events: list) -> np.ndarray:
        result = []
        for e in events:
            v, pos = self._time_positions(e)
            total  = v.sum()
            if total == 0:
                result.append(np.nan)
                continue
            t_cg = np.sum(pos * v) / total
            result.append(np.sqrt(np.sum(((pos - t_cg) ** 2) * v) / total))
        return np.array(result)

    def _time_positions(self, event: RainfallEvent):
        """
        Returns (values, positions).
        Raw events: positions in minutes.
        Dimensionless events: positions normalised 0–1.
        """
        v = event.values.flatten()
        n = len(v)
        if event.is_raw:
            positions = np.arange(1, n + 1) * self.temp_res
        else:
            positions = np.linspace(0, 1, n)
        return v, positions

    @staticmethod
    def _calc_dX(series: np.ndarray, percentile: float,
                 cumulative: np.ndarray = None) -> float:
        cum      = cumulative if cumulative is not None else np.cumsum(series) / series.sum()
        n        = len(cum)
        time_pct = np.linspace(0, 100, n)
        target   = percentile

        below = np.where(cum < target)[0]
        above = np.where(cum >= target)[0]

        if len(below) > 0 and len(above) > 0:
            x1, y1 = time_pct[below[-1]], cum[below[-1]]
            x2, y2 = time_pct[above[0]],  cum[above[0]]
        elif len(below) == 0 and cum[0] >= target:
            step   = 100 / (n - 1) if n > 1 else 100
            x1, y1 = 0, 0
            x2, y2 = step, cum[0]
        else:
            return np.nan

        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.nan
        return x1 + (target - y1) / slope if slope else np.nan

    def _calc_ARR_thirds(self, series: np.ndarray) -> int:
        cum     = np.cumsum(series)
        cum_norm = cum / cum[-1]
        t = self._calc_dX(series, 0.5, cumulative=cum_norm)
        if t is None or np.isnan(t):
            return np.nan
        if t < 40:   return 1
        if t <= 60:  return 2
        return 3

    def _thirds_rcg(self, series: np.ndarray) -> int:
        rcg = self._compute_rcg_interpolated(series)
        if rcg < 1 / 3:  return 1
        if rcg < 2 / 3:  return 2
        return 3

    @staticmethod
    def _classify_BSC(series: np.ndarray):
        cum      = np.cumsum(series)
        total    = cum[-1]
        cum_norm = cum / total if total > 0 else cum
        n        = len(cum_norm)
        q1, q2, q3 = round(n * 0.25), round(n * 0.50), round(n * 0.75)
        actual   = [cum_norm[min(q1, n-1)], cum_norm[min(q2, n-1)],
                    cum_norm[min(q3, n-1)], cum_norm[-1]]
        expected = [0.25, 0.50, 0.75, 1.0]
        code     = ''.join(['1' if a >= e else '0' for a, e in zip(actual, expected)])
        weights  = [3, 1, -1, -3]
        fli      = sum(w * int(b) for w, b in zip(weights, code))
        return np.array([code, fli])

    @staticmethod
    def _compute_mass_dist_indicators(series: np.ndarray,
                                       use_interpolation: bool = True) -> np.ndarray:
        series   = pd.Series(np.round(np.asarray(series).flatten(), 6))
        steps    = len(series)
        peak_idx = int(np.argmax(series))
        cum      = np.cumsum(series)
        total    = cum.iloc[-1]

        m1 = (cum[peak_idx] / total if peak_idx == 0
               else cum[peak_idx] / (total - cum[peak_idx - 1]))
        m2 = series[peak_idx] / total

        if use_interpolation:
            time_norm  = np.linspace(0, 1, steps)
            cum_interp = interp1d(time_norm, cum / total,
                                  kind='linear', fill_value='extrapolate')
            m3 = float(cum_interp(1 / 3))
            m4 = float(cum_interp(0.3))
            m5 = float(cum_interp(0.5))
        else:
            m3 = cum[int(np.round(steps / 3)) - 1]    / total
            m4 = cum[int(np.round(steps * 0.3)) - 1]  / total
            m5 = cum[int(np.round(steps / 2)) - 1]    / total

        return np.array([m1, m2, m3, m4, m5])

    @staticmethod
    def _frac_in_quarters(series: np.ndarray, interpolate: bool = True,
                           target_length: int = 20) -> np.ndarray:
        series = np.asarray(series).flatten()
        total  = series.sum()
        if total == 0:
            return np.array([np.nan] * 4)
        if interpolate:
            x_old = np.linspace(0, 1, len(series))
            x_new = np.linspace(0, 1, target_length)
            s     = np.interp(x_new, x_old, series)
            q     = target_length // 4
            return np.array([round(s[i*q:(i+1)*q].sum() / s.sum() * 100, 1)
                              for i in range(4)])
        n = len(series)
        q1, q2, q3 = n // 4, n // 2, 3 * n // 4
        slices = [series[:q1], series[q1:q2], series[q2:q3], series[q3:]]
        if len(set(len(sl) for sl in slices)) > 1:
            return np.array([np.nan] * 4)
        return np.array([round(sl.sum() / total * 100, 1) for sl in slices])

    @staticmethod
    def _nth_with_most(series: np.ndarray, n: int) -> int:
        cum         = np.cumsum(series)
        target_pts  = np.linspace(0, 1, n + 1)
        time_norm   = np.linspace(0, 1, len(cum))
        interp_func = interp1d(time_norm, cum, kind='linear', fill_value='extrapolate')
        return int(np.argmax(np.diff(interp_func(target_pts))))

    @staticmethod
    def _calculate_event_asymmetry(series: np.ndarray) -> float:
        series = np.asarray(series).flatten()
        n      = len(series)
        if n < 3:
            return np.nan
        ranks = stats.rankdata(series)
        U     = (ranks - 0.5) / n
        diff  = U[:-2] - U[2:]
        denom = np.mean(diff ** 2) ** (3 / 2)
        return np.mean(diff ** 3) / denom if denom != 0 else np.nan

    def _calculate_skew_p(self, event: RainfallEvent) -> float:
        series = event.values.flatten()
        n      = len(series)
        t      = np.arange(n) * self.temp_res if event.is_raw else np.linspace(0, 1, n)
        t_peak = t[np.argmax(series)]
        t_end  = t[-1] if t[-1] != 0 else 1.0
        return np.mean(((t - t_peak) / t_end) ** 3)

    def _calculate_event_loading(self, event: RainfallEvent) -> float:
        series    = np.round(event.values.flatten(), 6)
        if np.all(series == 0):
            return np.nan
        intensity = series / self._res_hours(event)

        def sth(s):
            m = np.mean(s)
            return np.nan if m == 0 else np.std(s) / m

        peak_idx = np.argmax(intensity)
        rising   = intensity[:peak_idx + 1]
        mirrored = np.concatenate([rising, rising[::-1][1:]])
        sth_orig = sth(intensity)
        sth_mirr = sth(mirrored)
        if np.isnan(sth_orig) or sth_orig == 0:
            return np.nan
        return ((sth_mirr - sth_orig) / sth_orig) * 100

    def _calculate_nrmse_peak(self, event: RainfallEvent) -> float:
        res_min = self.temp_res if event.is_raw else 1.0
        series  = np.round(event.values.flatten() / res_min, 6)
        ppeak   = series.max()
        P       = series.sum()
        if P == 0:
            return np.nan
        rmse = np.sqrt(np.sum((series - ppeak) ** 2) / len(series))
        return round(rmse / P, 2)

    @staticmethod
    def _calculate_tci(series: np.ndarray) -> float:
        series = np.asarray(series).flatten()
        T      = len(series)
        P      = series.sum()
        if P == 0:
            return 0.0
        tci_values = []
        for center in range(T):
            remaining = list(range(T))
            remaining.remove(center)
            remaining.sort(key=lambda i: (abs(i - center), -series[i]))
            order = [center] + remaining
            cum   = np.cumsum(series[order])
            cum_t = np.arange(1, T + 1)
            actual = np.trapz(cum, cum_t)
            ref    = 0.5 * T * P
            tci_values.append((actual - ref) / ref)
        return max(tci_values)

    @staticmethod
    def _find_heaviest_run_half(series: np.ndarray, threshold: float = 0.01) -> tuple:
        series = np.asarray(series).flatten()
        thresh = series.max() * threshold
        above  = series > thresh

        run_ids      = np.zeros(len(series), dtype=int)
        run_ids[1:]  = (above[1:] != above[:-1]).cumsum()

        valid_runs = []
        for run_id in np.unique(run_ids[above]):
            idx = np.where(run_ids == run_id)[0]
            valid_runs.append((idx[0], idx[-1], series[idx].sum()))

        if not valid_runs:
            return None, None, None

        start_idx, end_idx, _ = max(valid_runs, key=lambda x: x[2])
        mid = len(series) // 2
        if start_idx <= mid <= end_idx:
            return 'both_halves', start_idx, end_idx
        elif end_idx < mid:
            return 'first_half', start_idx, end_idx
        return 'second_half', start_idx, end_idx

    # ==================================================================
    # Plotting
    # ==================================================================

    def plot_boxplots(self, metrics):
        n_plots = len(metrics)
        cols    = min(n_plots, 4)
        rows    = int(np.ceil(n_plots / 4))
        fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
        axes = np.ravel(axes)
        for i, metric in enumerate(metrics):
            axes[i].boxplot(self.metrics[metric])
            axes[i].set_title(metric)
        plt.tight_layout()

    def plot_histograms(self, metrics):
        n_plots = len(metrics)
        cols    = min(n_plots, 4)
        rows    = int(np.ceil(n_plots / 4))
        fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
        axes = np.ravel(axes)
        for i, metric in enumerate(metrics):
            axes[i].hist(self.metrics[metric])
            axes[i].set_title(metric)
        plt.tight_layout()
