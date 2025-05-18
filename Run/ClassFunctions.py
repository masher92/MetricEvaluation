import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

temp_res = 5

class precip_time_series:
    def __init__(self, data_path):
        print("Making the class so I am")
        self.data,self.statid = self.read_raw_data_as_pandas_df(data_path)
        
        self.padded = False

        self.events = None # first and last time step
         
        self.raw_events = None # the actual event
        
        self.normalised_events = None # event with intensity normalised
        
        self.double_normalised_events = None # event with both time and intensity normalised
        
        self.interpolated_events = None # 
        
        self.DMCs = None
        
        self.DMCs_100 = None
        
    def read_raw_data_as_pandas_df_rasmus(self,raw_data_file_path):
        # Read file with timestamp as index
        precip = pd.read_excel(raw_data_file_path,index_col=1)

        # Timestamps str -> datetime
        precip.index = pd.to_datetime(precip.index)

        # Save ID of station
        station_id = str(precip.station.iloc[0])

        # Remove column with station ID
        precip = precip.drop("station",axis=1)
        
        return precip,station_id
 
    def read_raw_data_as_pandas_df(self, raw_data_file_path):

        # Read file with timestamp as index
        precip = pd.read_csv(raw_data_file_path, encoding="ISO-8859-1",index_col=0)
        precip.rename(columns={'precip_past1min':'precipitation (mm/hr)'},inplace=True)

        precip = precip[precip['precipitation (mm/hr)'] !=-9999]
        precip = precip[precip['precipitation (mm/hr)'] !=-999999.0]
        precip.reset_index(inplace=True)
        precip['timeobs'] = precip['timeobs'].apply(lambda x: datetime.datetime.fromtimestamp(x))
        precip.set_index('timeobs', inplace=True)        
        
        # Save ID of station
        filename = raw_data_file_path.split('/')[-1]
        station_id = filename.split('_')[0]
        print(station_id)  # Output: 615600
        
        # Get rid of duplicates
        precip  = self.resolve_duplicates(precip)

        # Fill in rows which should have a 0 value
        precip = self.fill_in_missing_vals(precip)

        # Run QC
        qc = pd.read_csv(f"/nfs/a319/gy17m2a/PhD/datadir/station_exclusion_periods_from_climadb.csv")
        qc.rename(columns = {'the_date':'start_time', 'hour':'end_time'}, inplace=True)
        qc_this_station = qc[qc['statid'] == int(station_id)]
        precip = self.run_quality_control(precip, qc_this_station)
        precip.loc[precip['QC_fail'], 'precipitation (mm/hr)'] = np.nan
        # Timestamps str -> datetime
        precip.index = pd.to_datetime(precip.index)
        del precip['QC_fail']
        if len(precip[precip["precipitation (mm/hr)"]<0])>0 :
            print("Values less than 0")

        return precip,station_id   
    
    def resolve_duplicates(self, df):
        # This list will store flags for indices where duplicate rows differ.
        duplicate_flags = []

        # Get all rows with duplicated indices (all occurrences)
        dup_df = df[df.index.duplicated(keep=False)]

        # Group by the index
        grouped = dup_df.groupby(dup_df.index)

        # Iterate over each group (each duplicated index)
        for idx, group in grouped:
            if len(group) > 1:
                # Check if all rows are identical by comparing each to the first row.
                # group.eq(group.iloc[0]) returns a DataFrame of booleans.
                if group.eq(group.iloc[0]).all().all():
                    # They are identical: flag it and keep only one row.
                    # print(f"Index {idx} has duplicate rows that are identical. Keeping one.")
                    #duplicate_flags.append((idx, "identical"))
                    pass
                else:
                    # They differ: flag this index.
                    # print(f"Warning: Index {idx} has duplicate rows that differ!")
                    duplicate_flags.append((idx, "differ"))

        # Remove duplicate rows, keeping only the first occurrence for each index.
        print(f"duplicate flags length: {len(duplicate_flags)}")
        df_cleaned = df[~df.index.duplicated(keep='first')]
        return df_cleaned    
    
    def fill_in_missing_vals(self, df):
        # Get the full range of datetime values (1-minute intervals)
        start_time = df.index.min()
        end_time = df.index.max()
        full_range = pd.date_range(start=start_time, end=end_time, freq='1T')
        
        df.index = pd.to_datetime(df.index).floor('T')
        # Reindex the DataFrame to the complete range.
        # Missing rows will have NaN in the original columns.
        df_reindexed = df.reindex(full_range)
        df_reindexed = df_reindexed.fillna(0)

        # Add a new column 'QC_fail'
        # For rows that were missing in the original index, flag them as True.
        # df_reindexed['QC_fail'] = ~df_reindexed.index.isin(df.index)
        # df_reindexed = df.reindex(full_range).fillna(0)

        return df_reindexed    
    
    def run_quality_control(self, df, qc_this_station):
        df_filtered=df.copy()
        df_filtered['QC_fail'] = False
        
        for i in range(len(qc_this_station)):
            # Get the start and end time of that missing chunk
            start_time =  pd.Timestamp(qc_this_station.iloc[i]['start_time'])
            end_time = qc_this_station.iloc[i]['end_time']
            # Create an extra start time variable, to ensure the hour leading up to the first date is included
            extra_start_time = start_time - pd.Timedelta(hours=1)

            # Create a list of datetime values at 1-minute intervals
            time_list = pd.date_range(start=extra_start_time, end=end_time, freq="T").to_list()
            # Ensure time_list is a DatetimeIndex for efficient lookup
            time_list = pd.to_datetime(time_list)  # Convert to DatetimeIndex if needed

            # Remove bad values
            df_filtered.loc[df_filtered.index.isin(time_list), 'QC_fail'] = True   
            
        return df_filtered
    
    def pad_and_resample(self, temp_res ,pad_value = 0):
        # Resample the data to the specified frequency and pad missing values with pad_value
        freq = f'{temp_res}min'
        self.data = self.data.resample(freq).sum().fillna(pad_value)
        self.data *= 60 / int(temp_res)
        self.padded = True
        
    def return_specific_event(self,event_idx):

        # Size of timesteps
        time_delta = self.data.index[1] - self.data.index[0]
        time_delta_minutes = time_delta.seconds / 60

        # Extract event data
        event = self.data.loc[self.events[event_idx][0]:self.events[event_idx][1]]

        return event  
    
    def return_specific_double_normalised_event(self,event_idx):

        # Size of timesteps
        time_delta = self.data.index[1] - self.data.index[0]
        time_delta_minutes = time_delta.seconds / 60
        
        # Extract event data
        event = self.double_normalised_events[event_idx]

        return event    

    def return_specific_normalised_event(self,event_idx):

        # Size of timesteps
        time_delta = self.data.index[1] - self.data.index[0]
        time_delta_minutes = time_delta.seconds / 60
        
        # Extract event data
        event = self.normalised_events[event_idx]

        return event       
    
    def return_specific_interpolated_event(self,event_idx):
        
        # Extract event data
        event = self.interpolated_events[event_idx]

        return event     
        
    def get_events(self,threshold,min_duration = 30, min_precip = 1):
        
        if not self.padded:
            self.pad_and_resample()

        self.init_events(threshold)
        self.filter_events_by_length(min_duration)
        self.filter_events_by_amount(min_precip)
               

    def init_events(self, threshold):
        precip = self.data

        ########################################
        ##### Prep: Calculate time step and rolling window
        # This calculates how many data points are in your threshold.
        # If your data is at 5-min intervals, then '11h' = 132 steps.
        ########################################

        time_delta = precip.index[1] - precip.index[0]
        threshold_minutes = pd.to_timedelta(threshold).total_seconds() / 60
        window_size = int(threshold_minutes // (time_delta.total_seconds() / 60))

        ########################################
        #####  2. Rolling sum and rolling count
        # Precip_sum converts the column to one showing the total precip in the last 11h
        # Valid count makes a column showing the number of valid precip vals in the last 11h (i.e. not NAN)
        ########################################    
        precip_sum = precip.rolling(window_size).sum(min_periods=1)
        precip_sum['precipitation (mm/hr)'] = precip_sum['precipitation (mm/hr)'].apply(lambda x: 0 if abs(x) < 1e-10 else x)
        valid_count = precip.rolling(window_size).count()

        ########################################
        #####3. Identify dry periods
        # Dry if rolling sum is 0 and full data coverage
        ########################################
        is_dry = (precip_sum == 0) & (valid_count == window_size)

        ########################################
        #####4. Loop through to find event boundaries
        # Uses a state machine approach to find paired start/end times
        ########################################
        events = []
        in_event = False
        current_start = None

        for i in range(1, len(precip)):
            prev_dry = is_dry.iloc[i - 1, 0]
            curr_dry = is_dry.iloc[i, 0]

            # Start of a rain event (dry -> wet)
            if not curr_dry and prev_dry and not in_event:
                current_start = precip.index[i]
                in_event = True

            # End of a rain event (wet -> dry)
            elif curr_dry and not prev_dry and in_event:
                current_end = precip.index[i] - pd.to_timedelta(threshold)
                events.append((current_start, current_end))
                in_event = False

        ########################################
        #####5. Close last event if still raining at end
        ########################################
        if in_event:
            for i in reversed(range(len(precip))):
                if precip.iloc[i, 0] != 0:
                    events.append((current_start, precip.index[i]))
                    break

        ########################################        
        #####6. Store events in class attribute
        ########################################
        self.events = events


    def filter_events_by_length(self,min_duration):
        
        # Remove events with duration under min duration
        filtered_events = [event for event in self.events if event[1]-event[0]>=pd.Timedelta(minutes=min_duration)]

        # Update events
        self.events = filtered_events
    
    def filter_events_by_amount(self,min_precip):
        
        # Remove events with total precip under minimum
        filtered_events = [event for event in self.events if self.data.loc[event[0]:event[1]].sum().values[0]>=min_precip]
        
        # update events
        self.events = filtered_events
        
    def create_raw_events(self, threshold):

        # Make sure events have been computed
        if self.events == None:
            self.get_events(threshold=threshold)
        
        # Make list of nparrays containing the values of the normalised curve
        raw_events = [self.data.loc[event[0]:event[1]] for event in self.events]

        # Assign to global value
        self.raw_events = raw_events        

    def create_double_normalised_events(self, threshold):
        # Make sure events have been computed
        if self.events is None:
            self.get_events(threshold=threshold)

        # Make list of nparrays containing the values of the double-normalized curve
        double_normalised_events = [self.get_double_normalised_event(self.data.loc[event[0]:event[1]].values) for event in self.events]

        # Assign to global value
        self.double_normalised_events = double_normalised_events

    def create_normalised_events(self, threshold):

        # Make sure events have been computed
        if self.events == None:
            self.get_events(threshold=threshold)
        
        # Make list of nparrays containing the values of the normalised curve
        normalised_events = [self.get_normalised_event(self.data.loc[event[0]:event[1]].values) for event in self.events]

        # Assign to global value
        self.normalised_events = normalised_events
        
    def create_incremental_event(self, cumulative_rainfall_df):
        if cumulative_rainfall_df is None:
            return None

        # Calculate incremental rainfall by differencing, preserving index
        incremental_rainfall = np.diff(cumulative_rainfall_df, prepend=0)
        time_normalized = np.linspace(0, 1,len(incremental_rainfall))
        
        df =pd.DataFrame({'DMC': incremental_rainfall}, index=time_normalized)
        return df    

    def create_DMCs (self, threshold):
        # Make sure events have been computed
        if self.events is None:
            self.get_events(threshold=threshold)

        # Make list of nparrays containing the values of the double-normalized curve
        DMCs_10 = [self.get_interpolated_event(event, 10) for event in self.double_normalised_events]
        DMCS_10_incremental = [self.create_incremental_event(event['DMC']) for event in DMCs_10]
        self.DMCs = DMCS_10_incremental  
        
        DMCs_100 = [self.get_interpolated_event(event, 100) for event in self.double_normalised_events]
        DMCS_100_incremental = [self.create_incremental_event(event['DMC']) for event in DMCs_100]
        self.DMCs_100 = DMCS_100_incremental         
    
    def get_normalised_event(self,series):
    
        # Calculate cumulative rainfall
        cumulative_rainfall = np.cumsum(series)
        # cumulative_rainfall = np.append([0],cumulative_rainfall)

        # normalize
        normalized_cumulative_rainfall = cumulative_rainfall/cumulative_rainfall[-1]
        time_normalized = np.linspace(0, 1,len(normalized_cumulative_rainfall))
        normalised_df = pd.DataFrame({'normalised_rainfall':normalized_cumulative_rainfall}, index=time_normalized)
        
        return normalised_df

    def get_double_normalised_event(self, series):
        # Calculate cumulative rainfall
        cumulative_rainfall = np.cumsum(series)
        # cumulative_rainfall = np.append([0], cumulative_rainfall)

        # Normalize cumulative rainfall
        normalized_cumulative_rainfall = cumulative_rainfall / cumulative_rainfall[-1]

        # Normalize the time axis (assuming time is at regular intervals)
        time_normalized = np.linspace(0, 1,len(normalized_cumulative_rainfall))
        
        double_normalised_df = pd.DataFrame({'normalised_rainfall':normalized_cumulative_rainfall}, index=time_normalized)

        return double_normalised_df

    def get_interpolated_event(self, normalized_cumulative_rainfall, n):
        normalized_cumulative_rainfall = normalized_cumulative_rainfall['normalised_rainfall']
        # Define target points for bin_number bins
        target_points = np.linspace(0, 1, n)

        # Create interpolation function based on existing data points
        rainfall_times = np.array(range(0, len(normalized_cumulative_rainfall)))

        # Normalize time from 0 to 1
        normalized_time = (rainfall_times - rainfall_times[0]) / (rainfall_times[-1] - rainfall_times[0])
        interpolation_func = interp1d(normalized_time, normalized_cumulative_rainfall, kind='linear', fill_value="extrapolate")

        # Interpolate values at target points
        interpolated_values = interpolation_func(target_points)
        interpolated_values = np.round(interpolated_values, 6)
        interpolated_values_df =pd.DataFrame({'DMC':interpolated_values}, index=target_points)
        return interpolated_values_df        
    
    def interpolate_rainfall(self,dim_less_curve,bin):

        # Define target points for splits
        target_points = np.linspace(0, 1, bin+1)

        # normalized time
        normalized_time = np.linspace(0, 1,len(dim_less_curve))

        # Create interpolation function based on existing data points
        interpolation_func = interp1d(normalized_time, dim_less_curve, kind='linear', fill_value="extrapolate")

        # Interpolate values at target points
        interpolated_values = interpolation_func(target_points)

        return interpolated_values    
   
    def plot_all_events(self):
        plt.figure(figsize=(20,10))
        plt.plot(self.data.index,self.data.values)
        for i,dates in enumerate(self.events):
            plt.vlines(dates[0],colors="green",linestyles='--',ymin=0,ymax=3)
            plt.vlines(dates[1],colors="red",linestyles='--',ymin=0,ymax=3)
        plt.legend(["Precipitation","Event start","Event end"])
        plt.ylabel("[mm]")
        plt.title("Padded precip data, with events")

    def plot_specific_event(self,event_idx):

        # Size of timesteps
        time_delta = self.data.index[1]-self.data.index[0]
        time_delta_minuts = time_delta.seconds/60

        plt.figure()
        event = (self.data.loc[self.events[event_idx][0]:self.events[event_idx][1]])

        # plot were right edge align w timestamp
        plt.bar(event.index-time_delta,event.values[:,0],width=pd.Timedelta(minutes=time_delta_minuts),align="edge")

        plt.legend(["Precipitation"])
        plt.title(f"Event {event_idx}")

    def plot_specific_event_w_hist(self,event_idx):

        # Size of timesteps
        time_delta = self.data.index[1] - self.data.index[0]
        time_delta_minutes = time_delta.seconds / 60

        # Extract event data
        event = self.data.loc[self.events[event_idx][0]:self.events[event_idx][1]]

        # Create a figure with two subplots (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # **Plot time series (left subplot)**
        axes[0].bar(event.index - time_delta, event.values[:, 0], 
                    width=pd.Timedelta(minutes=time_delta_minutes), align="edge")
        axes[0].set_title(f"Event {event_idx}")
        axes[0].set_ylabel("Precipitation (mm)")
        axes[0].set_xlabel("Time")
        axes[0].legend(["Precipitation"])

        # **Plot histogram (right subplot)**
        axes[1].hist(event.values[:, 0], bins=10, edgecolor='black', alpha=0.7)
        axes[1].set_title("Precipitation Histogram")
        axes[1].set_xlabel("Precipitation (mm)")
        axes[1].set_ylabel("Frequency")

        # Adjust layout for clarity
        plt.tight_layout()

    def plot_specific_normalised_events(self,event_idxs):
        plt.figure()
        for idx in event_idxs:
            x_values = np.linspace(0,1,len(self.normalised_events[idx]))
            plt.plot(x_values,self.normalised_events[idx],label = f"Event: {idx+1}")
        plt.legend()
        plt.title("normalised events")
        
    def plot_specific_normalised_event(self,event_idx):
        plt.figure()
        x_values = np.linspace(0,1,len(self.normalised_events[event_idx]))
        plt.plot(x_values,self.normalised_events[event_idx],label = f"Event: {event_idx+1}")
        plt.legend()
        plt.title("normalised events")        

class rainfall_analysis:
    def __init__(self,threshold, ts: precip_time_series):
        self.ts = ts
        self.metrics = {} 
        
        # Prepere ts for analysis
        if not self.ts.padded:
            ts.pad_and_resample()

        if ts.events == None:
            ts.get_events(threshold=threshold)
            
        if ts.raw_events == None:
            ts.create_raw_events(threshold)  
            
        if ts.normalised_events == None:
            ts.create_normalised_events(threshold)
            
        if ts.double_normalised_events == None:
            ts.create_double_normalised_events(threshold)      
            
        if ts.DMCs == None:
            ts.create_DMCs(threshold)   
    
    def interpolate_rainfall(self,dim_less_curve,bin):

        # Define target points for splits
        target_points = np.linspace(0, 1, bin+1)

        # normalized time
        normalized_time = np.linspace(0, 1,len(dim_less_curve))

        # Create interpolation function based on existing data points
        interpolation_func = interp1d(normalized_time, dim_less_curve, kind='linear', fill_value="extrapolate")

        # Interpolate values at target points
        interpolated_values = interpolation_func(target_points)

        return interpolated_values            
        
    ################################################
    def compute_intermittency(self,series):
            
        total_timesteps = len(series)
        wet = series>0
        transistions = (wet[:-1] != wet[1:]).sum()

        intermittency = transistions/total_timesteps

        return intermittency 

    def compute_rcg_idx(self,series):
            
        # Culmitative sum
        cumulative_sum = np.cumsum(series)

        # Normalize
        cumulative_sum /= cumulative_sum[-1]

        #first idx over 0.5
        idx = np.argmax(cumulative_sum>0.5)

        return idx

    def compute_rcg(self, series, suffix):

        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)
            
        n = len(series)
        positions = np.arange(n)
        total_mass = series.sum()
        rcg_idx = int(np.round(np.sum(positions * series) / total_mass))
        rcg = rcg_idx / n
        return rcg
    
    def compute_rcg_interpolated(self, series, suffix):
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)

        n = len(series)
        if n < 2:
            return np.nan  # Not enough data to interpolate

        total_mass = series.sum()
        if total_mass == 0:
            print(f"Zero mass encountered in event: {series}")
            return np.nan  # Avoid division by zero

        positions = np.arange(n)
        weighted_sum = np.sum(positions * series)
        rcg = weighted_sum / total_mass / (n - 1)

        return rcg


    def calculate_pci(self, precip_series, suffix):
        """
        Calculate the Precipitation Concentration Index (PCI) for a rainfall event.

        Parameters:
        precip_series (array-like): Rainfall depth values at regular intervals (e.g., every 5 minutes)

        Returns:
        float: PCI value
        """
        if suffix in ['_dblnorm', '_norm']:
            precip_series = np.diff(precip_series, prepend=0)
            
        precip_series = np.array(precip_series)

        if precip_series.sum() == 0:
            return 0  # Avoid division by zero, PCI is undefined for zero rainfall

        numerator = np.sum(precip_series**2)
        denominator = np.sum(precip_series)**2

        pci = (numerator / denominator) * 100
        return pci    
    
    
    def compute_mass_dist_indicators(self, series, suffix, use_interpolation=True):
        """
        Computes mass distribution indicators for a rainfall event.

        Parameters:
            series (array-like): Rainfall series.
            suffix (str): Used to determine if diff should be applied.
            use_interpolation (bool): Whether to compute indicators using interpolation or not.

        Returns:
            np.array: [M1, M2, M3, M4, M5]
        """
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)
        series = pd.Series(series)

        series = np.round(series, 6)
        steps = len(series)
        peak_idx = np.argmax(series)
        cumulative_sum = np.cumsum(series)

        total = cumulative_sum.iloc[-1]

        # M1: mass before peak
        if peak_idx == 0:
            m1 = cumulative_sum[peak_idx] / total
        else:
            m1 = cumulative_sum[peak_idx] / (total - cumulative_sum[peak_idx - 1])

        # M2: peak as fraction of total
        m2 = series[peak_idx] / total

        if use_interpolation:
            # Normalized time for interpolation
            time_norm = np.linspace(0, 1, steps)
            cum_interp = interp1d(time_norm, cumulative_sum / total, kind='linear', fill_value="extrapolate")
            m3 = float(cum_interp(1/3))
            m4 = float(cum_interp(0.3))
            m5 = float(cum_interp(0.5))
        else:
            m3 = cumulative_sum[int(np.round(steps / 3)) - 1] / total
            m4 = cumulative_sum[int(np.round(steps * 0.3)) - 1] / total
            m5 = cumulative_sum[int(np.round(steps / 2)) - 1] / total

        return np.array([m1, m2, m3, m4, m5])

    def classify_BSC(self, series, suffix, plot=False):
        """
        Classifies a rainfall event into a 4-digit binary shape code based on 
        the Terranova and Iaquinta (2011) method, for events of any length.

        Parameters:
            normalised_cumulative_event (np.array): 
                A NumPy array representing the normalised cumulative rainfall curve (0 to 1).
            plot (bool): If True, generates a plot comparing actual and uniform cumulative curves.

        Returns:
            str: A 4-digit binary shape code (e.g., "0110").
        """
        # Ensure the input is a NumPy array
        
        if suffix not in ['_norm', '_dblnorm']:
            series = np.cumsum(series)
        
        normalised_cumulative_event = np.array(series)

        # Get the number of time steps
        n = len(normalised_cumulative_event)

        # Compute indices which are found at the end of four quarters (roughly)
        q1 = round(n * 0.25)
        q2 = round(n * 0.50)
        q3 = round(n * 0.75)

        # Extract cumulative values at the end of each quarter, handling short events gracefully
        vals_at_end_of_quarters = [
            normalised_cumulative_event[min(q1, n-1)],  # End of Q1
            normalised_cumulative_event[min(q2, n-1)],  # End of Q2
            normalised_cumulative_event[min(q3, n-1)],  # End of Q3
            normalised_cumulative_event[-1],            # End of Q4 (should be ~1)
        ]

        # Expected uniform cumulative values
        expected_vals_at_end_of_quarters = [0.25, 0.50, 0.75, 1.0]

        # Generate the binary shape code
        binary_code = "".join(["1" if actual >= expected else "0" for actual, expected in zip(vals_at_end_of_quarters, expected_vals_at_end_of_quarters)])

        # Optional Plot
        if plot:
            time_steps = np.linspace(0, 1, n)  # Normalized time axis
            uniform_cumulative = np.linspace(0, 1, n)  # Perfectly uniform rainfall line

            plt.figure(figsize=(8, 5))
            plt.plot(time_steps, normalised_cumulative_event, label="Actual Cumulative Rainfall", marker='o')
            plt.plot(time_steps, uniform_cumulative, label="Uniform Rainfall Reference", linestyle="--", color="gray")

            # Mark quarter points
            for i, (q, actual, expected) in enumerate(zip([q1, q2, q3, n-1], vals_at_end_of_quarters, expected_vals_at_end_of_quarters)):
                plt.scatter(time_steps[q], actual, color="red", zorder=3, label=f"Q{i+1}: {binary_code[i]}")

            plt.xlabel("Normalized Time (0 to 1)")
            plt.ylabel("Cumulative Rainfall")
            plt.title(f"Binary Shape Code: {binary_code}")
            plt.legend()
            plt.grid()
            plt.show()
        
        # Convert to index
        binary_values = [int(b) for b in binary_code]            
        # Define weights for front-loading emphasis
        weights = [3, 1, -1, -3]  # Q1 = most front-loaded, Q4 = most back-loaded
        # Compute the front-loading index
        FLI = sum(w * b for w, b in zip(weights, binary_values))
        
        return np.array([binary_code,FLI])


    def calculate_event_loading(self, series, suffix):
        """
        Calculate event loading (EL) as the percent deviation in STH for a hypothetical,
        mirrored storm from the original storm STH.
        """

        def calculate_sth(storm):
            mean_val = np.mean(storm)
            std_val = np.std(storm)

            if mean_val == 0:
                return np.nan  # Return NaN to flag unusable case
            return std_val / mean_val

        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)

        series = np.round(series, 6)

        if np.all(series == 0):
            print("Warning: All-zero series encountered. Returning NaN for event loading.")
            return np.nan

        peak_index = np.argmax(series)
        rising = series[:peak_index + 1]
        mirrored_falling = rising[::-1][1:]
        mirrored_storm = np.concatenate([rising, mirrored_falling])

        sth_original = calculate_sth(series)
        sth_mirrored = calculate_sth(mirrored_storm)

        if np.isnan(sth_original) or sth_original == 0:
            print("Warning: Invalid STH encountered. Returning NaN for event loading.")
            return np.nan

        event_loading = ((sth_mirrored - sth_original) / sth_original) * 100
        return event_loading 

    
    def calculate_event_asymmetry(self, series, suffix):
        """Calculate asymmetry for a single rainfall event"""
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)                  
        n = len(series)
        if n < 3:  # Need at least 3 points for meaningful asymmetry
            return np.nan

        # Calculate empirical CDF (rank transform)
        ranks = stats.rankdata(series)
        U = (ranks - 0.5) / n

        # Use lag-1 (consecutive 5-min intervals)
        diff = U[:-2] - U[2:]

        # Calculate A(k)
        numerator = np.mean(diff**3)
        denominator = np.mean(diff**2)**(3/2)

        if denominator != 0:
            return numerator/denominator
        else:
            return np.nan    

    def compute_time_based_skewness(self, event_list, suffix):
        time_delta = self.ts.data.index[1] - self.ts.data.index[0]
        time_delta_minutes = time_delta.total_seconds() / 60

        skewness_list = []
        for event in event_list:
            values = event.values.flatten()
            if suffix in ['_dblnorm', '_norm']:
                values = np.diff(values, prepend=0)
            
            n = len(values)
            positions = np.arange(1, n + 1) * time_delta_minutes
            total = values.sum()

            # Center of mass
            t_cg = np.sum(positions * values) / total

            # Standard deviation
            sigma_t = np.sqrt(np.sum(((positions - t_cg) ** 2) * values) / total)

            # Skewness
            skew = np.sum(((positions - t_cg) ** 3) * values) / (total * sigma_t**3)
            skewness_list.append(skew)

        return np.array(skewness_list)
    
    def compute_time_based_kurtosis(self, event_list, suffix):
        time_delta = self.ts.data.index[1] - self.ts.data.index[0]
        time_delta_minutes = time_delta.total_seconds() / 60

        kurtosis_list = []
        for event in event_list:
            values = event.values.flatten()
            if suffix in ['_dblnorm', '_norm']:
                values = np.diff(values, prepend=0)            
            n = len(values)
            positions = np.arange(1, n + 1) * time_delta_minutes
            total = values.sum()

            # Center of mass
            t_cg = np.sum(positions * values) / total

            # Standard deviation
            sigma_t = np.sqrt(np.sum(((positions - t_cg) ** 2) * values) / total)

            # Kurtosis
            kurt = np.sum(((positions - t_cg) ** 4) * values) / (total * sigma_t**4)
            kurtosis_list.append(kurt)

        return np.array(kurtosis_list)

    def fourth_with_peak(self, series, suffix):
        if suffix not in ['_norm', '_dblnorm']:
            series = np.cumsum(series)

        # culm value at splits
        interpolated = self.interpolate_rainfall(series,3)

        incremental =  np.diff(interpolated, prepend=0)
        quintile = incremental.argmax()

        return quintile

    def fifth_with_peak(self, series, suffix):
        if suffix not in ['_norm', '_dblnorm']:
            series = np.cumsum(series)

        # culm value at splits
        interpolated = self.interpolate_rainfall(series,4)

        incremental =  np.diff(interpolated, prepend=0)
        quintile = incremental.argmax()

        return quintile

    def third_with_peak(self, series, suffix):
        if suffix not in ['_norm', '_dblnorm']:
            series = np.cumsum(series)

        # culm value at splits
        interpolated = self.interpolate_rainfall(series,2)

        incremental =  np.diff(interpolated, prepend=0)
        quintile = incremental.argmax()

        return quintile


    def calculate_skew_p(self, series, suffix):
        
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)      
        
        # Find the length of rainfall event
        n = len(series)

        # Create time array with 5-minute intervals
        # We want time points at 0, 5, 10, 15, ... minutes
        t = np.arange(n) * 5  # This creates proper 5-minute intervals

        # Find the time to the peak (in minutes)
        peak_idx = np.argmax(series)
        t_peak = t[peak_idx]

        # Calculate normalized time differences
        normalized_diff = (t - t_peak)/t[-1]  # normalize by total duration
        cubed_diff = normalized_diff**3
        skew_p = np.mean(cubed_diff)

        return skew_p  
    
    def calculate_event_dry_ratio (self, series):
        zeroes = np.count_nonzero(series==0)
        event_dry_ratio = zeroes/len(series) * 100
        return event_dry_ratio    
    
    def high_low_zone_indicators(self, series, suffix):
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)    

        mean_intensity = series.mean()
        indices_above = np.where(series > mean_intensity)[0]
        indices_below = np.where(series < mean_intensity)[0]

        # Proportion of time in high/low intensity zones
        frac_time_in_high_intensity_zone = len(indices_above) / len(series) * 100
        frac_time_in_low_intensity_zone = len(indices_below) / len(series) * 100

        # Fraction of rainfall in high intensity zone
        frac_rain_in_high_intensity_zone = series[indices_above].sum() / series.sum() * 100 if series.sum() > 0 else 0

        # Mean intensity in high intensity zone — handle empty case
        if len(indices_above) > 0:
            mean_intensity_high_intensity_zone = series[indices_above].mean()
        else:
            mean_intensity_high_intensity_zone = 0  # or np.nan if you want to signal "undefined"

        return np.array([
            frac_time_in_high_intensity_zone,
            frac_time_in_low_intensity_zone,
            frac_rain_in_high_intensity_zone,
            mean_intensity_high_intensity_zone
        ])

       
    def calculate_nrmse_peak(self, series, suffix):
        
        """
        In this example:
            The NRMSE calculation:
            Takes the value at each time step
            Computes how much each value differs from the peak (100 rainfall value)
            Squares these differences
            Takes the average
            Takes the square root
            Normalizes by total rainfall
            
            This value tells us how concentrated the rainfall is around its peak. A lower value (like this one) indicates 
            the rainfall is relatively concentrated around the peak, while a higher value would indicate the rainfall is more 
            spread out over time.            
        """
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0) 
        series = np.round(series,6)    
        # Array of rainfall values
        pi = np.array(series)
        # Finds the peak
        ppeak = np.max(pi)
        # Finds the total rainfall
        P = np.sum(pi)
        # Finds the length of the rainfall event
        n = len(pi)
        # Root-mean-square error between each ordinate (i.e. timestep) and the peak
        rmse = np.sqrt(np.sum((pi - ppeak)**2) / n)
        # Normalization by total rainfall
        nrmse = rmse / P
        return round(nrmse,2)


    def gini_coef(self, series, suffix):
        """Compute the Gini coefficient in O(n) time without sorting or large memory allocations."""
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)      
        
        n = len(series)
        
        series = series.flatten()
        
        if n == 0 or np.all(series == 0): 
            return 0  # Handle empty or zero arrays safely

        abs_diffs = np.sum(np.abs(series[:, None] - series))  # Efficient pairwise sum
        mean_value = np.mean(series)

        return abs_diffs / (2 * n**2 * mean_value) if mean_value != 0 else 0

    
    def lorentz_asymmetry(self, series, suffix):
        # https://doi.org/10.1016/j.jhydrol.2013.05.002
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)
        series = np.round(series, 6)

        if np.all(series == series[0]):
            return np.nan  # or return 0.0 if you'd prefer a default value

        n = len(series)
        mean = np.mean(series)

        lower = series[series < mean]
        upper = series[series > mean]

        if len(lower) == 0 or len(upper) == 0:
            return np.nan  # asymmetry is undefined in this case

        m = len(lower)
        x_m = lower.max()
        x_m1 = upper.min()

        delta = (mean - x_m) / (x_m1 - x_m)
        lorentz = (m + delta) / n + (lower.mean() + delta * x_m1) / np.sum(series)

        return lorentz
    
    
    def compute_frac_in_quarters(self, series, suffix, interpolate=False, target_length=20):
        """
        Computes the fraction of rainfall in each of four time quarters, with optional interpolation
        to ensure equal time slices. Applies necessary transformations depending on suffix.

        Parameters:
            series (array-like): Rainfall data (intensity or cumulative).
            suffix (str): Used to determine whether to convert from cumulative.
            interpolate (bool): Whether to interpolate to equal time quarters (recommended).
            target_length (int): Number of points to interpolate to (must be divisible by 4).

        Returns:
            np.array: Four values representing % of rainfall in each time quarter.
        """
        series = np.asarray(series)

        # Convert to intensity if needed
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)

        total = series.sum()
        if total == 0:
            return np.array([np.nan, np.nan, np.nan, np.nan])

        if interpolate:
            if target_length % 4 != 0:
                raise ValueError("target_length must be divisible by 4")

            # Interpolate rainfall to fixed time resolution
            x_old = np.linspace(0, 1, len(series))
            x_new = np.linspace(0, 1, target_length)
            interp_series = np.interp(x_new, x_old, series)

            # Slice into 4 equal time quarters
            quarter_len = target_length // 4
            quarters = [interp_series[i * quarter_len : (i + 1) * quarter_len] for i in range(4)]
            frac_in_quarters = [round(q.sum() / interp_series.sum() * 100, 1) for q in quarters]

        else:
            # Fallback to original fixed-index slicing (may be uneven if len(series) not divisible by 4)
            n = len(series)
            q1, q2, q3 = n // 4, n // 2, 3 * n // 4

            quarter_1 = series[:q1]
            quarter_2 = series[q1:q2]
            quarter_3 = series[q2:q3]
            quarter_4 = series[q3:]

            if len(quarter_1) == len(quarter_2) == len(quarter_3) == len(quarter_4):
                frac_q1 = round(quarter_1.sum() / total * 100, 1)
                frac_q2 = round(quarter_2.sum() / total * 100, 1)
                frac_q3 = round(quarter_3.sum() / total * 100, 1)
                frac_q4 = round(quarter_4.sum() / total * 100, 1)
                frac_in_quarters = [frac_q1, frac_q2, frac_q3, frac_q4]
            else:
                frac_in_quarters = [np.nan, np.nan, np.nan, np.nan]

        return np.array(frac_in_quarters)



    def calc_dX_with_interpolation(self, series, percentile, suffix, plot=False):
        """
        Calculates the time (% of event duration) at which a given percentile
        of cumulative rainfall is reached, using linear interpolation.
        """

        if suffix not in ['_norm', '_dblnorm']:
            series = np.cumsum(series)

        n = len(series)
        time_percent = np.linspace(0, 100, n)
        step_size = 100 / (n - 1) if n > 1 else 100  # Handle degenerate case

        percentile *= series[-1] 
        below = np.where(series < percentile)[0]
        above = np.where(series >= percentile)[0]

        if len(below) > 0 and len(above) > 0:
            i_below = below[-1]
            i_above = above[0]

            x1, y1 = time_percent[i_below], series[i_below]
            x2, y2 = time_percent[i_above], series[i_above]

        elif len(below) == 0 and series[0] >= percentile:
            # Edge case: percentile is hit in the first timestep
            x1, y1 = 0, 0
            x2, y2 = step_size, series[0]  # Now x2 is the end of the first step

        else:
            if plot:
                print("Interpolation not possible: percentile not reached.")
                plt.plot(time_percent, series)
                plt.axhline(percentile, color='red', linestyle='--')
                plt.title("Cumulative rainfall with missing percentile")
                plt.xlabel("Time (% of duration)")
                plt.ylabel("Cumulative rainfall")
                plt.grid(True)
                plt.show()
            return None

        slope = (y2 - y1) / (x2 - x1)
        x_at_percentile = x1 + (percentile - y1) / slope

        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(time_percent, series, label='Cumulative rainfall')
            plt.axhline(percentile, color='red', linestyle='--', label=f'{int(percentile*100)}th percentile')
            plt.plot([x1, x2], [y1, y2], 'ko-', label='Interpolation points')
            plt.axvline(x_at_percentile, color='green', linestyle='--', label='Interpolated time')
            plt.scatter([x_at_percentile], [percentile], color='green', zorder=5)
            plt.title(f'Percentile Time Calculation ({int(x_at_percentile)})')
            plt.xlabel('Time (% of duration)')
            plt.ylabel('Cumulative Rainfall')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        return x_at_percentile

    def calc_ARR_thirds(self, series, suffix):
        """
        Classifies a normalised rainfall event into one of three ARR thirds 
        based on when 50% of rainfall occurs.

        Returns:
            1 if < 40% of time,
            2 if between 40% and 60%,
            3 if > 60%
        """
        percentile = 0.5  # 50% of total rainfall
        time_for_percentile = self.calc_dX_with_interpolation(series, percentile, suffix)

        if time_for_percentile is None:
            return None  # Optional: handle edge case if interpolation fails

        if time_for_percentile < 40:
            return 1
        elif 40 <= time_for_percentile <= 60:
            return 2
        elif time_for_percentile > 60:
            return 3

    def compute_thirds_rcg_interpolated(self, event_series, suffix):
        # Compute the center of gravity (CoM) for the event series
        rcg = self.compute_rcg_interpolated(event_series, suffix)

        # Classify the CoM fractional value into thirds
        def classify_rcg_fraction(rcg_value):
            if rcg_value < 1/3:
                return 1
            elif rcg_value < 2/3:
                return 2
            else:
                return 3

        # Return the classification for the event
        return classify_rcg_fraction(rcg)
    
    
    def find_heaviest_run_half(self, series, suffix, threshold=0.8):
        """
        Determine whether the first or second half of a rainfall event contains the heaviest run.

        Parameters:
        - series: 1D array-like of precipitation values (e.g. mm per 5 min).
        - threshold: Rainfall threshold to define a "run".

        Returns:
        - 'first_half', 'second_half', or 'both_halves' depending on where the heaviest run is.
        - start and end index of the heaviest run.
        """
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)
        
        series = np.asarray(series)
        above = series > threshold

        # Identify run boundaries
        run_ids = np.zeros_like(series, dtype=int)
        run_ids[1:] = (above[1:] != above[:-1]).cumsum()

        # Get runs where above threshold
        valid_runs = []
        for run_id in np.unique(run_ids[above]):
            idx = np.where(run_ids == run_id)[0]
            run_total = series[idx].sum()
            valid_runs.append((idx[0], idx[-1], run_total))

        if not valid_runs:
            return None, None, None

        # Find the heaviest run
        start_idx, end_idx, _ = max(valid_runs, key=lambda x: x[2])

        # Determine midpoint index
        mid = len(series) // 2
        if start_idx <= mid <= end_idx:
            half = 'both_halves'
        elif end_idx < mid:
            half = 'first_half'
        else:
            half = 'second_half'

        return half, start_idx, end_idx
    
    def calculate_tci(self, series, suffix):
        if suffix in ['_dblnorm', '_norm']:
            series = np.diff(series, prepend=0)

        rainfall = np.array(series)
        T = len(series)  # Total duration (number of time steps)
        P = np.sum(series)  # Total rainfall

        if P == 0:
            return 0, None, [0] * T  # If no rainfall, TCI is zero

        tci_values = []  # To store TCI for each hypothetical temporal center

        # Step 5: Iterate over each possible temporal center
        for center in range(T):
            # Step 1: Mark the chosen time point as the hypothetical temporal center
            sorted_indices = [center]

            # Step 2: Rank the remaining time points based on proximity & rainfall
            remaining_indices = list(range(T))
            remaining_indices.remove(center)

            # Sort remaining indices by distance from the center, prioritizing higher rainfall
            remaining_indices.sort(key=lambda i: (abs(i - center), -rainfall[i]))
            sorted_indices.extend(remaining_indices)

            # Step 3: Compute cumulative rainfall following this order
            cumulative_rainfall = np.cumsum(rainfall[sorted_indices])
            cumulative_time = np.arange(1, T + 1)  # Time from 1 to T

            # Step 4: Compute TCI
            actual_curve = np.trapz(cumulative_rainfall, cumulative_time)  # Area under actual curve
            reference_line = 0.5 * T * P  # Area under the reference straight line
            dA = actual_curve - reference_line  # Difference in area
            A = 0.5 * T * P  # Area of the reference triangle

            tci = dA / A  # TCI for this center
            tci_values.append(tci)

        # Step 5: Identify the max TCI and corresponding temporal center
        max_tci = max(tci_values)
        best_center = np.argmax(tci_values)

        return max_tci# , best_center, tci_values

    
    def get_metrics(self):
        # These only apply to raw events
        self.metrics['total_precip'] = np.array([e.sum().values[0] for e in self.ts.raw_events])
        self.metrics["duration"] = np.array([len(e)*temp_res for e in self.ts.raw_events])     
        self.metrics["I30"] = np.array([e.rolling(window="30min").sum().max().values[0] for e in self.ts.raw_events])/30   
        self.metrics["time_to_peak"] = np.array([((e.idxmax().values[0]- e.index[0]).total_seconds()/60) for e in self.ts.raw_events])
        self.metrics["peak_position_ratio"] = self.metrics["time_to_peak"]/self.metrics["duration"]
        
        event_sets = {
            "": self.ts.raw_events,
            "_DMC_10": self.ts.DMCs,
#             "_DMC_100": self.ts.DMCs_100,
            "_dblnorm": self.ts.double_normalised_events,
#             "_norm": self.ts.normalised_events
        }

        for suffix, events in event_sets.items():
            print(suffix)
            if suffix in ['_norm', '_dblnorm']:
                self.metrics[f"max_intensity{suffix}"] = np.array([np.diff(e.iloc[:, 0].to_numpy(), prepend=0).max() for e in events])
                self.metrics[f"mean_intensity{suffix}"] = np.array([np.diff(e.iloc[:, 0].to_numpy(), prepend=0).mean() for e in events])
                self.metrics[f"min_intensity{suffix}"] = np.array([np.diff(e.iloc[:, 0].to_numpy(), prepend=0).min() for e in events])
                self.metrics[f"std{suffix}"] = np.array([np.diff(e.iloc[:, 0].to_numpy(), prepend=0).std() for e in events])
                self.metrics[f"skewness{suffix}"] = np.array([skew(np.diff(e.iloc[:, 0].to_numpy(), prepend=0), bias=False) for e in events])
                self.metrics[f"kurtosis{suffix}"] = np.array([skew(np.diff(e.iloc[:, 0].to_numpy(), prepend=0), bias=False) for e in events])
            else:
                self.metrics[f"std{suffix}"] = np.array([e.std().values[0] for e in events])
                self.metrics[f"max_intensity{suffix}"] = np.array([e.max().values[0] for e in events])
                self.metrics[f"mean_intensity{suffix}"] = np.array([e.mean().values[0] for e in events]) 
                self.metrics[f"min_intensity{suffix}"] = np.array([e.min().values[0] for e in events])
                self.metrics[f"cv{suffix}"] = self.metrics[f"std{suffix}"] / self.metrics[f"mean_intensity{suffix}"]
                self.metrics[f"skewness{suffix}"] = np.array([e.skew().values[0] for e in events])
                self.metrics[f"kurtosis{suffix}"] = np.array([e.kurtosis().values[0] for e in events])       
            
            self.metrics[f"cv{suffix}"] = self.metrics[f"std{suffix}"] / self.metrics[f"mean_intensity{suffix}"]                
                
            self.metrics[f"relative_amp{suffix}"] = (self.metrics[f"max_intensity{suffix}"] - self.metrics[f"min_intensity{suffix}"])/self.metrics[f"mean_intensity{suffix}"]
#             self.metrics[f"relative_amp_scaled{suffix}"] = [(np.max(vals := e.iloc[:, 0].to_numpy()[e.iloc[:, 0].to_numpy() > 0]) - np.min(vals)) / np.mean(vals) if np.any(e.iloc[:, 0].to_numpy() > 0) else np.nan for e in events]
            self.metrics[f"peak_mean_ratio{suffix}"] = self.metrics[f"max_intensity{suffix}"]/self.metrics[f"mean_intensity{suffix}"]
            self.metrics[f"peak_mean_ratio_scaled{suffix}"] = [np.max(e.iloc[:, 0].to_numpy() / np.mean(e.iloc[:, 0].to_numpy()))
                    for e in events]
            
            self.metrics[f"PCI{suffix}"] = np.array([self.calculate_pci(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"TCI{suffix}"] = np.array([self.calculate_tci(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"asymm_d{suffix}"] = np.array([self.calculate_event_asymmetry(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"Event Loading{suffix}"] = np.array([self.calculate_event_loading(e.values, suffix) for e in events]) 
            self.metrics[f"NRMSE_P{suffix}"] = np.array([self.calculate_nrmse_peak(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"skewp{suffix}"] = np.array([self.calculate_skew_p(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            
            self.metrics[f"gini{suffix}"] = np.array([self.gini_coef(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"lorentz_asymetry{suffix}"] = np.array([self.lorentz_asymmetry(e.values, suffix) for e in events])  
            
            temp = np.array([self.find_heaviest_run_half(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"heaviest_half{suffix}"] = temp[:, 0]
                       
            self.metrics[f"intermittency{suffix}"] = np.array([self.compute_intermittency(e.values) for e in events])
            self.metrics[f"event_dry_ratio{suffix}"] = np.array([self.calculate_event_dry_ratio(e.values) for e in events])
            
            self.metrics[f'ni{suffix}']= self.metrics[f"max_intensity{suffix}"]/self.metrics[f"mean_intensity{suffix}"]
            self.metrics[f"time_skewness{suffix}"] = self.compute_time_based_skewness(events, suffix)
            self.metrics[f"time_kurtosis{suffix}"] = self.compute_time_based_kurtosis(events, suffix)
            
            self.metrics[f"centre_gravity{suffix}"] = np.array([self.compute_rcg(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"centre_gravity_interpolated{suffix}"] = np.array([self.compute_rcg_interpolated(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            
            # Mass distribution indicators m1–m5
            temp = np.array([self.compute_mass_dist_indicators(e.iloc[:, 0].to_numpy(), suffix, False) for e in events])
            self.metrics[f"m1{suffix}"] = temp[:, 0]
            self.metrics[f"m2{suffix}"] = temp[:, 1]
            self.metrics[f"m3{suffix}"] = temp[:, 2]
            self.metrics[f"m4{suffix}"] = temp[:, 3]
            self.metrics[f"m5{suffix}"] = temp[:, 4]
            
            temp = np.array([self.compute_mass_dist_indicators(e.iloc[:, 0].to_numpy(), suffix, True) for e in events])
            self.metrics[f"m1_wi{suffix}"] = temp[:, 0]
            self.metrics[f"m2_wi{suffix}"] = temp[:, 1]
            self.metrics[f"m3_wi{suffix}"] = temp[:, 2]
            self.metrics[f"m4_wi{suffix}"] = temp[:, 3]
            self.metrics[f"m5_wi{suffix}"] = temp[:, 4]            

#             temp = np.array([self.compute_frac_in_quarters(e.iloc[:, 0].to_numpy(), suffix) for e in events])
#             self.metrics[f"frac_q1{suffix}"] = temp[:,0]
#             self.metrics[f"frac_q2{suffix}"] = temp[:,1]
#             self.metrics[f"frac_q3{suffix}"] = temp[:,2]
#             self.metrics[f"frac_q4{suffix}"] = temp[:,3]
            
            temp = np.array([self.compute_frac_in_quarters(e.iloc[:, 0].to_numpy(), suffix, True) for e in events])
            self.metrics[f"frac_q1_wi_{suffix}"] = temp[:,0]
            self.metrics[f"frac_q2_wi_{suffix}"] = temp[:,1]
            self.metrics[f"frac_q3_wi_{suffix}"] = temp[:,2]
            self.metrics[f"frac_q4_wi_{suffix}"] = temp[:,3]              

            # Indicators based on dividing event into low and high parts
            temp = np.array([self.high_low_zone_indicators(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"% time HIZ{suffix}"] = temp[:,0]
            self.metrics[f"% time LIZ{suffix}"] = temp[:,1]
            self.metrics[f"% rain HIZ{suffix}"] = temp[:,2]
            self.metrics[f"Mean Intensity HIZ{suffix}"] = temp[:,3]

            # Huff quantiles
            self.metrics[f"3rd_w_peak{suffix}"] = np.array([self.third_with_peak(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"4th_w_peak{suffix}"] = np.array([self.fourth_with_peak(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"5th_w_peak{suffix}"] = np.array([self.fifth_with_peak(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"third_ppr{suffix}"] = np.select([self.metrics["peak_position_ratio"] < 0.4, (self.metrics["peak_position_ratio"] >= 0.4) & (self.metrics["peak_position_ratio"] <= 0.6), self.metrics["peak_position_ratio"] > 0.6],[0, 1, 2])

            # third with highest percent rain
            self.metrics[f"3rd_ARR{suffix}"] = np.array([self.calc_ARR_thirds(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            self.metrics[f"3rd_rcg{suffix}"] = np.array([self.compute_thirds_rcg_interpolated(e.iloc[:, 0].to_numpy(), suffix) for e in events])
            
            # D50 etc
            self.metrics[f"T25{suffix}"] = np.array([self.calc_dX_with_interpolation(e.iloc[:, 0].to_numpy(), 0.25, suffix) for e in events])
            self.metrics[f"T50{suffix}"] = np.array([self.calc_dX_with_interpolation(e.iloc[:, 0].to_numpy(), 0.50, suffix) for e in events])
            self.metrics[f"T75{suffix}"] = np.array([self.calc_dX_with_interpolation(e.iloc[:, 0].to_numpy(), 0.75, suffix) for e in events])
            self.metrics[f"D50{suffix}"] = np.array([self.calc_dX_with_interpolation(e.iloc[:, 0].to_numpy(), 0.5, suffix) for e in events])
        
            # calculate BSC
            temp = np.array([self.classify_BSC(e.iloc[:, 0].to_numpy(), suffix, False) for e in events])         
            self.metrics[f"BSC{suffix}"] = temp[:,0]
            self.metrics[f"BSC_Index{suffix}"] = temp[:, 1].astype(int)
        
    def plot_boxplots(self, metrics):
        n_plots = len(metrics)
        cols = min(n_plots, 4)  # Max 4 columns
        rows = np.int16(np.ceil(n_plots / 4))  # Determine number of rows

        fig, axes = plt.subplots(rows, cols, figsize=(10, 8))  # Create subplots grid
        axes = np.ravel(axes)  # Flatten to 1D for easy iteration

        for i, metric in enumerate(metrics):
            axes[i].boxplot(self.metrics[metric])  # Use correct subplot
            axes[i].set_title(metric)

        plt.tight_layout()  # Adjust layout for clarity
    
    def plot_histograms(self,metrics):
        
        n_plots = len(metrics)
        cols = min(n_plots, 4)  # Max 4 columns
        rows = np.int16(np.ceil(n_plots / 4))  # Determine number of rows

        fig, axes = plt.subplots(rows, cols, figsize=(10, 8))  # Create subplots grid
        axes = np.ravel(axes)  # Flatten to 1D for easy iteration

        for i, metric in enumerate(metrics):
            axes[i].hist(self.metrics[metric])  # Use correct subplot
            axes[i].set_title(metric)

        plt.tight_layout()  # Adjust layout for clarity


