import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.interpolate import interp1d

class precip_time_series:
    def __init__(self, data_path):

        self.data,self.statid = self.read_raw_data_as_pandas_df(data_path)
        
        self.padded = False

        self.events = None
        
        self.dimensionless_events = None
        
        self.interpolated_events = None
    
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
        precip = pd.read_csv(raw_data_file_path, encoding="ISO-8859-1",index_col=1)

        # Timestamps str -> datetime
        precip.index = pd.to_datetime(precip.index)

        # Save ID of station
        station_id = str(precip.station.iloc[0])

        # Remove column with station ID
        precip = precip.drop("station",axis=1) 

        return precip,station_id    

    def pad_and_resample(self,freq = '5min',pad_value = 0):
        # Resample the data to the specified frequency and pad missing values with pad_value
        self.data = self.data.resample(freq).sum().fillna(pad_value)
        self.padded = True

    def return_specific_event(self,event_idx):

        # Size of timesteps
        time_delta = self.data.index[1] - self.data.index[0]
        time_delta_minutes = time_delta.seconds / 60

        # Extract event data
        event = self.data.loc[self.events[event_idx][0]:self.events[event_idx][1]]

        return event  
    
    def return_specific_dimensionless_event(self,event_idx):

        # Size of timesteps
        time_delta = self.data.index[1] - self.data.index[0]
        time_delta_minutes = time_delta.seconds / 60
        
        # Extract event data
        event = self.dimensionless_events[event_idx]

        return event       
    
    def return_specific_interpolated_event(self,event_idx):
        
        # Extract event data
        event = self.interpolated_events[event_idx]

        return event     
        
    def get_events(self,threshold='11h',min_duration = 30, min_precip = 1):
        
        if not self.padded:
            self.pad_and_resample()

        self.init_events(threshold)
        self.filter_events_by_length(min_duration)
        self.filter_events_by_amount(min_precip)

    def init_events(self,threshold):
        
        precip = self.data
        
        # Size of timesteps
        time_delta = precip.index[1]-precip.index[0]

        # Rolling 11 hour sum
        precip_sum = precip.rolling(threshold).sum()

        # dates with no precip last 11 hours
        dates_w_zero_sum = precip_sum.index[(precip_sum.mask(precip_sum!=0)==precip_sum).values[:,0]]

        # Add first date with rain
        for date in precip.index:
            if precip.loc[date].values[0] != 0:
                start_dates = [date]
                break

        # Save start and end dates
        end_dates   = []
        for date in tqdm(dates_w_zero_sum):
            if precip_sum.loc[date- time_delta].values[0]!=0:
                end_dates += [date- pd.to_timedelta(threshold)]
            if precip_sum.loc[date+ time_delta].values[0]!=0:
                start_dates += [date+ time_delta]
        
        # Add end to last event
        for date in reversed(precip.index):  # Iterate from last to first
            if precip.loc[date].values[0] != 0:  # Check if value is not zero
                end_dates += [date]
                break  # Stop at the first nonzero value
        
        # Save events as list of tuples
        events = []
        for i in range(len(end_dates)):
            events+=[(start_dates[i],end_dates[i])]

        # update events
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

    def create_interpolated_events(self):
        n=10
        # Make sure events have been computed
        if self.dimensionless_events == None:
            self.create_dimensionless_events()
        
        # Make list of nparrays containing the values of the dimensionless curve
        interpolated_events = [self.get_interpolated_event(event, n) for event in self.dimensionless_events]

        # Assign to global value
        self.interpolated_events = interpolated_events        

        
    def create_dimensionless_events(self):

        # Make sure events have been computed
        if self.events == None:
            self.get_events()
        
        # Make list of nparrays containing the values of the dimensionless curve
        dimensionless_events = [self.get_dimensionless_event(self.data.loc[event[0]:event[1]].values) for event in self.events]

        # Assign to global value
        self.dimensionless_events = dimensionless_events

    def get_dimensionless_event(self,series):
    
        # Calculate cumulative rainfall
        cumulative_rainfall = np.cumsum(series)
        cumulative_rainfall = np.append([0],cumulative_rainfall)

        # normalize
        normalized_cumulative_rainfall = cumulative_rainfall/cumulative_rainfall[-1]

        return normalized_cumulative_rainfall
    
    def get_interpolated_event(self, series, n):
        # Calculate cumulative rainfall
        normalized_cumulative_rainfall = series

        # Define target points for bin_number bins
        target_points = np.linspace(0, 1, n+1)

        # Create interpolation function based on existing data points
        rainfall_times = np.array(range(0, len(normalized_cumulative_rainfall)))

        # Normalize time from 0 to 1
        normalized_time = (rainfall_times - rainfall_times[0]) / (rainfall_times[-1] - rainfall_times[0])
        interpolation_func = interp1d(normalized_time, normalized_cumulative_rainfall, kind='linear', fill_value="extrapolate")

        # Interpolate values at target points
        interpolated_values = interpolation_func(target_points)
    
        return interpolated_values        

    def plot_all_events(self):
        plt.figure()
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

    def plot_specific_dimensionless_events(self,event_idxs):
        plt.figure()
        for idx in event_idxs:
            x_values = np.linspace(0,1,len(self.dimensionless_events[idx]))
            plt.plot(x_values,self.dimensionless_events[idx],label = f"Event: {idx+1}")
        plt.legend()
        plt.title("dimensionless events")
        
    def plot_specific_dimensionless_event(self,event_idx):
        plt.figure()
        x_values = np.linspace(0,1,len(self.dimensionless_events[event_idx]))
        plt.plot(x_values,self.dimensionless_events[event_idx],label = f"Event: {event_idx+1}")
        plt.legend()
        plt.title("Dimensionless events")        

class rainfall_analysis:
    def __init__(self,ts: precip_time_series):
        self.ts = ts
        self.metrics = {} 
        
        # Prepere ts for analysis
        if not self.ts.padded:
            ts.pad_and_resample()

        if ts.events == None:
            ts.get_events()
        
        if ts.dimensionless_events == None:
            ts.create_dimensionless_events()

    def compute_intermittency(self,series):
            
            total_timesteps = len(series)
            wet = series>0
            transistions = (wet[:-1] != wet[1:]).sum()
            
            intermittency = transistions/total_timesteps
    
            return intermittency 

    def compute_rcg_idx(self,series):
            
            # Culmitative sum
            culmitative_sum = np.cumsum(series)
            
            # Normalize
            culmitative_sum /= culmitative_sum[-1]

            #first idx over 0.5
            idx = np.argmax(culmitative_sum>0.5)

            return idx

    def compute_rcg(self):

        time_delta = self.ts.data.index[1]-self.ts.data.index[0]
        time_delta_minuts = time_delta.seconds/60

        # first index over center of mass
        rcg_indeces = np.array([self.compute_rcg_idx(self.ts.data.loc[event[0]:event[1]].values) for event in self.ts.events])

        # time of center
        toc = np.array([self.ts.data.loc[event[0]:event[1]].index[rcg_indeces[i]] for i,event in enumerate(self.ts.events)])

        # duration until center
        tcg = np.array([time_delta_minuts + (toc[i] - event[0]).total_seconds()/60 for i,event in enumerate(self.ts.events)]).reshape(self.metrics["duration"].shape)

        # rcg
        rcg = tcg/self.metrics["duration"]

        return rcg

    def compute_mass_dist_indicators(self,series):

        """
        Thoughts:
        Current implementation treats time as discrete, alternatively one could use averages if 
        0.33*T is not a multiple of stepsize.
        """
        # length of series
        steps = len(series)

        # peak_idx
        peak_idx = np.argmax(series)

        # Culmitative sum
        culmitative_sum = np.cumsum(series)


        # Mass distribution indicators
        if peak_idx == 0:
            m1   = culmitative_sum[peak_idx]/culmitative_sum[-1]
        else:
            m1   = culmitative_sum[peak_idx]/(culmitative_sum[-1]-culmitative_sum[peak_idx-1])
        
        m2   = (series[peak_idx]/culmitative_sum[-1])[0]
        m3   = culmitative_sum[np.int32(np.round(steps/3))-1]/culmitative_sum[-1]
        m4   = culmitative_sum[np.int32(np.round(steps*0.3))-1]/culmitative_sum[-1]
        m5   = culmitative_sum[np.int32(np.round(steps/2))-1]/culmitative_sum[-1]    

        return np.array([m1,m2,m3,m4,m5])

    def classify_BSC(self, series, plot=False):
        """
        Classifies a rainfall event into a 4-digit binary shape code based on 
        the Terranova and Iaquinta (2011) method, for events of any length.

        Parameters:
            dimensionless_cumulative_event (np.array): 
                A NumPy array representing the dimensionless cumulative rainfall curve (0 to 1).
            plot (bool): If True, generates a plot comparing actual and uniform cumulative curves.

        Returns:
            str: A 4-digit binary shape code (e.g., "0110").
        """
        # Ensure the input is a NumPy array
        dimensionless_cumulative_event = np.array(series)

        # Get the number of time steps
        n = len(dimensionless_cumulative_event)

        # Compute indices which are found at the end of four quarters (roughly)
        q1 = round(n * 0.25)
        q2 = round(n * 0.50)
        q3 = round(n * 0.75)

        # Extract cumulative values at the end of each quarter, handling short events gracefully
        vals_at_end_of_quarters = [
            dimensionless_cumulative_event[min(q1, n-1)],  # End of Q1
            dimensionless_cumulative_event[min(q2, n-1)],  # End of Q2
            dimensionless_cumulative_event[min(q3, n-1)],  # End of Q3
            dimensionless_cumulative_event[-1],            # End of Q4 (should be ~1)
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
            plt.plot(time_steps, dimensionless_cumulative_event, label="Actual Cumulative Rainfall", marker='o')
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


    def calculate_event_loading(self, series):  
        """  
        Calculate event loading (EL) as the percent deviation in STH for a hypothetical,   
        mirrored storm from the original storm STH.  

        Parameters:  
        rainfall: array-like precipitation amounts over time for the storm event.  

        Returns:  
        sth_original: STH of the original storm (CV)  
        sth_mirrored: STH of the hypothetical mirrored storm  
        event_loading (EL): percentage deviation of the mirrored STH from the original  
        """  

        def calculate_sth(storm):  
            """  
            Calculate a proxy for spatio-temporal homogeneity (STH) using the coefficient of variation.  
            A lower CV indicates more uniform (homogeneous) distribution over time.  
            """  
            mean_val = np.mean(storm)  
            # Avoid division by zero  
            if mean_val == 0:  
                return np.inf  
            return np.std(storm) / mean_val  

        # Identify the index of the peak (maximum rainfall) as the point separating rising from falling limbs  
        peak_index = np.argmax(series)  

        time_steps = np.arange(1, len(series))  

        # Extract the rising limb (inclusive of the peak)  
        rising = series[:peak_index+1]  

        # We can construct a hypothetical mirrored storm by reflecting the rising limb.  
        # Remove the first entry from the mirrored falling limb to avoid duplicating the peak.  
        mirrored_falling = rising[::-1][1:]  

        # Create the hypothetical mirrored storm  
        mirrored_storm = np.concatenate([rising, mirrored_falling])  

        # Calculate STH for both storms using the coefficient of variation  
        sth_original = calculate_sth(series)  
        sth_mirrored = calculate_sth(mirrored_storm)  

        # Calculate event loading (EL) as percentage deviation  
        # A negative EL means a front-loaded event (mirrored storm is less homogeneous)  
        # A positive EL means a rear-loaded event (mirrored storm is more homogeneous)  
        event_loading = ((sth_mirrored - sth_original) / sth_original) * 100  

        return event_loading # sth_original, sth_mirrored,   

    
    def calculate_event_asymmetry(self, series):
        """Calculate asymmetry for a single rainfall event"""
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

    
    def compute_time_based_skewness(self):

        time_delta = self.ts.data.index[1]-self.ts.data.index[0]
        time_delta_minuts = time_delta.seconds/60

        # first index over center of mass
        CoM_idx = np.array([self.compute_rcg_idx(self.ts.data.loc[event[0]:event[1]].values) for event in self.ts.events])

        # time of center
        toc = np.array([self.ts.data.loc[event[0]:event[1]].index[CoM_idx[i]] for i,event in enumerate(self.ts.events)])

        # duration until center
        tcg = np.array([time_delta_minuts + (toc[i] - event[0]).total_seconds()/60 for i,event in enumerate(self.ts.events)]).reshape(self.metrics["duration"].shape)
        
        time_based_skewness_list = np.zeros(tcg.shape)

        for i, event in enumerate(self.ts.events):
            series = self.ts.data.loc[event[0]:event[1]].values

            numerator = np.array([((j+1)*time_delta_minuts-tcg[i])**3 * precip for j,precip in enumerate(series)]).sum()

            total_precip = series.sum()

            sigma_t = np.sqrt(np.array([((j+1)*time_delta_minuts-tcg[i])**2 * precip for j,precip in enumerate(series)]).sum()/total_precip)

            time_based_skewness_list[i] = numerator/(total_precip*sigma_t**3)

        
        return time_based_skewness_list

    def compute_time_based_kurtosis(self):

        time_delta = self.ts.data.index[1]-self.ts.data.index[0]
        time_delta_minuts = time_delta.seconds/60

        # first index over center of mass
        CoM_idx = np.array([self.compute_rcg_idx(self.ts.data.loc[event[0]:event[1]].values) for event in self.ts.events])

        # time of center
        toc = np.array([self.ts.data.loc[event[0]:event[1]].index[CoM_idx[i]] for i,event in enumerate(self.ts.events)])

        # duration until center
        tcg = np.array([time_delta_minuts + (toc[i] - event[0]).total_seconds()/60 for i,event in enumerate(self.ts.events)]).reshape(self.metrics["duration"].shape)
        
        time_based_kurtosis_list = np.zeros(tcg.shape)

        for i, event in enumerate(self.ts.events):
            series = self.ts.data.loc[event[0]:event[1]].values

            numerator = np.array([((j+1)*time_delta_minuts-tcg[i])**4 * precip for j,precip in enumerate(series)]).sum()

            total_precip = series.sum()

            sigma_t = np.sqrt(np.array([((j+1)*time_delta_minuts-tcg[i])**2 * precip for j,precip in enumerate(series)]).sum()/total_precip)

            time_based_kurtosis_list[i] = numerator/(total_precip*sigma_t**4)

        
        return time_based_kurtosis_list

    def fourth_with_peak(self,dimensionless_event):

        # culm value at splits
        interpolated = self.interpolate_rainfall(dimensionless_event,4)

        # precip in each split
        split_sum = interpolated[1:]-interpolated[:-1]

        # Quantile with most precip
        quantile = np.argmax(split_sum)+1

        return quantile

    def fifth_with_peak(self,dimensionless_event):

        # culm value at splits
        interpolated = self.interpolate_rainfall(dimensionless_event,5)

        # precip in each split
        split_sum = interpolated[1:]-interpolated[:-1]

        # Quintile with most precip
        quintile = np.argmax(split_sum)+1

        return quintile
    
    
    def third_with_peak(self,dimensionless_event):

        # culm value at splits
        interpolated = self.interpolate_rainfall(dimensionless_event,3)

        # precip in each split
        split_sum = interpolated[1:]-interpolated[:-1]

        # Quintile with most precip
        quintile = np.argmax(split_sum)+1

        return quintile
 
    def third_classification_max_percent(self,dimensionless_event):

        # culm value at splits
        interpolated = self.interpolate_rainfall(dimensionless_event,3)

        # precip in each split
        split_sum = interpolated[1:]-interpolated[:-1]

        # third with most precip
        third = np.argmax(split_sum)+1

        return third

    def third_classification_CoM(self,dimensionless_event):

        # culm value at splits
        interpolated = self.interpolate_rainfall(dimensionless_event,3)

        # third with most precip
        third = np.argmax(interpolated>=0.5)

        return third    

    def interpolate_rainfall(self,dim_less_curve,bin):
        if dim_less_curve is None or len(dim_less_curve) == 2:
            return None
        
        # Define target points for splits
        target_points = np.linspace(0, 1, bin+1)
        
        # normalized time
        normalized_time = np.linspace(0, 1,len(dim_less_curve))

        # Create interpolation function based on existing data points
        interpolation_func = interp1d(normalized_time, dim_less_curve, kind='linear', fill_value="extrapolate")
        
        # Interpolate values at target points
        interpolated_values = interpolation_func(target_points)
        
        return interpolated_values


    def calculate_skew_p(self, series):
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
    
    def high_low_zone_indicators (self, series):
        # Find the mean intensity
        mean_intensity= series.mean()
        # Find the indices where values are above/below the mean
        indices_above = np.where(series>mean_intensity)[0]
        indices_below = np.where(series<mean_intensity)[0]
        # Find the proportion of the event's time spent in the high/low intensity zones
        frac_time_in_high_intensity_zone = len(indices_above)/len(series)*100
        frac_time_in_low_intensity_zone = len(indices_below)/len(series)*100
        # Find the fraction of rainfall in the high intensity zone
        frac_rain_in_high_intensity_zone = series[indices_above].sum()/series.sum() * 100
        # Find the mean intensity in the high_intensity zone
        mean_intensity_high_intensity_zone = series[indices_above].mean()    
        return np.array([frac_time_in_high_intensity_zone,frac_time_in_low_intensity_zone, frac_rain_in_high_intensity_zone, mean_intensity_high_intensity_zone])
    
#     def calculate_skewp_normalized(self, series):
#         """
#         Compute the SkewP parameter as described in the paper.

#         Parameters:
#             rainfall_values (array-like): Rainfall intensities at each time step.
#             time_intervals (array-like): Corresponding time steps.

#         Returns:
#             float: SkewP value (expected to be between -0.25 and 0.25).
#         """
#         rainfall_values = np.array(series)
#         num_steps = len(series)

#         # Generate time steps assuming uniform intervals
#         time_intervals = np.arange(num_steps)

#         # Compute the centroid time of rainfall mass curve
#         total_rainfall = np.sum(series)
#         centroid_time = np.sum(time_intervals * rainfall_values) / total_rainfall

#         # Find time of peak intensity
#         peak_time = time_intervals[np.argmax(series)]

#         # Total duration
#         event_duration = time_intervals[-1] - time_intervals[0]

#         # Compute SkewP (normalized)
#         skewp = (peak_time - centroid_time) / event_duration
#         return skewp
        
    def calculate_nrmse_peak(self, series):
        
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

    def gini_coef(self, series):
        n = len(series)
        mean = np.mean(series)

        # vectorized sum for numerator
        abs_diff_matrix = np.abs(series[:, None] - series[None, :])
        sum = np.sum(abs_diff_matrix)

        # Compute gini
        gini = sum/(2*(n**2)*mean)

        return gini
    
    def lorentz_asymmetry(self,series):
        # https://doi.org/10.1016/j.jhydrol.2013.05.002
        n = len(series)
        mean = np.mean(series)
        m = len(series[series<mean])
        x_m = (series[series<mean]).max()
        x_m1 = (series[series>mean]).min()
        
        delta = (mean - x_m)/(x_m1-x_m)

        lorentz = (m+delta)/n + ((series[series<mean]).mean() + delta*x_m1)/np.sum(series)

        return lorentz

    def compute_frac_in_quarters(self, series):

        # Compute the split indices
        n = len(series)
        q1, q2, q3 = n // 4, n // 2, 3 * n // 4

        # Split the array into 4 quarters
        quarter_1 = series[:q1]  # First quarter
        quarter_2 = series[q1:q2]  # Second quarter
        quarter_3 = series[q2:q3]  # Third quarter
        quarter_4 = series[q3:]  # Fourth quarter

        frac_q1 = round(quarter_1.sum()/series.sum() *100,1)
        frac_q2 = round(quarter_2.sum()/series.sum() *100,1)
        frac_q3 = round(quarter_3.sum()/series.sum() *100,1)
        frac_q4 = round(quarter_4.sum()/series.sum() *100,1)

        return np.array([frac_q1, frac_q2, frac_q3, frac_q4])
    
    def calc_dX_with_interpolation(self, dimensionless_event, percentile):

        time_percentage = (np.arange(0, len(dimensionless_event) + 1) / len(dimensionless_event)) * 100

        # Find the indices where the cumulative rainfall crosses the percentile_value
        indices_below = np.where(dimensionless_event < percentile)[0]
        indices_above = np.where(dimensionless_event >= percentile)[0]

        # Ensure there are indices both below and above the percentile value
        if len(indices_below) > 0 and len(indices_above) > 0:
            index_below = indices_below[-1]  # Last index below the percentile value
            index_above = indices_above[0]    # First index above the percentile value

            # Perform linear interpolation to find the exact intersection point
            x_below = time_percentage[index_below]
            y_below = dimensionless_event[index_below]

            x_above = time_percentage[index_above]
            y_above = dimensionless_event[index_above]

            # Calculate the slope
            slope = (y_above - y_below) / (x_above - x_below)
            # Use the formula to find the exact x value where the y value equals percentile_value
            time_for_percentile = x_below + (percentile - y_below) / slope

            return time_for_percentile  
 
    def calc_ARR_thirds (self, dimensionless_event):
        percentile = 0.5
        time_percentage = (np.arange(0, len(dimensionless_event) + 1) / len(dimensionless_event)) * 100

        # Find the indices where the cumulative rainfall crosses the percentile_value
        indices_below = np.where(dimensionless_event < percentile)[0]
        indices_above = np.where(dimensionless_event >= percentile)[0]

        # Ensure there are indices both below and above the percentile value
        if len(indices_below) > 0 and len(indices_above) > 0:
            index_below = indices_below[-1]  # Last index below the percentile value
            index_above = indices_above[0]    # First index above the percentile value

            # Perform linear interpolation to find the exact intersection point
            x_below = time_percentage[index_below]
            y_below = dimensionless_event[index_below]

            x_above = time_percentage[index_above]
            y_above = dimensionless_event[index_above]

            # Calculate the slope
            slope = (y_above - y_below) / (x_above - x_below)
            # Use the formula to find the exact x value where the y value equals percentile_value
            time_for_percentile = x_below + (percentile - y_below) / slope

        if time_for_percentile < 40:
            return 1
        elif time_for_percentile >= 40 and time_for_percentile <=60:
            return 2
        elif time_for_percentile >60:
            return 3        
 
    def compute_thirds_rcg(self):
        
        def classify_rcg(rcg):
            if rcg < 0.33:
                return 1
            elif rcg >= 0.33 and rcg <=0.66:
                return 2
            elif rcg >0.66:
                return 3  

        time_delta = self.ts.data.index[1]-self.ts.data.index[0]
        time_delta_minuts = time_delta.seconds/60

        # first index over center of mass
        rcg_indeces = np.array([self.compute_rcg_idx(self.ts.data.loc[event[0]:event[1]].values) for event in self.ts.events])

        # time of center
        toc = np.array([self.ts.data.loc[event[0]:event[1]].index[rcg_indeces[i]] for i,event in enumerate(self.ts.events)])

        # duration until center
        tcg = np.array([time_delta_minuts + (toc[i] - event[0]).total_seconds()/60 for i,event in enumerate(self.ts.events)]).reshape(self.metrics["duration"].shape)

        # rcg
        rcg_array = tcg/self.metrics["duration"]
        
        thirds_rcg_classification = np.array([classify_rcg(rcg) for rcg in rcg_array])
        return thirds_rcg_classification
        
    def calc_dX_with_interpolation(self, dimensionless_event, percentile):

        time_percentage = (np.arange(0, len(dimensionless_event) + 1) / len(dimensionless_event)) * 100

        # Find the indices where the cumulative rainfall crosses the percentile_value
        indices_below = np.where(dimensionless_event < percentile)[0]
        indices_above = np.where(dimensionless_event >= percentile)[0]

        # Ensure there are indices both below and above the percentile value
        if len(indices_below) > 0 and len(indices_above) > 0:
            index_below = indices_below[-1]  # Last index below the percentile value
            index_above = indices_above[0]    # First index above the percentile value

            # Perform linear interpolation to find the exact intersection point
            x_below = time_percentage[index_below]
            y_below = dimensionless_event[index_below]

            x_above = time_percentage[index_above]
            y_above = dimensionless_event[index_above]

            # Calculate the slope
            slope = (y_above - y_below) / (x_above - x_below)
            # Use the formula to find the exact x value where the y value equals percentile_value
            time_for_percentile = x_below + (percentile - y_below) / slope

            return time_for_percentile            


    def calculate_tci(self, series):
        """
        Calculate the Temporal Concentration Index (TCI) for a given rainfall event.

        Parameters:
            rainfall (list or np.array): A sequence of rainfall values over time.

        Returns:
            max_tci (float): The highest TCI value for the event.
            best_center (int): The index of the temporal center corresponding to max_tci.
            tci_values (list): List of TCI values for each possible temporal center.
        """

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

        padded_precip = self.ts.data
        events_list = self.ts.events

        # resolution    
        time_delta = padded_precip.index[1]-padded_precip.index[0]
        time_delta_minuts = time_delta.seconds/60

        #####################################
        # Properties of Events to calculate
        #####################################

        # Max intensity [mm/min]
        self.metrics["max_intensity"] = np.array([padded_precip.loc[event[0]:event[1]].max().values[0] for event in events_list])/time_delta_minuts
        
        # Min intensity [mm/min]
        self.metrics["min_intensity"] = np.array([padded_precip.loc[event[0]:event[1]].min().values[0] for event in events_list])/time_delta_minuts
        
        # Mean intensity [mm/min]
        self.metrics["mean_intensity"] = np.array([padded_precip.loc[event[0]:event[1]].mean().values[0] for event in events_list])/time_delta_minuts

        # Peak to mean ratio 
        self.metrics["pmr"] = self.metrics["max_intensity"]/self.metrics["mean_intensity"]
        
        # Realtive amplitude of rainfall intensity
        self.metrics["relative_amp"] = (self.metrics["max_intensity"] - self.metrics["min_intensity"])/self.metrics["mean_intensity"]
    
        # Time to peak intensity [min]
        self.metrics["ttp"] = np.array([
            time_delta_minuts + np.float32(
                ((padded_precip.loc[event[0]:event[1]].idxmax() - event[0]).values.item()) / (60 * 10**9)
            ) for event in events_list
        ])


    # Duration [min]
        self.metrics["duration"] = np.array([time_delta_minuts+(event[1]-event[0]).days*24*60+(event[1]-event[0]).seconds/60 for event in events_list]).reshape(self.metrics["ttp"].shape)
        
        # Rainfall peak coefficient
        #TODO

        # Sum of precipiation [mm]
        self.metrics["total_precip"] = np.array([padded_precip.loc[event[0]:event[1]].sum().values[0] for event in events_list])

        # Standard deviation [mm/min]
        self.metrics["std"] = np.array([padded_precip.loc[event[0]:event[1]].std().values[0] for event in events_list])/np.square(time_delta_minuts)
        
#         # Number of wet/dry times - intermittency
        self.metrics["intermittency"] = np.array([self.compute_intermittency(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
    
        self.metrics["event_dry_ratio"] = np.array([self.calculate_event_dry_ratio(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
        
#         # Skewness
        self.metrics["skewness"] = np.array([padded_precip.loc[event[0]:event[1]].skew().values[0] for event in events_list])

        # Kurtosis
        self.metrics["kurtosis"] = np.array([padded_precip.loc[event[0]:event[1]].kurtosis().values[0] for event in events_list])

        # Peak position reatio (r)
        self.metrics["ppr"] = self.metrics["ttp"]/self.metrics["duration"]

         # Centre of Gravity Position Indicator (rcg)
        self.metrics["rcg"] = self.compute_rcg()

        # Thirds based on centre of mass
        self.metrics["thirds_rcg"] = self.compute_thirds_rcg()
    
        # Mass distribution indicators shape: #event rows and 5 columns for (m1..m5). 
        temp = np.array([self.compute_mass_dist_indicators(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
        self.metrics["m1"] = temp[:,0]
        self.metrics["m2"] = temp[:,1]
        self.metrics["m3"] = temp[:,2]
        self.metrics["m4"] = temp[:,3]
        self.metrics["m5"] = temp[:,4]
        
        # Mass distribution indicators shape: #event rows and 5 columns for (m1..m5). 
        temp = np.array([self.compute_frac_in_quarters(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
        self.metrics["frac_q1"] = temp[:,0]
        self.metrics["frac_q2"] = temp[:,1]
        self.metrics["frac_q3"] = temp[:,2]
        self.metrics["frac_q4"] = temp[:,3]   
        
        # Indicators based on dividing event into low and high parts
        temp = np.array([self.high_low_zone_indicators(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
        self.metrics["frac_time_in_high_intensity_zone"] = temp[:,0]
        self.metrics["frac_time_in_low_intensity_zone"] = temp[:,1]
        self.metrics["frac_rain_in_high_intensity_zone"] = temp[:,2]
        self.metrics["mean_intensity_high_intensity_zone"] = temp[:,3]
        
        # Rainfall Intensity Irregularity (ni)
        self.metrics["ni"] = np.array([padded_precip.loc[event[0]:event[1]].max().values[0]/padded_precip.loc[event[0]:event[1]].mean().values[0] for event in events_list])

        # I30
        self.metrics["I30"] = np.array([padded_precip.loc[event[0]:event[1]].rolling(window="30min").sum().max().values[0] for event in events_list])/30

         # Time based skewness
        self.metrics["time_based_skewness"] = self.compute_time_based_skewness()

        # Time based kurtosis
        self.metrics["time_based_kurtosis"] = self.compute_time_based_kurtosis()

#         #####################################
#         # Rainfall metrics
#         #####################################

        # huff quantiles
        self.metrics["4th_w_peak"] = np.array([self.fourth_with_peak(dimensionless_event) for dimensionless_event in self.ts.dimensionless_events])

        # quintile class
        self.metrics["5th_w_peak"] = np.array([self.fifth_with_peak(dimensionless_event) for dimensionless_event in self.ts.dimensionless_events])
        
        # quintile class
        self.metrics["3rd_w_peak"] = np.array([self.third_with_peak(dimensionless_event) for dimensionless_event in self.ts.dimensionless_events])
                        
        # third with highest percent rain
        self.metrics["third_class_max_percent"] = np.array([self.third_classification_max_percent(dimensionless_event) for dimensionless_event in self.ts.dimensionless_events])
        
        # third with center of mass
        self.metrics["third_class_CoM"] = np.array([self.third_classification_CoM(dimensionless_event) for dimensionless_event in self.ts.dimensionless_events])
        
        self.metrics["D50"] = np.array([self.calc_dX_with_interpolation(dimensionless_event, 0.5) for dimensionless_event in self.ts.dimensionless_events])      
        
        self.metrics["T25"] = np.array([self.calc_dX_with_interpolation(dimensionless_event, 0.25) for dimensionless_event in self.ts.dimensionless_events])              
        
        self.metrics["T50"] = np.array([self.calc_dX_with_interpolation(dimensionless_event, 0.5) for dimensionless_event in self.ts.dimensionless_events])       
        
        self.metrics["T75"] = np.array([self.calc_dX_with_interpolation(dimensionless_event, 0.75) for dimensionless_event in self.ts.dimensionless_events])    
        
        # Calculate ARR thids (based on location of D50
        self.metrics["ARR_Thirds"] = np.array([self.calc_ARR_thirds(dimensionless_event) for dimensionless_event in self.ts.dimensionless_events])   
        
        # calculate BSC
        temp = np.array([self.classify_BSC(dimensionless_event, False) for dimensionless_event in self.ts.dimensionless_events])         
        self.metrics["BSC"] = temp[:,0]
        self.metrics["BSC_Index"] = temp[:, 1].astype(int)
        
         # Skewness
        self.metrics["skewp"] = np.array([self.calculate_skew_p(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
        
         # NRMSE_Peak
        self.metrics["NRMSE_P"] = np.array([self.calculate_nrmse_peak(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
        
         # Event loading 
        self.metrics["event_loading_ghanghas"] = np.array([self.calculate_event_loading(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
        
         # Asymmetry of dependence
#         self.metrics["asymm_dependence"] = np.array([self.calculate_event_asymmetry(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
        
        # TCI
        # self.metrics["TCI"] = np.array([self.calculate_tci(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
        
        #####################################
        # Metrics from other disciplines
        #####################################
        # Gini coefficient
        self.metrics["gini"] = np.array([self.gini_coef(padded_precip.loc[event[0]:event[1]].values) for event in events_list])

        # Lorentz asymmetry coefficient
        self.metrics["lorentz_asymetry"] = np.array([self.lorentz_asymmetry(padded_precip.loc[event[0]:event[1]].values) for event in events_list])
        

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


