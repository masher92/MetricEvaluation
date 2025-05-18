import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import datetime 
import warnings
import sys
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)

from ClassFunctions_OtherRes import precip_time_series, rainfall_analysis
# from PlottingFunctions import *

temp_res = int(sys.argv[2])
pickle_str = sys.argv[1]
print(pickle_str)

if 'svk' in pickle_str:
    directory = 'DanishRainData_SVK'
else:
    directory = 'DanishRainData'    

all_events = []

file_name = pickle_str.split('.pkl')[0]
if os.path.isfile(f"/nfs/a319/gy17m2a/Metrics/DanishRainData_Outputs/{temp_res}mins/All_events_{file_name}"):
    print(f"{file_name} is already a file")
else:
    print(f"{file_name} is not already a file")

    with open(f'/nfs/a319/gy17m2a/Metrics/DanishRainDataPickles/{file_name}.pkl', 'rb') as f:
           five_min_pickle = pickle.load(f)
    five_min_events = five_min_pickle.events

    if len(five_min_events) !=0:

        # Get the timeseries
        print(f"/nfs/a319/gy17m2a/Metrics/{directory}/{file_name}")
        ts = precip_time_series(f"/nfs/a319/gy17m2a/Metrics/{directory}/{file_name}", temp_res)
        print(f"/nfs/a319/gy17m2a/Metrics/{directory}/{file_name}")
        # Resample to 30 minutes
        ts.pad_and_resample(f'{temp_res}')

        # Get the events (this should be based on the start and end time stamp from the corresponding 5 minute data )
        analysis = rainfall_analysis('11h', temp_res, ts, file_name)

        if ts.events != None:
            analysis.get_metrics(temp_res)
            df = pd.DataFrame(analysis.metrics)
            df['gauge_num'] = file_name.split('_')[0]
            all_events.append(df)
            all_events_df = pd.concat(all_events)

            # Add start and end times
            start_times = []
            end_times = []
            for timestamp in ts.events:
                start_times.append(timestamp[0])
                end_times.append(timestamp[1])
            all_events_df['start_time'] =start_times
            all_events_df['end_time'] =end_times
            print(f" Events before saving: {len(all_events_df)}")
            print('-------------------------------')
            all_events_df['event_num'] = ts.original_event_indices
            all_events_df.to_csv(f"/nfs/a319/gy17m2a/Metrics/DanishRainData_Outputs/{temp_res}mins/All_events_{file_name}", index=False)
    else:
        print("its empty")

         
                                                    
                                             