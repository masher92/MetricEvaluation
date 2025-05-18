import pandas as pd
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
warnings.simplefilter(action='ignore', category=FutureWarning)

from ClassFunctions import precip_time_series, rainfall_analysis
# from PlottingFunctions import *

file=sys.argv[1]
directory = sys.argv[2]

print(file, directory)

temp_res = 5

all_events = []

if pd.read_csv(f"/nfs/a319/gy17m2a/Metrics/{directory}/{file}").empty:
    print("The CSV file has no data rows.")
else:   
    ts = precip_time_series(f"/nfs/a319/gy17m2a/Metrics/{directory}/{file}")
    if len(ts.data[ts.data['precipitation (mm/hr)']<0]) >0:
        print("Not including, still has negatives")
        negatives.append(file)
    else:
        ts.pad_and_resample(f'{temp_res}')

        # check if enough values
        dt_index = ts.data.index
        full_range = pd.date_range(start=dt_index.min(), end=dt_index.max(), freq=f'{temp_res}T')
        missing = full_range.difference(dt_index)
        print(f"Number of missing time steps: {len(missing)}")
        if len(missing) > 0:
            print("First few missing timestamps:")
            print(missing[:10])

        analysis = rainfall_analysis('11h', ts)

        if ts.events != None:
            with open(f'/nfs/a319/gy17m2a/Metrics/DanishRainDataPickles/{file}.pkl', 'wb') as f:
                pickle.dump(ts, f, 4)
            
            analysis.get_metrics()
            df = pd.DataFrame(analysis.metrics)
            df['gauge_num'] = file.split('_')[0]
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
    
            all_events_df.to_csv(f"/nfs/a319/gy17m2a/Metrics/DanishRainData_Outputs/{temp_res}mins/All_events_{file}", index=False)