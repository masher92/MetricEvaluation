#!/bin/bash

# Directory containing CSV or pickle files
source_dir="/nfs/a319/gy17m2a/MetricEvaluation/DanishRainDataPickles"

# Activate the conda environment
conda activate ukcp18

# Maximum number of concurrent Screen sessions
max_sessions=15
temp_res=30

# Loop through each file in the directory
for file_path in "$source_dir"/*; do
    # Extract the filename
    file=$(basename "$file_path")
    session_name="event_${file}"

    # Wait until the number of active Screen sessions is below the maximum
    while [ "$(screen -ls | grep -c 'event_')" -ge "$max_sessions" ]; do
        sleep 5
    done

    echo "Launching Screen session: $session_name"

    # Start a detached Screen session running the Python script
    screen -dmS "event_$filename" python FindEvents_OtherMins.py "$file" $temp_res
done