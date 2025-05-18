#!/bin/bash

# Directory containing CSV or pickle files
source_dir="/nfs/a319/gy17m2a/Metrics/DanishRainDataPickles"

# Output directory and temporal resolution
output_dir="/nfs/a319/gy17m2a/Metrics/DanishRainData_Outputs"
temp_res=10

# Activate the conda environment
# conda activate ukcp18

# Maximum number of concurrent Screen sessions
max_sessions=30

# Loop through each file in the directory
for file_path in "$source_dir"/*; do
    # Extract the filename and stem
    file=$(basename "$file_path")
    file_stem="${file%.pkl}"
    session_name="event_${file_stem}"

    # Output file path to check
    output_file="${output_dir}/${temp_res}mins/All_events_${file_stem}"

    # Skip if output file already exists
    if [ -f "$output_file" ]; then
        echo "Skipping $file_stem — output already exists."
        continue
    fi

    # Wait until the number of active Screen sessions is below the maximum
    while [ "$(screen -ls | grep -c 'event_')" -ge "$max_sessions" ]; do
        sleep 5
    done

    echo "Launching Screen session: $session_name"

    # Start a detached Screen session running the Python script
    screen -dmS "$session_name" python FindEvents_OtherMins.py "$file" "$temp_res"
done

