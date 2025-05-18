#!/bin/bash
# Ensure conda can be activated in scripts

# Define directories and corresponding script/config values
declare -a dirs=(
   "/nfs/a319/gy17m2a/Metrics/DanishRainData_SVK"
    "/nfs/a319/gy17m2a/Metrics/DanishRainData"
)
declare -a scripts=(
   "/nfs/a319/gy17m2a/Metrics/scripts/Run_5mins/FindEvents_5mins.py"
    "/nfs/a319/gy17m2a/Metrics/scripts/Run_5mins/FindEvents_5mins.py"
)
declare -a labels=(
   "DanishRainData_SVK"
    "DanishRainData"
)

# Maximum number of concurrent Screen sessions
max_sessions=30

# Loop over both directories
for i in "${!dirs[@]}"; do
    source_dir="${dirs[$i]}"
    script_path="${scripts[$i]}"
    directory_label="${labels[$i]}"

    for file_path in "$source_dir"/*; do
        file=$(basename "$file_path")
        name_no_ext="${file%.*}"
        session_name="session_${name_no_ext}"

        # Wait until the number of active Screen sessions is below the maximum
        while [ "$(screen -ls | grep -c 'session_')" -ge "$max_sessions" ]; do
            sleep 5
        done

        echo "Launching Screen session: $session_name"

        # Start a detached Screen session running the Python script
        screen -dmS "$session_name" /bin/bash -c "
            
            python FindEvents_5mins.py \"$file\" \"$directory_label\"
            
        "
    done
done

echo "All Screen sessions launched."