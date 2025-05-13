# #!/bin/bash

# # Directory with CSV files
# source_dir="/nfs/a319/gy17m2a/MetricEvaluation/DanishRainData_SVK"
# # Path to the Python script
# script_path="/nfs/a319/gy17m2a/MetricEvaluation/scripts/FindEvents_5mins.py"

# directory="DanishRainData_SVK" 

# conda activate ukcp18;

# # Loop through each file in the destination directory
# for file_path in "$source_dir"/*; do
#     # Get filename only
#     file=$(basename "$file_path")
#     name_no_ext="${file%.*}"

#     echo "Launching screen session for $file"

#     screen -dmS "session_$name_no_ext" /bin/bash -c "
#         cd $(dirname "$script_path") &&
#         python $(basename "$script_path") $file $directory
#     "
# done

# echo "All screen sessions launched."


# # Directory with CSV files
# source_dir="/nfs/a319/gy17m2a/MetricEvaluation/DanishRainData"
# # Path to the Python script
# script_path="/nfs/a319/gy17m2a/MetricEvaluation/scripts/FindEvents_5mins.py"

# directory="DanishRainData" 

# conda activate ukcp18;

# # Loop through each file in the destination directory
# for file_path in "$source_dir"/*; do
#     # Get filename only
#     file=$(basename "$file_path")
#     name_no_ext="${file%.*}"

#     echo "Launching screen session for $file"

#     screen -dmS "session_$name_no_ext" /bin/bash -c "
#         cd $(dirname "$script_path") &&
#         python $(basename "$script_path") $file $directory
#     "
# done

# echo "All screen sessions launched."


#!/bin/bash


# Ensure conda can be activated in scripts
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust this if your conda path is different
conda activate ukcp18

# Define directories and corresponding script/config values
declare -a dirs=(
#    "/nfs/a319/gy17m2a/MetricEvaluation/DanishRainData_SVK"
    "/nfs/a319/gy17m2a/MetricEvaluation/DanishRainData"
)
declare -a scripts=(
   "/nfs/a319/gy17m2a/MetricEvaluation/scripts/Run_5mins/FindEvents_5mins.py"
#     "/nfs/a319/gy17m2a/MetricEvaluation/scripts/Run_5mins/FindEvents_5mins.py"
)
declare -a labels=(
#    "DanishRainData_SVK"
    "DanishRainData"
)

# Maximum number of concurrent Screen sessions
max_sessions=15

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
            cd $(dirname "$script_path") &&
            python $(basename "$script_path") \"$file\" \"$directory_label\"
        "
    done
done

echo "All Screen sessions launched."
