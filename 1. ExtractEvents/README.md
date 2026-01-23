# Extract events
How to run the code:

1. open mobaxterm
2. navigate to directory containing file (cd /nfs/a319/gy17m2a/Metrics/scripts/1. ExtractEvents)
3. conda activate ukcp18
4. ./run_5mins.sh 
    - This loops through all the rain gauges in the directory which stores the rain gauge data. 
    - For each rain gauge it opens a screen session and runs FindEvents_5mins.py  
5. /run_othermins.sh
    - Need to specify the variable $temp_res$
    - This loops through all the rain gauges in the directory which stores the rain gauge data. 
    - For each rain gauge it opens a screen session and runs FindEvents_Othermins.py  
    
    
## Screen commands
killall screen : kills all screens running
screen -r : checks which screens are currently running    