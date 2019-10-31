#!/bin/bash
# arg1: run_file_name              
# arg2: retry_max_count
# arg3: code directory
# arg4: input directory
# arg5: work directory
# arg6: output directory

code_directory=${3}
input_directory=${4}
work_directory=${5}
output_directory=${6}

echo "code_directory: ${code_directory}"
#echo "input_directory: ${input_directory}"
echo "work_directory: ${work_directory}"
echo "output_directory: ${output_directory}"
echo '***'

echo '#'
echo '#  Starting Process: Fetch and extract OWM weather data'
echo '#'

# Construct Paths
curr_time="$(date '+%s')"
work_path_results=${work_directory}/results_${curr_time}
work_path_tmp=${work_directory}/tmp_${curr_time}

# Debug: Show Paths
echo "!! work_path_results", ${work_path_results}
echo "!! work_path_tmp", ${work_path_tmp}

# This is a single shot process without any input files
# All input is in environment vars


# Make work sub dir if not there
mkdir -p ${work_path_tmp}
mkdir -p ${work_path_results}

echo "   Starting python script"
cd /weather/weather-data-master/weather_import

echo "Running:  python main.py ${work_path_tmp} ${work_path_results}  "
python main.py ${work_path_tmp} ${work_path_results}  

# Move the files to output
${code_directory}/move_to_output.sh ${output_directory} ${work_path_results}/*

echo '   Done'

