#!/bin/bash
# arg1: work_path
# arg2: code directory
# arg3: input directory
# arg4: work directory
# arg5: output directory
# arg6: regions csv_file

code_directory=${3}
input_directory=${4}
work_directory=${5}
output_directory=${6}

echo "code_directory: ${code_directory}"
echo "input_directory: ${input_directory}"
echo "work_directory: ${work_directory}"
echo "output_directory: ${output_directory}"
echo '***'

echo '#'
echo '#  Starting Process: Building model for closest keywords'
echo '#'

# Construct Paths
work_path_results=${work_directory}/results

# Debug: Show Paths
echo "!! work_path_results = " ${work_path_results}

# File is in the work dir ... ready to be processed

# Make work sub dir if not there
mkdir -p ${work_path_results}

echo '#  Calculating model'
python3 ${code_directory}/entry_point.py

# Move the file(s) to output
${code_directory}/move_to_output.sh ${output_directory} ${work_directory}/*

rm -rf ${work_directory}/run

echo '#'
echo '#  End Process'
echo '#'
