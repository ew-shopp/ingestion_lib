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
echo '#  Starting Process: ...'
echo '#'

# Do processing here

# Move the file(s) to output
${code_directory}/move_to_output.sh ${output_directory} <output_file>

echo '#'
echo '#  End Process'
echo '#'

