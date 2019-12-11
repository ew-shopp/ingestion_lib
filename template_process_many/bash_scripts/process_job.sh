#!/bin/bash
# arg1: work_path
# arg2: code directory
# arg3: input directory
# arg4: work directory
# arg5: output directory

work_path=${1}
code_directory=${2}
input_directory=${3}
work_directory=${4}
output_directory=${5}

echo "work_path: ${work_path}"
echo "code_directory: ${code_directory}"
echo "input_directory: ${input_directory}"
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


