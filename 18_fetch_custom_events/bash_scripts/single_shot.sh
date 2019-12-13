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
echo "input_directory: ${input_directory}"
echo "work_directory: ${work_directory}"
echo "output_directory: ${output_directory}"

echo "checking configuration file in input directory"
nconfFiles="$(find "${input_directory}" -name "*.properties" | wc -l | tr -d '[:space:]' )"
if [[ "${nconfFiles}" -gt "0" ]]; then
        # Extract File Name in random pos
        file_num=`shuf -i1-${nconfFiles} -n1`
        config_file="$(find "${input_directory}" -name "*.properties" | head "-${file_num}" | tail -1)"
        echo "// Found ${nconfFiles} Files"
        echo "// Picking the configuration file num ${file_num}"
        echo "// File to process ${config_file}"
fi

echo '***'
echo '#'
echo '#  Starting Process: ...'
echo '#'

# Do processing here
echo "   Starting java application"
echo "Running:  java -jar ${code_directory}/app.jar --working_path=${work_directory} --spring.config.location=file:///${config_file} --results_dir=run"
java -Djava.security.egd=file:/dev/./urandom -jar ${code_directory}/app.jar --working_path=${work_directory} --spring.config.location=file:///${config_file} --results_dir=run


# Move the file(s) to output
${code_directory}/move_to_output.sh ${output_directory} ${work_directory}/run/*

rm -rf ${work_directory}/run

echo '#'
echo '#  End Process'
echo '#'

