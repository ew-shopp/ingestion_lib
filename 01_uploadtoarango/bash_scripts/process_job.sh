#!/bin/bash
# arg1: work_path
# arg2: code directory
# arg3: input directory
# arg4: work directory
# arg5: output directory
# arg6: server_url
# arg7: usr_password
# arg8: db_name
# arg9: collection_name

work_path=${1}
code_directory=${2}
input_directory=${3}
work_directory=${4}
output_directory=${5}
server_addr=${6}
usr_pwd=${7}
db_name=${8}
collection_name=${9}

echo "work_path: ${work_path}"
echo "code_directory: ${code_directory}"
#echo "input_directory: ${input_directory}"
echo "work_directory: ${work_directory}"
echo "output_directory: ${output_directory}"
echo "server_addr: ${server_addr}"
echo "usr_pwd: ${usr_pwd}"
echo "db_name: ${db_name}"
echo "collection_name: ${collection_name}"
echo '***'

echo '#'
echo '#  Starting Process: UploadToArango'
echo '#'

# Construct Paths
file_name=${work_path##*/}
file_name_no_ext=${file_name%.*}
out_file_path=${work_directory}/${file_name_no_ext}.trig

# Debug: Show Paths
echo "!! out_file_path", ${out_file_path}

# File is in the work dir ... ready to be processed

# Upload the file
curl --data-binary "@${work_path}" -v -u "${usr_pwd}" -H "Content-Type: application/json" -X POST  "${server_addr}/_db/${db_name}/_api/import?collection=${collection_name}&type=auto&onDuplicate=error" 


touch ${out_file_path}
# Move the files to output
${code_directory}/move_to_output.sh ${output_directory} ${out_file_path}

echo '   Done'

