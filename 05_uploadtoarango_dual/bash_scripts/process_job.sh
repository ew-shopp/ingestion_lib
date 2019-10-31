#!/bin/bash
# arg1: work_path
# arg2: code directory
# arg3: input directory
# arg4: work directory
# arg5: output directory
# arg6: server_url
# arg7: usr_password
# arg8: db_name
# arg9: doc_collection_name
# arg10: edge_collection_name
# arg11: from_prefix_name
# arg12: to_prefix_name

work_path=${1}
code_directory=${2}
input_directory=${3}
work_directory=${4}
output_directory=${5}
server_addr=${6}
usr_pwd=${7}
db_name=${8}
doc_collection_name=${9}
edge_collection_name=${10}
from_prefix_name=${11}
to_prefix_name=${12}

echo "work_path: ${work_path}"
echo "code_directory: ${code_directory}"
#echo "input_directory: ${input_directory}"
echo "work_directory: ${work_directory}"
echo "output_directory: ${output_directory}"
echo "server_addr: ${server_addr}"
echo "usr_pwd: ${usr_pwd}"
echo "db_name: ${db_name}"
echo "doc_collection_name: ${doc_collection_name}"
echo "edge_collection_name: ${edge_collection_name}"
echo "from_prefix_name: ${from_prefix_name}"
echo "to_prefix_name: ${to_prefix_name}"
echo '***'

echo '#'
echo '#  Starting Process: UploadToArango dual'
echo '#'

# Construct Paths
file_name=${work_path##*/}
file_name_no_ext=${file_name%.*}
out_file_path=${work_directory}/${file_name_no_ext}.trig

# Debug: Show Paths
echo "!! out_file_path", ${out_file_path}

# File is in the work dir ... ready to be processed

# Check filename to see if its edge or doc

regexp_edge='.*edge\.json'    # RegExp for checking edge document file
is_document="yes"             # Default to document collection file
work_path_lc=${work_path,,}   # Make filename lower case

if [[ ${work_path_lc} =~ ${regexp_edge} ]]; then
    echo "EDGE collection file detected"
    is_document="no"
fi

if [ $is_document == "yes" ]; then
    # Upload the file to document collection
    curl --data-binary "@${work_path}" -v -u "${usr_pwd}" -H "Content-Type: application/json" -X POST  "${server_addr}/_db/${db_name}/_api/import?collection=${doc_collection_name}&type=auto&onDuplicate=error" 
    
else

    # Upload the file to edge collection
    curl --data-binary "@${work_path}" -v -u "${usr_pwd}" -H "Content-Type: application/json" -X POST  "${server_addr}/_db/${db_name}/_api/import?collection=${edge_collection_name}&type=auto&onDuplicate=error&fromPrefix=${from_prefix_name}&toPrefix=${to_prefix_name}" 
fi

touch ${out_file_path}
# Move the files to output
${code_directory}/move_to_output.sh ${output_directory} ${out_file_path}

echo '   Done'

