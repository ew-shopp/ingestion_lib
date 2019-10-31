#!/bin/bash
# arg1: work_path
# arg2: code directory
# arg3: input directory
# arg4: work directory
# arg5: output directory

work_path=${1}
code_directory=${2}
#input_directory=${3}
work_directory=${4}
#output_directory=${5}

server_addr=${WE_SERVER_ADDR}
usr=${WE_SERVER_USR}
usr_pwd=${WE_SERVER_PWD}
db_name=${WE_DB_NAME}
table_name=${WE_TABLE_NAME}
table_columns=${WE_TABLE_COLUMNS}

echo "work_path: ${work_path}"
echo "code_directory: ${code_directory}"
#echo "input_directory: ${input_directory}"
echo "work_directory: ${work_directory}"
#echo "output_directory: ${output_directory}"
echo "server_addr: ${server_addr}"
echo "usr: ${usr}"
#echo "usr_pwd: ${usr_pwd}"
echo "db_name: ${db_name}"
echo "table_name: ${table_name}"
echo "table_columns: ${table_columns}"
echo '***'

echo '#'
echo '#  Starting Process: UploadToMySql'
echo '#'

# Construct Paths
file_name=${work_path##*/}
file_name_no_ext=${file_name%.*}
out_file_path=${work_directory}/${file_name_no_ext}.trig

# Debug: Show Paths
echo "!! out_file_path", ${out_file_path}

# File is in the work dir ... ready to be processed

echo "Table size before upload"
mysql --host=${server_addr} --user=${usr} --password=${usr_pwd} --local-infile -e "SELECT count(*) from ${table_name};"  ${db_name} 

# Upload the file
mysql --host=${server_addr} --user=${usr} --password=${usr_pwd} --local-infile  -e "LOAD DATA LOW_PRIORITY LOCAL INFILE '${work_path}' INTO TABLE ${table_name} CHARACTER SET utf8 FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' ESCAPED BY '\"' LINES TERMINATED BY '\n' IGNORE 1 LINES ${table_columns};"  ${db_name}

echo "Table size after upload"
mysql --host=${server_addr} --user=${usr} --password=${usr_pwd} --local-infile -e "SELECT count(*) from ${table_name};"  ${db_name} 


touch ${out_file_path}
# Move the files to output
${code_directory}/move_to_output.sh ${output_directory} ${out_file_path}

echo '   Done'

