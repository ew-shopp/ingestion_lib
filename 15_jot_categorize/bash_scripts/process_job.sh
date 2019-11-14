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
echo "Categorizer: JOB_CAT_FASTEXT_FULL_PATH: ${JOB_CAT_FASTEXT_FULL_PATH}"
echo "Categorizer: JOB_CAT_EMBEDDER_FULL_PATH: ${JOB_CAT_EMBEDDER_FULL_PATH}"
echo "Categorizer: JOB_CAT_CATEGORIES_FULL_PATH: ${JOB_CAT_CATEGORIES_FULL_PATH}"
echo "Categorizer: JOB_CAT_CATEGORIES_COLUMN: ${JOB_CAT_CATEGORIES_COLUMN}"
echo "Categorizer: JOB_CAT_CATEGORIES_ID_COLUMN: ${JOB_CAT_CATEGORIES_ID_COLUMN}"
echo "Categorizer: JOB_CAT_KW_DELIMITER: ${JOB_CAT_KW_DELIMITER}"
echo "Categorizer: JOB_CAT_KEYWORDS_COLUMN: ${JOB_CAT_KEYWORDS_COLUMN}"

echo "Normalizer: jar_full_path: ${JOB_NOR_JAR_FULL_PATH}"
echo '***'

echo '#'
echo '#  Starting Process: Add categories'
echo '#'


# Construct Paths
file_name=${work_path##*/}
file_name_no_ext=${file_name%.*}
work_path_tmp=${work_directory}/tmp
work_path_results=${work_directory}/results

echo "!! work_path_results: "${work_path_results}

cat_path=${work_path_tmp}/cat_from_${file_name_no_ext}.csv
echo "!! categories tmp: "${cat_path}

cat_norm_path=${work_path_tmp}/cat_from_${file_name_no_ext}_norm.csv
echo "!! categories normalized tmp: "${cat_norm_path}

merged_path=${work_path_results}/${file_name_no_ext}_with_cat.csv
echo "!! file with categories: "${merged_path}

# Files are now in the work dir ... ready to be processed

# Making Directories
mkdir -p ${work_path_tmp}
mkdir -p ${work_path_results}

echo '#  Extracting categories'
python3 ${code_directory}/categoriser.py \
    ${JOB_CAT_FASTEXT_FULL_PATH} \
    ${JOB_CAT_EMBEDDER_FULL_PATH} \
    ${JOB_CAT_CATEGORIES_FULL_PATH} \
    ${work_path} \
    ${cat_path} \
    --n_categories 3 \
    --categories_delimiter ',' \
    --categories_column "${JOB_CAT_CATEGORIES_COLUMN}" \
    --categories_id_column "${JOB_CAT_CATEGORIES_ID_COLUMN}" \
    --keywords_delimiter "${JOB_CAT_KW_DELIMITER}" \
    --keywords_column "${JOB_CAT_KEYWORDS_COLUMN}"

RESULT=$?
if [ $RESULT -eq 0 ]; then
	echo '#  Normalizing categories'
	java -Xmx4g -jar ${JOB_NOR_JAR_FULL_PATH} ${cat_path} ${cat_norm_path}

	RESULT=$?
	if [ $RESULT -eq 0 ]; then
		echo '#  Joining categories to the input file'
		python3 ${code_directory}/csvjoin_dask.py \
		   --kw_delimiter "${JOB_CAT_KW_DELIMITER}" \
		   --kw_column "${JOB_CAT_KEYWORDS_COLUMN}" \
		   ${work_path} \
		   ${cat_norm_path} \
		   ${merged_path}

		RESULT=$?
		if [ $RESULT -eq 0 ]; then
			echo '#  Move the files to output'
			${code_directory}/move_to_output.sh \
			   ${output_directory} \
			   ${merged_path}
		fi
	fi
fi


echo '#'
echo '#  End Process'
echo '#'


