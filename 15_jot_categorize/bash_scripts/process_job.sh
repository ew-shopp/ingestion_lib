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
echo "Categorizer: JOB_CAT_KEYWORDS_COUNTRY_CODE: ${JOB_CAT_KEYWORDS_COUNTRY_CODE}"
if [ -z "${JOB_CAT_KEYWORDS_CHUNK_SIZE}"]; then
	JOB_CAT_KEYWORDS_CHUNK_SIZE=100000
	echo "**** Setting JOB_CAT_KEYWORDS_CHUNK_SIZE to default ****"
fi
echo "Categorizer: JOB_CAT_KEYWORDS_CHUNK_SIZE: ${JOB_CAT_KEYWORDS_CHUNK_SIZE}"
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

work_hdr_path=${work_path_tmp}/${file_name_no_ext}_work_hdr.csv
echo "!! work_hdr tmp: "${work_hdr_path}

cat_hdr_path=${work_path_tmp}/${file_name_no_ext}_curr_cat_hdr.csv
echo "!! categories_hdr tmp: "${cat_hdr_path}

cat_norm_path=${work_path_tmp}/${file_name_no_ext}_curr_cat_hdr_norm.csv
echo "!! categories normalized tmp: "${cat_norm_path}

merged_path=${work_path_tmp}/${file_name_no_ext}_cat_hdr.csv
echo "!! file with categories tmp: "${merged_path}

out_no_hdr_path=${work_path_results}/${file_name_no_ext}_cat.csv
echo "!! file with categories: "${out_no_hdr_path}

# Files are now in the work dir ... ready to be processed

# Making Directories
mkdir -p ${work_path_tmp}
mkdir -p ${work_path_results}

echo '#  Adding temp header needed for internal operations'
echo "c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19" > ${work_hdr_path}
cat ${work_path} >> ${work_hdr_path}

echo '#  Extracting categories'
python3 ${code_directory}/categoriser.py \
    ${JOB_CAT_FASTEXT_FULL_PATH} \
    ${JOB_CAT_EMBEDDER_FULL_PATH} \
    ${JOB_CAT_CATEGORIES_FULL_PATH} \
    ${work_hdr_path} \
    ${cat_hdr_path} \
    --n_categories 3 \
    --categories_delimiter ',' \
    --categories_column "${JOB_CAT_CATEGORIES_COLUMN}" \
    --categories_id_column "${JOB_CAT_CATEGORIES_ID_COLUMN}" \
    --keywords_delimiter ',' \
    --keywords_column 'c15' \
    --keywords_filter_column 'c9' \
    --keyword_chunk_size ${JOB_CAT_KEYWORDS_CHUNK_SIZE} \
    --keywords_filter_starts_with ${JOB_CAT_KEYWORDS_COUNTRY_CODE}

RESULT=$?
if [ $RESULT -eq 0 ]; then
	echo '#  Normalizing categories'
	java -Xmx4g -jar ${JOB_NOR_JAR_FULL_PATH} ${cat_hdr_path} ${cat_norm_path}

	RESULT=$?
	if [ $RESULT -eq 0 ]; then
		echo '#  Joining categories to the input file'
		python3 ${code_directory}/csvjoin_dask.py \
		   --kw_delimiter ',' \
		   --kw_column 'c15' \
		   --kw_filter_column 'c9' \
		   --kw_filter_starts_with ${JOB_CAT_KEYWORDS_COUNTRY_CODE} \
		   ${work_hdr_path} \
		   ${cat_norm_path} \
		   ${merged_path}

		RESULT=$?
		if [ $RESULT -eq 0 ]; then
			echo '#  Stripping temp header'
			tail -n +2 ${merged_path} > ${out_no_hdr_path}

			RESULT=$?
			if [ $RESULT -eq 0 ]; then
				echo '#  Move the files to output'
				${code_directory}/move_to_output.sh \
				   ${output_directory} \
				   ${out_no_hdr_path}

				RESULT=$?
				if [ $RESULT -eq 0 ]; then
					echo '#  Removing temporary files'
					rm ${work_hdr_path}
					rm ${cat_hdr_path}
					#rm ${cat_norm_path}
					rm ${merged_path}
				fi
			fi
		fi
	fi
fi


echo '#'
echo '#  End Process'
echo '#'


