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
echo "Categorizer: JOB_CAT_KEYWORDS_COUNTRY_CODE: ${JOB_CAT_KEYWORDS_COUNTRY_CODE}"
echo '***'

echo '#'
echo '#  Starting Process: Collecting unique keywords'
echo '#'


# Construct Paths
file_name=${work_path##*/}
file_name_no_ext=${file_name%.*}
work_path_tmp=${work_directory}/tmp
work_path_results=${work_directory}/results

echo "!! work_path_results: "${work_path_results}

work_hdr_path=${work_path_tmp}/${file_name_no_ext}_work_hdr.csv
echo "!! work_hdr tmp: "${work_hdr_path}

unique_keywords=${work_path_results}/unique_keywords.csv
echo "!! file with unique keywords: "${unique_keywords}

# Files are now in the work dir ... ready to be processed

# Making Directories
mkdir -p ${work_path_tmp}
mkdir -p ${work_path_results}

echo '#  Adding temp header needed for internal operations'
echo "c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19" > ${work_hdr_path}
cat ${work_path} >> ${work_hdr_path}

echo '#  Extracting keywords'
python3 ${code_directory}/aggregate_keywords.py \
    ${work_hdr_path} \
    ${unique_keywords} \
    --keywords_delimiter ',' \
    --keywords_column 'c15' \
    --keywords_filter_column 'c9' \
    --keywords_filter_starts_with ${JOB_CAT_KEYWORDS_COUNTRY_CODE}

RESULT=$?
if [ $RESULT -eq 0 ]; then
	echo '#  Removing temporary files'
	rm ${work_hdr_path}
fi

echo '#'
echo '#  End Process'
echo '#'


