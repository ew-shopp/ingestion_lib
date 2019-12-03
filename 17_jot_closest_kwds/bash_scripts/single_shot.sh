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
echo '#  Starting Process: Building model for closest keywords'
echo '#'

# Construct Paths
work_path_results=${work_directory}/results

# Debug: Show Paths
echo "!! work_path_results", ${work_path_results}

# This is a single shot process without any input files
# All input is in environment vars
echo "Categorizer: JOB_CAT_ALL_KEYWORDS_FULL_PATH: ${JOB_CAT_ALL_KEYWORDS_FULL_PATH}"
echo "Categorizer: JOB_CAT_FASTEXT_FULL_PATH: ${JOB_CAT_FASTEXT_FULL_PATH}"
echo "Categorizer: JOB_CAT_EMBEDDER_FULL_PATH: ${JOB_CAT_EMBEDDER_FULL_PATH}"
echo "Categorizer: JOB_CAT_CATEGORIES_FULL_PATH: ${JOB_CAT_CATEGORIES_FULL_PATH}"
echo "Categorizer: JOB_CAT_CATEGORIES_COLUMN: ${JOB_CAT_CATEGORIES_COLUMN}"
echo "Categorizer: JOB_CAT_CATEGORIES_ID_COLUMN: ${JOB_CAT_CATEGORIES_ID_COLUMN}"
echo "Categorizer: JOB_CAT_N_KEYWORDS: ${JOB_CAT_N_KEYWORDS}"

out_model_path=${work_path_results}/closest_keywords_model.csv
echo "!! Output model: "${out_model_path}


# Make work sub dir if not there
mkdir -p ${work_path_results}

echo '#  Calculating model'
python3 ${code_directory}/categoriser.py \
    relevance_to_category \
    ${JOB_CAT_FASTEXT_FULL_PATH} \
    ${JOB_CAT_EMBEDDER_FULL_PATH} \
    ${JOB_CAT_CATEGORIES_FULL_PATH} \
    ${JOB_CAT_ALL_KEYWORDS_FULL_PATH} \
    ${out_model_path} \
    --n_keywords ${JOB_CAT_N_KEYWORDS} \
    --categories_delimiter ',' \
    --categories_column "${JOB_CAT_CATEGORIES_COLUMN}" \
    --categories_id_column "${JOB_CAT_CATEGORIES_ID_COLUMN}" \
    --keywords_delimiter ',' \
    --keywords_column 'keyword' 

# Move the files to output
${code_directory}/move_to_output.sh ${output_directory} ${out_model_path}

echo '   Done'

