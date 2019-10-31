#!/bin/bash
# arg1: work_path
# arg2: code directory
# arg3: input directory
# arg4: work directory
# arg5: output directory
# arg6: regions csv_file

work_path=${1}
code_directory=${2}
input_directory=${3}
work_directory=${4}
output_directory=${5}
regions_csv=${6}

echo "work_path: ${work_path}"
echo "code_directory: ${code_directory}"
#echo "input_directory: ${input_directory}"
echo "work_directory: ${work_directory}"
echo "output_directory: ${output_directory}"
echo "regions_csv: ${regions_csv}"
echo '***'

echo '#'
echo '#  Starting Process: ECMWF weather data'
echo '#'

# Construct Paths
file_name=${work_path##*/}
file_name_no_ext=${file_name%.*}
work_path_results=${work_directory}/${file_name_no_ext}_weather
work_dir_grib=${work_directory}/grib/
local_path_grib=grib/${file_name_no_ext}_weather


# Debug: Show Paths
#echo "!! work_path", ${work_path}
echo "!! work_path_results", ${work_path_results}
echo "!! work_dir_grib", ${work_dir_grib}
echo "!! work_path_grib", ${work_path_grib}

# File is in the work dir ... ready to be processed

# Make grib dir in work dir if not there
mkdir -p ${work_directory}/grib

echo "   Starting python script"
cd /weather/weather-data-master/weather_import

# Make grib dir locally if not there
mkdir -p grib

echo "Running:  python main.py ${work_path} ${regions_csv} ${work_path_results} ${local_path_grib} "
python main.py ${work_path} ${regions_csv} ${work_path_results} ${local_path_grib} 

# Move grib file to work dir
mv ${local_path_grib}*  ${work_dir_grib}

# Move the files to output
${code_directory}/move_to_output.sh ${output_directory} ${work_path_results}*

echo '   Done'

