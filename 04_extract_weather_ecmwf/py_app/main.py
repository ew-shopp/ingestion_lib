
from datetime import date
from datetime import datetime

from weather.weather import WeatherExtractor, WeatherApi
from ewshopp_area import EwArea

import pandas as pd
import sys
import json, collections
import os

from weather_forecast_2_json import Weather_forecast_2_json



## python main.py ${job_json_file} ${region_csv_file} ${work_path_results} 

region_coordinates_csv_file = sys.argv[1]
out_result_base = sys.argv[2]
if len(sys.argv) == 4:
    job_json_file = sys.argv[3]

input_type = os.environ.get('WE_INPUT_TYPE',"env")


date_format = '%Y-%m-%d'

if input_type == "env":
    # getting inputs from environmen variables
    forecast_days = os.environ.get('WE_FORECAST_DAYS',16)
    start_date_string = os.environ.get('WE_START_DATE')
    end_date_string = os.environ.get('WE_END_DATE')
    json_format = os.environ.get('WE_JSON_OUTPUT',False)
    grib_file = os.environ.get('WE_GRIB_FILE')
    key_sequence = (os.environ.get('WE_KEY_SEQUENCE')).split(',')
    new_key_sequence = (os.environ.get('WE_NEW_KEY_SEQUENCE')).split(',')
    if( (len(key_sequence) != len(new_key_sequence))):
        print "WE_KEY_SEQUENCE and WE_NEW_KEY_SEQUENCE must have the same number of elements"
        sys.exit(1)
    key_map = collections.OrderedDict(zip(key_sequence, new_key_sequence))

elif input_type == "json":
    # getting inputs from json

    # use OrderedDict to preserve key order in json dicts in python 2.7
    with open(job_json_file) as json_file:
        job = json.load(json_file, object_pairs_hook=collections.OrderedDict)
    # print json.dumps(job, indent=2, separators=(',', ': ')) # debug printout
    forecast_days = int(job.get('forecast_days', 16))
    start_date_string = str(job['start_date'])
    end_date_string = str(job['end_date'])
    json_format = job.get('json_format', False)
    grib_file = str(job['grib_file'])

    key_map = job.get('key_sequence_map', None)

else:
    print "WE_INPUT_TYPE environment variable should be set to env or json"
    sys.exit(1)



print "<%s> <%s>" % (start_date_string, end_date_string)
print "Grib file:<%s>" % (grib_file)

file_suffix = '_combined.json' if json_format else '_combined.csv'

start_date = datetime.strptime(start_date_string.strip(), date_format).date()
end_date = datetime.strptime(end_date_string.strip(), date_format).date()

print "Extracting data from grib file based on Region bounding box coordinates"
print "start_date:%s  end_date:%s" % (start_date.strftime(date_format), end_date.strftime(date_format))
# query the downloaded data

# load forecasted weather data
print 'load grib file'
we = WeatherExtractor()
we.load([grib_file])

print 'get_forecast per entry'
csv = pd.read_csv(region_coordinates_csv_file, header=None)

wf2j = Weather_forecast_2_json(we, json_format, key_map,
            file_name_combined=out_result_base+file_suffix)
            
for i in range(len(csv)):
    print 'Region: i %d %s' % (i, csv.loc[i,1])
    if(i == 0):
        wf2j.fetch_forecast_region(start_date, end_date, csv.loc[i], True)
    else:
        wf2j.fetch_forecast_region(start_date, end_date, csv.loc[i], False)
            
print 'Ending program'



