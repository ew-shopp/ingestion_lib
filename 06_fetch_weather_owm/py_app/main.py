
from datetime import date
from datetime import datetime

from weather.weather import WeatherExtractor, WeatherApi
from ewshopp_area import EwArea

import pandas as pd
import sys
import json, collections
import os

from weather_forecast_2_json import Weather_forecast_2_json



## python main.py ${work_path_tmp} ${work_path_results} 

work_path_tmp = sys.argv[1]
out_result_base = sys.argv[2]

input_type = os.environ.get('WE_INPUT_TYPE',"env")


date_format = '%Y-%m-%d'

if input_type == "env":
    # getting inputs from environment variables
    
    owm_key =  os.environ.get('WE_OWM_KEY','Missing')
    region_csv_file =  os.environ.get('WE_REGION_CSV_FILE','unknown.csv')
    forecast_days = os.environ.get('WE_FORECAST_DAYS',16)
    json_format = os.environ.get('WE_JSON_OUTPUT',False)

    key_sequence = (os.environ.get('WE_KEY_SEQUENCE')).split(',')
    new_key_sequence = (os.environ.get('WE_NEW_KEY_SEQUENCE')).split(',')
    if( (len(key_sequence) != len(new_key_sequence))):
        print "WE_KEY_SEQUENCE and WE_NEW_KEY_SEQUENCE must have the same number of elements"
        sys.exit(1)
    key_map = collections.OrderedDict(zip(key_sequence, new_key_sequence))

else:
    print "WE_INPUT_TYPE environment variable should be set to env "
    sys.exit(1)

# Use current date and number of days requested in forecast_days
start_date = datetime.now().date()
start_date_string = start_date.strftime(date_format)

print "<%s> " % (start_date_string)

file_suffix = '_combined.json' if json_format else '_combined.csv'

print 'opening OWM API'
wa = WeatherApi(source='owm', key=owm_key)

print 'get_forecast per entry'
csv = pd.read_csv(region_csv_file, header=None)

we = WeatherExtractor()
wf2j = Weather_forecast_2_json(we, json_format, key_map,
            file_name_combined=out_result_base + '/' + start_date_string + file_suffix)
            
for i in range(len(csv)):
    print 'Region: i %d %s' % (i, csv.loc[i,1])
    owm_json_filename = work_path_tmp + '/' + csv.loc[i,1] + '.json'

    ur = [csv.loc[i,4], csv.loc[i,5]]
    ll = [csv.loc[i,6], csv.loc[i,7]]
    lat = (ur[0] + ll[0]) / 2
    lon = (ur[1] + ll[1]) / 2

    print 'Requesting data'
    # download forecast data 
    wa.get(latlon=(lat, lon), 
           target=owm_json_filename)


    # load forecasted weather data
    print 'load data file'
    we.load([owm_json_filename])

    if(i == 0):
        wf2j.fetch_forecast_region_owm(start_date, start_date, csv.loc[i], True)
    else:
        wf2j.fetch_forecast_region_owm(start_date, start_date, csv.loc[i], False)
            
print 'Ending program'



