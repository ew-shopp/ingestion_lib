# This script requires an ECMWF API-key to operate
# It shall be located in the file $HOME/.ecmwfapirc

#contents of $HOME/.ecmwfapirc (Unix/Linux)
#{
#    "url"   : "https://api.ecmwf.int/v1",
#    "key"   : "XXXXXXXXXXXXXXXXXXXXXX",
#    "email" : "john.smith@example.com"
#}

from datetime import date
from datetime import datetime

from weather.weather import WeatherExtractor, WeatherApi
from ewshopp_area import EwArea

import pandas as pd
import sys

from weather_forecast_2_json import Weather_forecast_2_json

## python main.py ${job_json_file} ${region_csv_file} ${work_path_results} ${work_path_grib}
job_json_file = sys.argv[1]    
region_coordinates_csv_file = sys.argv[2]    
out_result_base = sys.argv[3]
grib_result_base = sys.argv[4]


grib_file = grib_result_base+'.grib'

job = pd.read_json(job_json_file, typ="series", convert_dates=False)

date_format = '%Y-%m-%d'
start_date_string = str(job.loc['start_date'])
end_date_string = str(job.loc['end_date'])
print "<%s> <%s>" % (start_date_string, end_date_string)

start_date = datetime.strptime(start_date_string.strip(), date_format).date()
end_date = datetime.strptime(end_date_string.strip(), date_format).date()

## Hack for requesting one day
## For a get request the start_date has to be before the end_date. Resulting in minimum two days
## For a fetch request the start_date and end_date can be equal. Resulting in minimum one day
start_get_date = (pd.Timestamp(start_date, freq='D')-1).date()

print "grib:%s  region_coordinates:%s out_result_base:%s grib_result_base:%s" % (grib_file, region_coordinates_csv_file, out_result_base, grib_result_base)
print "Get data from ECMWF - start_get_date:%s  end_date:%s" % (start_get_date.strftime(date_format), end_date.strftime(date_format))

print 'Requesting germay forecast'

wa = WeatherApi()

# download forecast data 
wa.get(from_date=start_get_date, to_date=end_date,
       area=EwArea.Germany, 
       target=grib_file)

print 'Stored forecast'


print "Extracting data from grib file based on Region bounding box coordinates"
print "start_date:%s  end_date:%s" % (start_date.strftime(date_format), end_date.strftime(date_format))
# query the downloaded data

# load forecasted weather data
print 'load grib file'
we = WeatherExtractor()
we.load([grib_file])

print 'get_forecast per entry'
csv = pd.read_csv(region_coordinates_csv_file, header=None)

wf2j = Weather_forecast_2_json(we,
            file_name_combined=out_result_base+'_combined.json')
            
for i in range(len(csv)):
    print 'Region: i %d %s' % (i, csv.loc[i,1])
    wf2j.fetch_forecast_region(start_date, end_date, csv.loc[i])
            
print 'Ending program'



