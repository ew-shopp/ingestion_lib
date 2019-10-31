# The weather extractor

This script fetches the region weather forecast stored in the grib files and transforms that in a CSV (default) or json forecast for the cities inputed in an input CSV file.

## Using it

The takes as arguments:
* the coordinates of the cities to which one wants the forecast
* the name of the outputfile (without extension)
* (optionally) the name of the input json file with the forecast parameters

The forecast parameters can be passed via a json file or via environment variables. The default is via environment variables. If you want the parameters to be passed via json, you must set the environment variable WE_INPUT_TYPE to "json". If the variable is not set or set to "env", the script will take the parameters from  environment variables.

Below is a description of the parameters to be passed via environment variables:

* WE_INPUT_TYPE: to be set to "env" or "json". "env" means forecast parameters inputed via environment variables and "json" via json file.
* WE_START_DATE: start date in format YYYY-MM-DD
* WE_END_DATE: end date in format YYYY-MM-DD
* WE_GRIB_FILE: grib file with the forecast 
* WE_FORECAST_DAYS: number of forecast days
* WE_KEY_SEQUENCE: specify which weather forecast fields of the grib will be shown on the csv and their order. The fields must be separated by commas (no spaces)
* WE_NEW_KEY_SEQUENCE: this is used to possibly rename the forecast fields when writting the csv. It must constain the same number of elements as the WE_KEY_SEQUENCE. And be empty for the fields not to be renamed. The position field will be expanded and renamed to "Latitude" and "Longitude" regardless of the WE_NEW_KEY_SEQUENCE.
* WE_JSON_OUTPUT: if set to true, the output forecast will be written to a json file instead of a cvs. The default is false.

### Example of parameter settings:

```
WE_INPUT_TYPE=env
WE_START_DATE=2018-05-14
WE_END_DATE=2018-05-14
WE_GRIB_FILE=path-to/weather-slovenia_m5_y2018.grib
WE_FORECAST_DAYS=16
WE_KEY_SEQUENCE=2d,2t,10u,10v,ptype,sd,sf,sund,ssr,sp,tcc,tp,vis,ws,rh,cityName,region,strRegion,geonameId,position,validTime,validityDateTime
WE_NEW_KEY_SEQUENCE=,,,,,,,,,,,,,,,,,,,,,
```

Example of  [json with parameters](https://github.com/ew-shopp/weather_import/blob/master/docker_daily_import/04_extract_weather/job_selected_fields_slovenia.json) 