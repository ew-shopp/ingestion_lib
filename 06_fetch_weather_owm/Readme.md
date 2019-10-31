# The weather extractor OWM - OpenWeatherMap

This script fetches weather date for city locations specified in CSV file.
Each city is a separate request to OWM and results in todays forcast stored in a JSON file.
The forecast for all cities are collected and transformed that in one CSV (default) or json file.

## Using it

The takes as arguments:
* the job description (json format) file with path to city list and output format specification
* the name of the outputfile (without extension)
* (optionally) the name of the input json file with the forecast parameters

The forecast parameters can be passed via a json file or via environment variables. The default is via environment variables. Json is not supported. If the variable is not set or set to "env", the script will take the parameters from  environment variables.

Below is a description of the parameters to be passed via environment variables:

* WE_INPUT_TYPE: to be set to "env" or "json". "env" means forecast parameters inputed via environment variables and "json" via json file.
* WE_OWM_KEY: OWM API key used when accessing the weather service 
* WE_REGION_CSV_FILE: csv file listing all the regions and their coordinates 
* WE_KEY_SEQUENCE: specify which weather forecast fields of the grib will be shown on the csv and their order. The fields must be separated by commas (no spaces)
* WE_NEW_KEY_SEQUENCE: this is used to possibly rename the forecast fields when writting the csv. It must constain the same number of elements as the WE_KEY_SEQUENCE. And be empty for the fields not to be renamed. The position field will be expanded and renamed to "Latitude" and "Longitude" regardless of the WE_NEW_KEY_SEQUENCE.
* WE_JSON_OUTPUT: if set to true, the output forecast will be written to a json file instead of a cvs. The default is false.

### Example of parameter settings:

```
WE_INPUT_TYPE=env
WE_OWM_KEY=lkljrjwerlkj134u9024
WE_REGION_CSV_FILE=path-to/city.csv
WE_KEY_SEQUENCE=2d,2t,10u,10v,ptype,sd,sf,sund,ssr,sp,tcc,tp,vis,ws,rh,cityName,region,strRegion,geonameId,position,validTime,validityDateTime
WE_NEW_KEY_SEQUENCE=,,,,,,,,,,,,,,,,,,,,,
```

