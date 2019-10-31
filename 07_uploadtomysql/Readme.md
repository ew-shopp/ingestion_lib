# The weather extractor

This script fetchesa csv file with one header row and upload the data to a MySQL database table

## Using it

The script will monitor the input dir for csv files to upload.

A set of environment variables are neded to access the database

* WE_SERVER_ADDR: IP address of name to the MySQL server
* WE_SERVER_USR: Username
* WE_SERVER_PWD: Password
* WE_DB_NAME: Database name.
* WE_TABLE_NAME: Table name in the database to upload to.
* WE_TABLE_COLUMNS: Column names to use. Has to be a comma separated list.

### Example of parameter settings:

```
WE_SERVER_ADDR=192.168.1.21
WE_SERVER_USR=my_user
WE_SERVER_PWD=xyzz
WE_DB_NAME=test_import
WE_TABLE_NAME=weather_data
WE_TABLE_COLUMNS=(storeId, date, temp_avg, temp_max, temp_min, precip_tot, cloud_cov_avg)
```

