from datetime import date
from weather.weather import WeatherExtractor, WeatherApi
import pandas as pd

from weather_combined import Weather_combined
from weather_separate import Weather_separate

class Weather_forecast_2_json:
    def __init__(self, we, json_format=False, key_map=None, forecast_days=16,
                 file_name_combined=None, file_name_separate=None):
        self._we = we
        self._json_format = json_format
        self._key_map = key_map
        self._forecast_days = 16 # Forecase the next 16 days (probably only 8 present)
        if forecast_days is not None:
            self._forecast_days = forecast_days

        self._fc = None
        if file_name_combined != None: 
            self._fc = open(file_name_combined, 'w')
            
        self._fs = None
        if file_name_separate != None: 
            self._fs = open(file_name_separate, 'w')
            
        self._file_name_combined = file_name_combined
        self._file_name_separate = file_name_separate

    
    def __del__(self):
        if self._fc != None: 
            print 'Closing file %s' % self._file_name_combined
            self._fc.close()
        if self._fs != None: 
            print 'Closing file %s' % self._file_name_separate
            self._fs.close()
    
    def fetch_forecast(self, from_date, to_date, city_elem, print_header):
        base_dates = pd.date_range(from_date, to_date)
        self._fetch_forecast_daterange(base_dates, city_elem, print_header)

    def fetch_forecast_region(self, from_date, to_date, region_elem, print_header):
        base_dates = pd.date_range(from_date, to_date)
        self._fetch_forecast_region_daterange(base_dates, region_elem, print_header)

    def _print_data(self, weather_data):
        for row in weather_data.iterrows():
            # row is tuple (index, columns)
            measure = row[1]
            print "Measurement of %s at from %s for %s" % (measure['shortName'], measure['validDateTime'], measure['validityDateTime'])
            for lat, lon, val in zip(measure['lats'], measure['lons'], measure['values']):
                print "%f N %f S = %f" % (lat, lon, val)

    def _fetch_forecast_daterange(self, base_dates, city_elem, print_header):

        wc = Weather_combined()
        ws = Weather_separate()
        points = [{'lat': city_elem.loc[1], 'lon': city_elem.loc[2]}]
        for bd in base_dates:
            base_date = bd.date()
            base_date_plus_N = (bd+self._forecast_days).date() # Forecast the next N days
            weather_data = self._we.get_forecast(base_date=base_date, 
                from_date=base_date, to_date=base_date_plus_N,
                aggtime='hour', aggloc='points',
                interp_points=points)
        
            wc.add_city_weather_data(city_elem.loc[0], weather_data)
            ws.add_city_weather_data(city_elem.loc[0], weather_data)
            
        if self._fc != None: 
            self._write_to_json(self._fc, wc, print_header)
        if self._fs != None: 
            self._write_to_json(self._fs, ws, print_header)
        

    def _fetch_forecast_region_daterange(self, base_dates, region_elem, print_header):

        wc = Weather_combined()
        ws = Weather_separate()
        ur = [region_elem.loc[4], region_elem.loc[5]]
        ll = [region_elem.loc[6], region_elem.loc[7]]
        # bounding box defined by its corner points with latitudes and longitudes: (45.45, 13.70) and (46.85, 16.56) """
        # bounding_box = [[45.45, 13.70], [46.85, 16.56]]
        bounding_box = [ur, ll]
        for bd in base_dates:
            base_date = bd.date()
            base_date_plus_N = (bd+self._forecast_days).date() # Forecase the next N days (probably only 8 present)
            weather_data = self._we.get_forecast(base_date=base_date, 
                from_date=base_date, to_date=base_date_plus_N,
                aggtime='hour', aggloc='bbox',
                bounding_box=bounding_box)

            wc.add_region_weather_data(region_elem.loc[1], region_elem.loc[0], region_elem.loc[3], weather_data)
            ws.add_region_weather_data(region_elem.loc[1], region_elem.loc[0], region_elem.loc[3], weather_data)

        if self._fc != None:
            self._write_to_file(self._fc, wc, print_header)
        if self._fs != None: 
            self._write_to_file(self._fs, ws, print_header)

    # return keys key_map also in dataframe
    def get_common_keys(self, df, drop_missing=True):
        common_keys = []
        for k in self._key_map.keys():
            if k in df.columns:
                common_keys.append(k)
            else: # check if append anyway
                if not drop_missing:
                    common_keys.append(k)
        return common_keys

    # Get the subset of the key_map dictionary that should be renamed
    def get_rename_dict(self, keys):
        rename_dict = {}
        for k in keys:
            v = self._key_map.get(k, None)
            if v is not None and len(v) > 0:
                rename_dict[k] = v
        return rename_dict

    def _write_to_file(self, f, w, print_header):
        # Read in full dataframe at once to simplify field filtering and renaming using pandas
        df = pd.DataFrame.from_dict(w.get_dict_values(), orient='columns')
        #print df.describe() # debug to verify dataset size



        # Get the specified keys and rename those specified
        if self._key_map is not None:
            c_keys = self.get_common_keys(df, True)
            df = df[c_keys]

            # This code breaks the position collumn which cotains strings of the form "[latitude, longitude]" into
            # collumns latitude and longitude. It replaces the position collumn with the new collumns in the dataframe
            # and set theirs index to Latitude and Longitude. Those indexes are not going to be renamed
            if 'position' in df.columns:
                col = df[['position']]
                lat = col['position'].apply(lambda x: x[0])

                longi = col['position'].apply(lambda x: x[1])

                df.insert(df.columns.get_loc('position'), 'Longitude', longi)
                df.insert(df.columns.get_loc('Longitude'), 'Latitude', lat)
                df.drop('position', axis=1, inplace=True)
            # end of position breakdown code

            # Get the dictionary subset that should be renamed
            rn_dict = self.get_rename_dict(c_keys)
            if len(rn_dict)>  0: # rename fields
                df=df.rename(index=str, columns=rn_dict)

        if self._json_format:
            df.to_json(path_or_buf=f, orient='records', date_format='iso', date_unit='s', lines=True)
        else:
            df.to_csv(path_or_buf=f, header=print_header, index=False, date_format='%Y-%m-%dT%H:%M:%SZ') # iso datespec format string




