from datetime import date
from weather.weather import WeatherExtractor, WeatherApi
import pandas as pd

from weather_combined import Weather_combined
from weather_separate import Weather_separate

class Weather_forecast_2_json:
    def __init__(self, we, file_name_combined=None, file_name_separate=None):
        self._we = we
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
    
    def fetch_forecast(self, from_date, to_date, city_elem):
        base_dates = pd.date_range(from_date, to_date)
        self._fetch_forecast_daterange(base_dates, city_elem)

    def fetch_forecast_region(self, from_date, to_date, region_elem):
        base_dates = pd.date_range(from_date, to_date)
        self._fetch_forecast_region_daterange(base_dates, region_elem)

    def _print_data(self, weather_data):
        for row in weather_data.iterrows():
            # row is tuple (index, columns)
            measure = row[1]
            print "Measurement of %s at from %s for %s" % (measure['shortName'], measure['validDateTime'], measure['validityDateTime'])
            for lat, lon, val in zip(measure['lats'], measure['lons'], measure['values']):
                print "%f N %f S = %f" % (lat, lon, val)

    def _fetch_forecast_daterange(self, base_dates, city_elem):

        wc = Weather_combined()
        ws = Weather_separate()
        points = [{'lat': city_elem.loc[1], 'lon': city_elem.loc[2]}]
        for bd in base_dates:
            base_date = bd.date()
            base_date_plus_16 = (bd+16).date() # Forecase the next 16 days (probably only 8 present)
            weather_data = self._we.get_forecast(base_date=base_date, 
                from_date=base_date, to_date=base_date_plus_16, 
                aggtime='hour', aggloc='points',
                interp_points=points)
        
            wc.add_city_weather_data(city_elem.loc[0], weather_data)
            ws.add_city_weather_data(city_elem.loc[0], weather_data)
            
        if self._fc != None: 
            self._write_to_json(self._fc, wc)
        if self._fs != None: 
            self._write_to_json(self._fs, ws)
        

    def _fetch_forecast_region_daterange(self, base_dates, region_elem):

        wc = Weather_combined()
        ws = Weather_separate()
        ur = [region_elem.loc[4], region_elem.loc[5]]
        ll = [region_elem.loc[6], region_elem.loc[7]]
        # bounding box defined by its corner points with latitudes and longitudes: (45.45, 13.70) and (46.85, 16.56) """
        # bounding_box = [[45.45, 13.70], [46.85, 16.56]]
        bounding_box = [ur, ll]
        for bd in base_dates:
            base_date = bd.date()
            base_date_plus_16 = (bd+16).date() # Forecase the next 16 days (probably only 8 present)
            weather_data = self._we.get_forecast(base_date=base_date, 
                from_date=base_date, to_date=base_date_plus_16, 
                aggtime='hour', aggloc='bbox',
                bounding_box=bounding_box)


            wc.add_region_weather_data(region_elem.loc[1], region_elem.loc[0], region_elem.loc[3], weather_data)
            ws.add_region_weather_data(region_elem.loc[1], region_elem.loc[0], region_elem.loc[3], weather_data)

        if self._fc != None: 
            self._write_to_json(self._fc, wc)
        if self._fs != None: 
            self._write_to_json(self._fs, ws)
        

    def _write_to_json(self, f, w):
        d_arr = w.get_dict_values()
        #print 'Total dict'
        #print d_arr
        for d in d_arr:
            #print 'For dict'
            #print d
            pds = pd.Series(d)
            json_str = pds.to_json(path_or_buf=None, orient='index', date_format='iso', date_unit='s')     
            print >> f, json_str
    


