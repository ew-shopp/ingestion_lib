import numpy as np

class Weather_combined:
    def __init__(self):
        self._dict = {}
        
    def increment_element(self, d, field):
        if field not in d:
            d[field] = 1
        else:
            d[field] = d[field] + 1
            
    def add_city_weather_data(self, city_name, weather_data):
        # make combined documents
        masked_with_nan = {}
        for row in weather_data.iterrows():
            # row is tuple (index, columns)
            measure = row[1]
            # print "Measurement of %s at from %s for %s" % (measure['shortName'], measure['validDateTime'], measure['validityDateTime'])
            for lat, lon, val in zip(measure['lats'], measure['lons'], measure['values']):
                # print "%f N %f S = %f" % (lat, lon, val)
                common = "%s,%f,%f,%s,%s" % (city_name, lat, lon, measure['validDateTime'], measure['validityDateTime'])
                if common not in self._dict:
                    self._dict[common] = {}
                    #print "Common: %s" % common
                self._dict[common]['cityName'] = city_name
                self._dict[common]['position'] = [lat, lon]
                self._dict[common]['validTime'] = measure['validDateTime']
                self._dict[common]['validityDateTime'] = measure['validityDateTime']
                if type(val) == np.ma.core.MaskedConstant:
                    self._dict[common][measure['shortName']+'masked'] = True
                    self._dict[common][measure['shortName']] = np.nan
                    self._dict[common]['value'] = np.nan
                    increment_element(masked_with_nan, measure['shortName'])
                else:
                    self._dict[common][measure['shortName']] = val
                    
        for key in masked_with_nan:
            print "Replacing masked element '%s' with NaN %d times" % (key, masked_with_nan[key])

            
    def add_region_weather_data(self, region, str_region, geoname_id, weather_data):
        # make combined documents
        masked_with_nan = {}
        for row in weather_data.iterrows():
            # row is tuple (index, columns)
            measure = row[1]
            # print "Measurement of %s at from %s for %s" % (measure['shortName'], measure['validDateTime'], measure['validityDateTime'])
            for lat, lon, val in zip(measure['lats'], measure['lons'], measure['values']):
                # print "%f N %f S = %f" % (lat, lon, val)
                common = "%s,%f,%f,%s,%s" % (region, lat, lon, measure['validDateTime'], measure['validityDateTime'])
                if common not in self._dict:
                    self._dict[common] = {}
                    #print "Common: %s" % common
                self._dict[common]['region'] = region
                self._dict[common]['strRegion'] = str_region
                self._dict[common]['geonameId'] = geoname_id
                self._dict[common]['position'] = [lat, lon]
                self._dict[common]['validTime'] = measure['validDateTime']
                self._dict[common]['validityDateTime'] = measure['validityDateTime']
                if type(val) == np.ma.core.MaskedConstant:
                    self._dict[common][measure['shortName']+'masked'] = True
                    self._dict[common][measure['shortName']] = np.nan
                    self._dict[common]['value'] = np.nan
                    increment_element(masked_with_nan, measure['shortName'])
                else:
                    self._dict[common][measure['shortName']] = val
                    
        for key in masked_with_nan:
            print "Replacing masked element '%s' with NaN %d times" % (key, masked_with_nan[key])

            
    def get_dict_values(self):           
        return self._dict.values()

    
