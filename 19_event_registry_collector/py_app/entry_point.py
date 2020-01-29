import argparse
import os
import configparser
import regex as re
from pprint import pprint
import sys
import json
from src.collector import eventregistry_collector as ercoll
from src.db_connector import arangodb_connector as arandb

# Get classes
class Luogo:
    def __init__(self, place):
        self.place = place
class Start:
    def __init__(self, inizio):
        self.inizio = inizio
class End:
    def __init__(self, fine):
        self.fine = fine

# Get available collector services
def get_avail_coll_services(cfg):
    """
    Extract fields from ini file about available collectors
    """
    coll_services = {}
    i = 1

    for section in cfg.sections():
        if 'collector' in section:
            coll_services[str(i)] = {
                "name": section.split('.',1)[1],
                "section": section,
                "accesstoken": cfg[section]["accesstoken"],
            }

            i += 1

    return coll_services

def coll_service_selection(avail_coll_services):
    """
    Let the user choose the service to which perform the query
    """
    #print("Please, select collector service (type related number key):")

    #for coll_serv_key in avail_coll_services:
        #print("[" + coll_serv_key + "] " + avail_coll_services[coll_serv_key]["name"]) # fare a dizionario e fare check sulle chiavi disponibili
        #print(coll_serv_key)
    coll_valid = False
    selected_collector = ""

    while not coll_valid:
        #print("Your selection:", end=" ")
        selected_collector = '1' # Solo event registry
        if selected_collector not in avail_coll_services:
            print("[Error] Your choice does not correspond to any of the available services, retry")
        else:
            coll_valid = True

    return avail_coll_services[selected_collector]

# Events collection
def collect_from_service(selected_collector, n_luoghi, n_date):
    """
    Query services to get events and collect results in a JSON file
    """

    # TODO: Way to do this in a non-hardcoded way?
    collector_cstr = {
        "eventregistry": ercoll.ERCollector,
    }

    accesstoken = selected_collector["accesstoken"]
    collector = collector_cstr[selected_collector["name"]](accesstoken)

    # ER query filtering
    filters = {}
    #print("\nSpecify filtering, leave blank if you do not want set a filter")
    #print("Location:", end=" ")

    # Filter location
    #filters["location"] = input()
    filters["location"] = luogo.place[n_luoghi]

    # Filter event date
    date_start_valid = False
    date_end_valid = False

    # TODO: IMPORTANT! Add dateStart and dateEnd
    while not (date_start_valid & date_end_valid):
        #print("Event start date (yyyy-MM-dd):", end=" ")
        filters["startDate"] = dateStart.inizio[n_date]

        pattern_date = re.compile("(^$|^\d{4}(-)(((0)[0-9])|((1)[0-2]))(-)([0-2][0-9]|(3)[0-1])$)")
        match_start_date = pattern_date.match(filters["startDate"])

        if match_start_date is not None:
            date_start_valid = True
        else:
            print("[ERROR] Input date invalid or format not recognized")


        #print("Event end date (yyyy-MM-dd):", end=" ")
        filters["endDate"] = dateEnd.fine[n_date]

        match_end_date = pattern_date.match(filters["endDate"])

        if match_end_date is not None:
            date_end_valid = True
        else:
            print("[ERROR] Input date invalid or format not recognized")

    for filter in filters:
        if filters[filter] == '':
            filters[filter] = None

    collector.exec_query(filters)

def conn_db_selection(avail_db_conn):
    """
    Let the user choose the db connector to interface with db engine
    """

    #print("Please, select collector service (type related number key):")

    #for conn_db_key in avail_db_conn:
        #print("[" + conn_db_key + "] " + avail_db_conn[conn_db_key]["name"]) # fare a dizionario e fare check sulle chiavi disponibili

    conn_db_valid = False
    selected_conn_db = ""

    while not conn_db_valid:
        #print("Your selection:", end=" ")
        selected_conn_db = input()
        if selected_conn_db not in avail_db_conn:
            print("[Error] Your choice does not correspond to any of the available connectors, retry")
        else:
            conn_db_valid = True

    return avail_db_conn[selected_conn_db]


# Read location and date
def LocationDate(cfg):
    """
    read configuration file
    """
    for section in cfg.sections():
        if 'Location' in section:
            luogo = Luogo(cfg.get(section, 'location').split('\n'))
        elif 'Date' in section:
            dateStart = Start(cfg.get(section, 'start_date').split('\n'))
            dateEnd = End(cfg.get(section, 'end_date').split('\n'))
    return luogo,dateEnd,dateStart


if __name__ == "__main__":

    # read config file
    cfg = configparser.ConfigParser()
    cfg.read('config/config.ini')
    luogo, dateEnd, dateStart = LocationDate(cfg)
    mode_download, mode_mapping = mode(cfg)
    for j in range(len(luogo.place)):
        for i in range(len(dateStart.inizio)):
        # init available collector services
            avail_coll_services = get_avail_coll_services(cfg)
            selected_collector = coll_service_selection(avail_coll_services)
            collect_from_service(selected_collector, j, i)
