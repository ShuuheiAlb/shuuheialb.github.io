import json
import requests
import sys
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# Path functions
station_path = "/station"
weather_station_path = "/weather/station"
def get_location_path (id):
    return f"/location/{id}"
def get_weather_station_path (station_code):
    return f"/weather/station/{station_code}"

# Response function
def opennem_response(path, params={}, exception_flag=True):
    url = "https://api.opennem.org.au" + path
    response = requests.get(url, params=params)
    if exception_flag:
        response.raise_for_status()
    return response

# For debug
def json_print(data):
    print(json.dumps(data, indent=2))

def exploratory_test():
    path = "/weather/station"
    json_print(opennem_response(path).json())
    sys.exit()

#exploratory_test()

try:
    # Start small: Solar PV in SA

    # Compile station list
    data = opennem_response(station_path).json()["data"]
    stations = set()
    for entry in data:
        station_code = entry["code"]
        location_id = entry["location_id"]
        for facility in entry["facilities"]:
            if not ("fueltech" in facility and facility["fueltech"]["code"] == "solar_utility"):
                continue
            if not (facility["network_region"] == "SA1"):
                continue
            stations.add((station_code, location_id))

    # Compile weather station list
    data = opennem_response(weather_station_path).json()
    weather_stations = []
    for entry in data:
        if entry["state"] == "SA":
            weather_station_code = entry["code"]
            lattitude = entry["lat"]; longitude = entry["lng"]
            weather_stations.append((weather_station_code, longitude, lattitude))

    # Energy supply data for each station, hourly for 5 year
    # Dataframe will be plant code, date, energy, temperature + weather, lattitude + location
    energy_supply_df = pd.DataFrame(columns=["Name", "Date", "Energy", "Lattitude"])
    for station in stations:
        station_code, location_id = station
        ebs_path = f"/stats/energy/station/NEM/{station_code}"
        params = {
            "interval": "1d",
            "period": "1Y"
        }
        response = opennem_response(ebs_path, params, False)
        if response.status_code != 200: # Some just do not have statistics available
            continue
        data = response.json()["data"]

        # Get the station's coordinate, and the closest weather station's temperature
        location_record = opennem_response(get_location_path(location_id)).json()["record"]
        station_coord = (location_record["lng"], location_record["lat"])
        #from math import dist
        #min([dist(station_coord, (ws[1], ws[2])) for ws in weather_stations])

        # Create dataframe
        for entry in data: # maybe factorise
            if entry["data_type"] == "energy":
                single_plant_supply_list = entry["history"]["data"]
                single_plant_supply_df = pd.DataFrame({"Name": [station_code] * len(single_plant_supply_list),
                                            "Date" : range(len(single_plant_supply_list)),
                                            "Energy": entry["history"]["data"],
                                            "Longitude": [location_record["lng"]] * len(single_plant_supply_list),
                                            "Lattitude": [location_record["lat"]] * len(single_plant_supply_list)})
                energy_supply_df = pd.concat([energy_supply_df, single_plant_supply_df])
    # Aggregate sum for all plant codes
    energy_supply_df = energy_supply_df.groupby(["Name", "Date", "Lattitude"])["Energy"].sum().reset_index()
    
    print(energy_supply_df)



except requests.exceptions.RequestException as e:
    print(f"Error: {e}")