import json
import requests
import sys
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

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
    path = "/fueltechs"
    json_print(opennem_response(path).json())
    sys.exit()

#exploratory_test()

# SOON: use Pandas
try:
    # Make a GET request to retrieve station data
    station_path = "/station"
    data = opennem_response(station_path).json()["data"]
    network_station_map = {}
    for entry in data:
        station_code = entry["code"]
        for facility in entry["facilities"]:
            # Limit for now to stations using non-rooftop solar PV
            if not ("fueltech" in facility and facility["fueltech"]["code"] == "solar_utility"):
                continue
            
            network_code = facility["network"]["code"]
            if network_code not in network_station_map:
                network_station_map[network_code] = []
            network_station_map[network_code].append(station_code)
    
    # For diversity, pick 2 from VIC, SA, WA, 3 from NSW, QLD

    # Retrieve energy supply data in each station: hourly for a year
    # Dataframe will be date, location, weather, energy
    x = pd.DataFrame()
    energy_supply_by_station = {}
    for network_code in network_station_map:
        for station_code in network_station_map[network_code]:
            energy_supply_path = f"/stats/energy/station/{network_code}/{station_code}"
            params = {
                "interval": "1h",
                "period": "1d"
            }
            response = opennem_response(energy_supply_path, params, False)
            if response.status_code != 200: # Some just do not have statistics available
                continue
        
            energy_supply_data = response.json()["data"]
            for entry in energy_supply_data:
                info_type = entry["data_type"]
                if info_type == "energy":
                    if station_code not in energy_supply_by_station:
                        energy_supply_by_station[station_code] = np.array(entry["history"]["data"])
                    energy_supply_by_station[station_code] += np.array(entry["history"]["data"])

    print(f"total {len(energy_supply_by_station)} stations")
    for s in energy_supply_by_station:
        print(f"{s} has {len(energy_supply_by_station[s])} points")

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")