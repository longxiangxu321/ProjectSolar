from pvlib import solarposition
import pandas as pd
import numpy as np
import pytz
import json
from datetime import datetime
import os
from cjio import cityjson


def obtain_tud(config):
    tz = config["timezone"]

    lat, lon = config["lat"], config["long"]
    times = pd.date_range(config["start_time"], config["end_time"], freq=config["frequency"], tz=tz)
    solpos = solarposition.get_solarposition(times, lat, lon)
    # remove nighttime
    solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]

    solpos.reset_index(inplace=True)
    solpos.columns.values[0] = 'timestep'
    solpos.columns.values[0]='timestep'
    ground_recording_filename = os.path.join(config["data_root"], "measured_data.csv")
    ground_recording = pd.read_csv(ground_recording_filename)
    solpos['timestep'] = pd.to_datetime(solpos['timestep']).dt.tz_localize(None)
    ground_recording['local_time'] = pd.to_datetime(ground_recording['local_time'])
    filtered_ground_recording = ground_recording.drop_duplicates(subset='local_time', keep='first').reset_index(drop=True)
    filtered_sunpos = solpos[solpos['timestep'].isin(filtered_ground_recording['local_time'])]
    filtered_sunpos.loc[:, 'timestep'] = filtered_sunpos['timestep'].dt.tz_localize(tz)

    filtered_sunpos.set_index('timestep', inplace=True)
    print("Number of sun positions: ", len(filtered_sunpos))
    return filtered_sunpos

def obtain_regular(config):
    tz = config["timezone"]

    lat, lon = config["lat"], config["long"]
    times = pd.date_range(config["start_time"], config["end_time"], freq=config["frequency"], tz=tz)
    solpos = solarposition.get_solarposition(times, lat, lon)
    # remove nighttime
    solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]

    return solpos
     

def main():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)

    with open('config.json', 'r') as file:
        CONFIG = json.load(file)
    
    cfg = CONFIG["study_area"]

    scenario = CONFIG["scenario"]
    if scenario == "tud":
        print("obtaining solar position in tud scenario")
        solpos = obtain_tud(cfg)
    else:
        print("obtaining solar position in regular scenario")
        solpos = obtain_regular(cfg)



    current_time = datetime.now()

    # Format the datetime to the specified format
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

    # Concatenate with the "output" string
    result_string = "output_" + formatted_time

    result_dir = os.path.join(cfg["data_root"], result_string)
    intermediate_dir = os.path.join(result_dir, "intermediate")

    CONFIG["output_folder_name"] = result_string

    os.makedirs(intermediate_dir, exist_ok=True)
    solar_path = os.path.join(intermediate_dir, "sun_pos.csv")
    solpos.to_csv(solar_path)

    cfg_path = os.path.join(result_dir, "config.json")
    with open(cfg_path, 'w') as file:
        json.dump(CONFIG, file, indent=4)

    with open('config.json', 'w') as file:
        json.dump(CONFIG, file, indent=4)

    print("created folder for the simulation: ", result_dir)
    # Print the result
    print("saving solar pos to: ",solar_path)


if __name__ == '__main__':
        main()