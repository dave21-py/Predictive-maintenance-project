import pandas as pd
import numpy as np
import os
import json


class WindDataLoader():
    def __init__(self, data_path, farm_name="Wind Farm C"):
        self.base_path = os.path.join(data_path, farm_name)
        self.dataset_path = os.path.join(self.base_path, "datasets")
        self.failures_path = os.path.join(self.base_path, "event_info.csv")
        self.features_path = os.path.join(self.base_path, "selected_features.json")

        # Load the selected features list created from the notebook
        with open(self.features_path, "r") as f:
            self.selected_features = json.load(f)
        
    def load_failures(self):
        """Loads the failure log"""
        df = pd.read_csv(self.failures_path, delimiter=";")
        return df

    def load_turbine_data(self, asset_id):
        """Loads sensor data for single turbine and flter features"""
        file_path = os.path.join(self.dataset_path, f"{asset_id}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data for asset {asset_id} not found.")


        cols_to_use = ['time_stamp'] + [c for c in self.selected_features if c!= 'time_stamp']

        # Read csv
        df = pd.read_csv(file_path, delimiter=";", usecols=lambda c: c in cols_to_use)

        # Parse dates
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        df = df.sort_values('time_stamp').set_index('time_stamp')
        return df


if __name__ == "__main__":
    loader = WindDataLoader("data")
    print(f"Loaded {len(loader.selected_features)} features.")

    df = loader.load_turbine_data(50)
    print(f"Asset 50 data shape: {df.shape}")

