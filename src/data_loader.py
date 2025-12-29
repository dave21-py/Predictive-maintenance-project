import pandas as pd
import numpy as np
import os
import sys
import logging


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DataLoader")
        self.base_path = config['paths']['raw_data']

    def load_failures(self):
        """Loads the failure events file."""
        path = os.path.join(self.base_path, "datasets", "event_info.csv")

        # Check if file exists
        if not os.path.exists(path):
            path = os.path.join(self.base_path, "event_info.csv")
        
        if not os.path.exists(path):
            self.logger.error(f"File not found, Checked {path}")
            sys.exit(1)

        self.logger.info(f"Loading failures from {path}")
        df = pd.read_csv(path, delimiter=";")
        return df

    def load_turbine(self, asset_id):
        """Loads sensor data for a specific turbine."""
        path = os.path.join(self.base_path, "datasets", f"{asset_id}.csv")
        self.logger.info(f"Loading sensor data for Asset {asset_id}")
        if not os.path.exists(path):
            self.logger.error(f"File not found in {path}")
            raise FileNotFoundError(f"Asset ID {asset_id} not found.")

        try:
            # Read csv
            df = pd.read_csv(path, delimiter=";")

            # Convert time
            df['time_stamp'] = pd.to_datetime(df['time_stamp'])
            df = df.set_index('time_stamp').sort_index()

            self.logger.info(f"Loaded {df.shape[0]} rows for asset {asset_id}")
            return df

        except Exception as e:
            self.logger.error(f"Error reading Asset {asset_id}")
            raise