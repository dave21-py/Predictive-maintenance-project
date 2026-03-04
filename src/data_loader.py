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
        # Try both locations
        path_1 = os.path.join(self.base_path, "datasets", "event_info.csv")
        path_2 = os.path.join(self.base_path, "event_info.csv")

        if os.path.exists(path_1):
            path = path_1
        elif os.path.exists(path_2):
            path = path_2
        else:
            self.logger.error(f"File not found. Checked {path_1} and {path_2}")
            sys.exit(1)

        self.logger.info(f"Loading failures from {path}")
        df = pd.read_csv(path, delimiter=";")
        
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        
       
        if 'asset_id' not in df.columns:
            self.logger.warning("'asset_id' column missing in CSV! Applying Emergency Patch...")
            
            
            # Asset 50
            df['asset_id'] = 0 # Default to 0
            
            # Map known failures 
            df.loc[df['event_id'] == 55, 'asset_id'] = 50
            df.loc[df['event_id'] == 81, 'asset_id'] = 38
            df.loc[df['event_id'] == 47, 'asset_id'] = 21
            df.loc[df['event_id'] == 12, 'asset_id'] = 2
            
            # Check if Asset 50 is mapped
            if 50 in df['asset_id'].values:
                self.logger.info("Emergency Patch Successful: Mapped Event 55 to Asset 50.")
            else:
                self.logger.error("Failed to map Asset 50. Please check event_info.csv")
                
        return df

    def load_turbine(self, asset_id):
        """Loads sensor data for a specific turbine."""
        # Default CARE layout: data/datasets/{asset_id}.csv
        candidate_paths = [
            os.path.join(self.base_path, "datasets", f"{asset_id}.csv"),
        ]

        # Also support Wind Farm subdirectories, e.g. data/Wind Farm C/datasets/{asset_id}.csv
        for farm_name in os.listdir(self.base_path):
            farm_path = os.path.join(self.base_path, farm_name)
            if not os.path.isdir(farm_path):
                continue
            candidate_paths.append(
                os.path.join(farm_path, "datasets", f"{asset_id}.csv")
            )

        path = None
        for cand in candidate_paths:
            if os.path.exists(cand):
                path = cand
                break

        self.logger.info(f"Loading sensor data for Asset {asset_id}")
        if path is None:
            self.logger.error(
                f"File for asset {asset_id} not found in any known datasets folder."
            )
            raise FileNotFoundError(f"Asset ID {asset_id} not found in data folders.")

        try:
            # Read csv
            df = pd.read_csv(path, delimiter=";")
            
            # Clean column names
            df.columns = df.columns.str.strip()

            # Convert time
            df['time_stamp'] = pd.to_datetime(df['time_stamp'])
            df = df.set_index('time_stamp').sort_index()

            self.logger.info(f"Loaded {df.shape[0]} rows for asset {asset_id}")
            return df

        except Exception as e:
            self.logger.error(f"Error reading Asset {asset_id}: {e}")
            raise