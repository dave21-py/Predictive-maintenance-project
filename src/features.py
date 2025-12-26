import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta

class FeatureEngineer():
    def __init__(self, window_days=7):
        self.window_days = window_days

    def create_target(self, df, failures_df, asset_id):
        """
        Create a binary target variable
        1 = Failure occurs 
        0 = Everything's fine
        """
        df['target'] = 0

        # Get the failures event for this specific asset
        asset_failures = failures_df[failures_df["asset_id"] == asset_id]

        for _, row in asset_failures.iterrows():
            failure_start = pd.to_datetime(row['event_start'])

            # Danger zone, 7 days before failure
            start_window = failure_start - timedelta(days=self.window_days)

            # 1
            mask = (df.index >= start_window) & (df.index < failure_start)
            df.loc[mask, 'target'] = 1
        return df

if __name__ == "__main__":
    from data_loader import WindDataLoader

    loader = WindDataLoader("data")
    df = loader.load_turbine_data(50)
    failures = loader.load_failures()

    engineer = FeatureEngineer(window_days=7)
    df_labeled = engineer.create_target(df, failures, asset_id = 50)

    print("Target Distribution")
    print(df_labeled['target'].value_counts())



