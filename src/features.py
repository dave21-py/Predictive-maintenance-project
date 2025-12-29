import pandas as pd
import numpy as np
from datetime import timedelta
import logging


class FeatureEngineer:
    """
    Handles all data transformation. 
    Design Principle: 'Transformations should be stateless if possible.'
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Feature Engineer")
        self.window_days = config['feature_engineering']['window_days']
        self.rolling_windows = config['feature_engineering']['rolling_windows']

    def create_target(self, df, failures_df, asset_id):
        """
        Labels the data for evaluation/training.
        1 = Failure imminent (within window_days)
        0 = Normal operation
        """
        self.logger.info(f"Creating target labels for Asset {asset_id}...")
        
        df['target'] = 0

        asset_failures = failures_df[failures_df['asset_id'] == asset_id]

        if asset_failures.empty:
            self.logger.warning(f"No failures found for Asset {asset_id}. Target will be all 0s.")
            return df


        for _, row in asset_failures.iterrows():
            failure_start = pd.to_datetime(row['event_start'])

            # Predict it X days in advance
            start_window = failure_start - timedelta(days=self.window_days)

            # Masking for faster rows
            mask = (df.index >= start_window) & (df.index < failure_start)
            df.loc[mask, 'target'] = 1

        return df

    def create_rolling_features(self, df):
        """
        Generates statistical features (Mean, Std) to capture volatility.
        This handles the 'Silent Failure' problem where averages look normal
        but vibration (std deviation) is high.
        """
        self.logger.info("Generating rolling features")

        exclude = ['target', 'asset_id', 'group_id', 'status_type_id']
        sensor_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c not in exclude]

        new_features = []

        for col in sensor_cols:
            for w in self.rolling_windows:
                roller = df[col].rolling(window=w)



                # Std
                std = roller.std()
                std.name = f"{col}_std_{w}"
                
                # Mean = Trend
                mean = roller.mean()
                mean.name = f"{col}_mean_{w}"
                
                new_features.extend([std, mean])
        
        # 2. Concat everything at once (Prevents 'Fragmentation' warning)
        self.logger.info(f"Concatenating {len(new_features)} new features...")
        features_df = pd.concat(new_features, axis=1)
        
        # 3. Join back to original data
        df_final = pd.concat([df, features_df], axis=1)
        
        return df_final




