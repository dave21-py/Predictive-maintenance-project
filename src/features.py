import pandas as pd
from datetime import timedelta

class FeatureEngineer():
    def __init__(self, window_days=14):
        self.window_days = window_days

    def create_rolling_features(self, df):
        """
        Creates rolling statistics efficiently using pd.concat
        """
        # 1. Identify sensor columns (exclude non-sensor data)
        # We assume sensors are float/int and not IDs/Targets
        exclude_cols = ['target', 'asset_id', 'id', 'status_type_id', 'group_id']
        sensor_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c not in exclude_cols]
        
        # Windows: 6 (1 hour), 144 (24 hours)
        windows = [6, 144] 
        
        new_features = []
        
        print(f"Generating rolling features for {len(sensor_cols)} sensors...")
        
        for col in sensor_cols:
            for w in windows:
                # Calculate Rolling objects once
                roller = df[col].rolling(window=w)
                
                # Create Series and name them
                std_series = roller.std()
                std_series.name = f'{col}_std_{w}'
                
                mean_series = roller.mean()
                mean_series.name = f'{col}_mean_{w}'
                
                new_features.extend([std_series, mean_series])

        # Concatenate all new features at once (Fixes Fragmentation)
        print("Concatenating features...")
        features_df = pd.concat(new_features, axis=1)
        
        # Merge back with original dataframe
        df_final = pd.concat([df, features_df], axis=1)
        
        return df_final

    def create_target(self, df, failures_df, asset_id):
        df['target'] = 0
        asset_failures = failures_df[failures_df["asset_id"] == asset_id]

        for _, row in asset_failures.iterrows():
            failure_start = pd.to_datetime(row['event_start'])
            # Wide window to capture the build-up
            start_window = failure_start - timedelta(days=self.window_days)
            
            mask = (df.index >= start_window) & (df.index < failure_start)
            df.loc[mask, 'target'] = 1
            
        return df