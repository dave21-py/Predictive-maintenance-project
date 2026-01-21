import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

class DeepLearningDataUtils:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        # Remove features with 0 variance (constants)
        self.selector = VarianceThreshold(threshold=0.0)
        self.selected_features = []

    def prepare_train_val_loaders(self, df):
        """
        SPLITS data into Train and Validation.
        Removes constant columns to prevent NaN errors.
        """
        drop_cols = ['time_stamp', 'asset_id', 'target', 'train_test', 'status_type_id']
        # 1. Initial selection of numeric columns
        features = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        X = df[features].values
        
        # 2. DROP CONSTANT COLUMNS (The Fix for NaNs)
        X_selected = self.selector.fit_transform(X)
        
        # Save which features we kept so we can use them during testing
        support = self.selector.get_support(indices=True)
        self.selected_features = [features[i] for i in support]
        
        print(f"DEBUG: Dropped {len(features) - len(self.selected_features)} constant features.")
        
        # 3. Scale Data (Now safe from Divide-by-Zero)
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 4. Convert to Tensor
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        full_dataset = TensorDataset(tensor_x, tensor_x)
        
        # 5. Split
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['models']['vae']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['models']['vae']['batch_size'], shuffle=False)
        
        return train_loader, val_loader, self.selected_features

    def prepare_test_loader(self, df, feature_cols):
        """
        Prepares test data using the SCALER from training.
        """
        # Ensure we look at the exact same features
        X = df[feature_cols].values
        
        # Transform (Scale)
        X_scaled = self.scaler.transform(X)
        
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_x)
        
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        return loader