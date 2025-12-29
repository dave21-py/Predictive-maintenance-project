import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

class SupervisedModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("SupervisedModel")
        self.model = None
        self.features = [] 

    def train(self, df_train):
        """
        Trains XGBoost on the Fleet Data.
        """
        self.logger.info(f"Training XGBoost on {len(df_train)} rows...")
        
        # --- FIX: ROBUST FEATURE SELECTION ---
        # 1. Select only Numeric columns
        # 2. Drop known metadata columns explicitly
        exclude_cols = ['target', 'asset_id', 'group_id', 'status_type_id', 'train_test']
        
        # Filter: Must be numeric AND not in the exclude list
        self.features = [c for c in df_train.columns 
                         if c not in exclude_cols 
                         and pd.api.types.is_numeric_dtype(df_train[c])]
        
        self.logger.info(f"Training on {len(self.features)} features.")
        
        X_train = df_train[self.features]
        y_train = df_train['target']
        
        # Handle Imbalance
        neg = len(y_train[y_train == 0])
        pos = len(y_train[y_train == 1])
        ratio = neg / pos if pos > 0 else 1
        
        self.logger.info(f"Imbalance Ratio: {ratio:.2f}")
        
        # Initialize & Fit
        self.model = XGBClassifier(
            n_estimators=self.config['models']['supervised'].get('n_estimators', 100),
            scale_pos_weight=ratio,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        self.logger.info("Training complete.")

    def evaluate(self, df_test):
        """
        Tests the model on Asset 50.
        """
        self.logger.info("Evaluating Supervised Model on Test Asset...")
        
        # Use exact same features as training
        X_test = df_test[self.features]
        y_test = df_test['target']
        
        y_pred = self.model.predict(X_test)
        
        print("\n=== SUPERVISED MODEL PERFORMANCE (XGBoost) ===")
        print(classification_report(y_test, y_pred))
        print(f"Recall: {recall_score(y_test, y_pred):.2f}")
        print("==============================================")
        
        self._plot_confusion_matrix(y_test, y_pred)

    def _plot_confusion_matrix(self, y_true, y_pred):
        save_path = self.config['paths']['outputs']
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
        plt.title('Confusion Matrix (Supervised)')
        plt.savefig(os.path.join(save_path, 'confusion_matrix_supervised.png'))
        plt.close()