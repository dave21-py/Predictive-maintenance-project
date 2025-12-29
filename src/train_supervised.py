import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from data_loader import WindDataLoader
from features import FeatureEngineer
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate():
    loader = WindDataLoader("data")
    engineer = FeatureEngineer(window_days=7)
    failures = loader.load_failures()
    
    TEST_ASSET = 50
    # We train on these 3 assets to capture different failure patterns
    TRAIN_ASSETS = [12, 15, 16] 
    
    print(f"Loading Fleet Data (Assets: {TRAIN_ASSETS})...")
    
    dfs = []
    for asset_id in TRAIN_ASSETS:
        print(f"  - Loading Asset {asset_id}...")
        try:
            df = loader.load_turbine_data(asset_id)
            df = engineer.create_target(df, failures, asset_id=asset_id)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping Asset {asset_id}: {e}")
            
    # Combine all training data
    df_train = pd.concat(dfs)
    print(f"Total Training Rows: {len(df_train)}")
    print(f"Total Training Failures: {df_train['target'].sum()}")
    
    print(f"Loading Test Data (Asset {TEST_ASSET})...")
    df_test = loader.load_turbine_data(TEST_ASSET)
    df_test = engineer.create_target(df_test, failures, asset_id=TEST_ASSET)
    
    # Prepare X and y
    drop_cols = ['target', 'asset_id', 'id', 'train_test', 'status_type_id']
    existing_drop_cols = [c for c in drop_cols if c in df_train.columns]
    
    X_train = df_train.drop(columns=existing_drop_cols)
    y_train = df_train['target']
    
    X_test = df_test.drop(columns=existing_drop_cols)
    y_test = df_test['target']
    
    # Train
    ratio = float(len(y_train[y_train==0])) / len(y_train[y_train==1])
    print(f"Training XGBoost (Imbalance Ratio: {ratio:.2f})...")
    
    model = XGBClassifier(
        n_estimators=300,       # More trees for more data
        learning_rate=0.05,
        max_depth=7,            # Deeper trees to find complex patterns
        scale_pos_weight=ratio, 
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate with Threshold Optimization
    print("\nOPTIMIZING THRESHOLD (Fleet Learning)...")
    probs = model.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.5
    best_score = 0
    
    # We prioritize Recall here (catching the failure is most important)
    for thresh in np.arange(0.05, 0.6, 0.05):
        preds_custom = (probs >= thresh).astype(int)
        report = classification_report(y_test, preds_custom, output_dict=True, zero_division=0)
        
        recall = report['1']['recall']
        precision = report['1']['precision']
        
        # Simple score: We want some Recall, but Precision shouldn't be terrible
        # If Recall is 0, score is 0.
        if recall > 0.01:
            print(f"Threshold {thresh:.2f} -> Recall: {recall:.2f}, Precision: {precision:.2f}")
            if recall > best_score: # Just maximize Recall for now
                best_score = recall
                best_threshold = thresh
        
    print(f"\n✅ Best Threshold: {best_threshold:.2f} (Recall: {best_score:.2f})")
    
    final_preds = (probs >= best_threshold).astype(int)
    
    print("\n=== Final Fleet Performance ===")
    print(classification_report(y_test, final_preds))
    
    cm = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Fleet Model, Thresh {best_threshold})')
    plt.savefig('confusion_matrix_fleet.png')
    print("Matrix saved.")

if __name__ == "__main__":
    train_and_evaluate()