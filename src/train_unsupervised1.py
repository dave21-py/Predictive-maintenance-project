import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from data_loader import WindDataLoader
from features import FeatureEngineer

def train_and_evaluate_clustering():
    print("Loading Data (Asset 50)...")
    loader = WindDataLoader("data")
    engineer = FeatureEngineer(window_days=14)
    failures = loader.load_failures()
    
    # Load Asset 50
    df = loader.load_turbine_data(50)
    df = engineer.create_target(df, failures, asset_id=50)
    
    # 1. Feature Engineering
    df = engineer.create_rolling_features(df)
    
    # --- WP 1: Feature Selection ---
    print("\n--- Performing Feature Selection (Correlation Analysis) ---")
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corrwith(numeric_df['target']).abs()
    
    if 'target' in correlations:
        correlations = correlations.drop('target')
        
    top_features = correlations.sort_values(ascending=False).head(20)
    selected_features = top_features.index.tolist()
    
    print("Top 5 Predictive Features found:")
    print(top_features.head(5))
    print(f"Selecting Top {len(selected_features)} features for the model.")
    
    X = df[selected_features].ffill().bfill().fillna(0)
    y_true = df['target']
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Train Isolation Forest
    print(f"Training Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=200, contamination='auto', random_state=42, n_jobs=-1)
    iso_forest.fit(X_scaled)
    scores = iso_forest.decision_function(X_scaled)
    
    # --- WP 2: Optimization ---
    print("\n--- Optimizing Threshold for High Recall ---")
    target_recall = 0.80
    best_threshold = 0
    final_percentile = 0
    
    for p in range(15, 51, 1):
        thresh = np.percentile(scores, p)
        y_tmp = [1 if s < thresh else 0 for s in scores]
        rec = recall_score(y_true, y_tmp)
        if rec >= target_recall:
            best_threshold = thresh
            final_percentile = p
            print(f"Target Reached! Percentile: {p}% -> Recall: {rec:.2f}")
            break
            
    if best_threshold == 0:
        print(f"Target not reached. Using max sensitivity (50%).")
        final_percentile = 50
        best_threshold = np.percentile(scores, 50)
    
    print(f"Final Applied Threshold: {best_threshold:.4f} (Top {final_percentile}% Anomalies)")
    y_pred = [1 if s < best_threshold else 0 for s in scores]
    
    # 4. Evaluate
    print("\n=== Unsupervised Performance ===")
    rec = recall_score(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    print(f"FINAL RECALL: {rec:.2f}")
    
    # 5. Plot
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, -scores, label='Anomaly Score', color='purple', alpha=0.6, linewidth=0.5)
    # Corrected variable name here:
    plt.axhline(-best_threshold, color='orange', linestyle='--', label=f'Threshold')
    
    failure_regions = df[df['target'] == 1].index
    if len(failure_regions) > 0:
        plt.scatter(failure_regions, [-min(scores)] * len(failure_regions), color='red', s=10, label='Actual Failure')

    plt.title(f"Asset 50 Anomaly Detection (Recall: {rec:.2f})")
    plt.legend()
    plt.savefig('anomaly_scores_optimized.png')
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Recall: {rec:.2f})')
    plt.savefig('confusion_matrix_optimized.png')
    print("Plots saved.")

if __name__ == "__main__":
    train_and_evaluate_clustering()