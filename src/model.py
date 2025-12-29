import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

class UnsupervisedModel:
    def __init__(self, config):
        self.config = config
        self.model_config = config['models']['unsupervised']
        self.logger = logging.getLogger("UnsupervisedModel")
        self.model = None
        self.scaler = StandardScaler()
        self.selected_features = []

    def select_features(self, df):
        """
        Feature Selection.
        Instead of using all 4000+ features, we find the ones that 
        actually correlate with the failure target.
        """
        self.logger.info("Starting feature selection (Correlation Analysis)")

        numeric_df = df.select_dtypes(include=[np.number])


        # Calculate correlation with target
        # (We use the labels here purely for 'Analysis', not for 'Training' the isolation forest)
        correlations = numeric_df.corrwith(numeric_df['target']).abs()

        # 3. Drop the target itself
        if 'target' in correlations:
            correlations = correlations.drop('target')
        
        correlations = correlations.fillna(0)
        
        # 4. Pick Top 20
        top_features = correlations.sort_values(ascending=False).head(20)
        self.selected_features = top_features.index.tolist()
        
        self.logger.info(f"Top 3 Features: {self.selected_features[:3]}")
        self.logger.info(f"Selected {len(self.selected_features)} features.")
        
        return self.selected_features

    def train(self, df):
        """
        Trains the Isolation Forest on the selected features.
        """
        self.logger.info("Preparing data for training...")
        
        # Filter to selected features
        X = df[self.selected_features]
        
        # Handle NaNs (Critical for Production)
        X = X.ffill().bfill().fillna(0)
        
        # Scale Data
        self.logger.info("Scaling data...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        n_trees = self.model_config['n_estimators']
        self.logger.info(f"Training Isolation Forest ({n_trees} trees)...")
        self.model = IsolationForest(
            n_estimators=n_trees,
            contamination=self.model_config['contamination'],
            random_state=self.model_config['random_state'],
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        
        # Return scores for optimization
        scores = self.model.decision_function(X_scaled)
        return scores

    def optimize_threshold(self, scores, y_true):
        """
        Threshold Optimization.
        """
        self.logger.info("Optimizing Threshold for High Recall...")
        
        target_recall = 0.80
        best_threshold = 0
        final_percentile = 0
        
        # Search from top 15% to 50% anomalies
        for p in range(15, 51, 1):
            thresh = np.percentile(scores, p)
            # Scores < thresh are anomalies (1)
            y_pred = [1 if s < thresh else 0 for s in scores]
            
            rec = recall_score(y_true, y_pred)
            
            if rec >= target_recall:
                best_threshold = thresh
                final_percentile = p
                self.logger.info(f"Optimization Success! Percentile: {p}% -> Recall: {rec:.2f}")
                break
        
        # Fallback
        if best_threshold == 0:
            self.logger.warning("Could not reach 0.80 Recall. Defaulting to 50%.")
            best_threshold = np.percentile(scores, 50)
            
        return best_threshold

    def evaluate(self, scores, y_true, threshold):
        """
        Generates final report and plots.
        """
        self.logger.info("Evaluating Model Performance...")
        
        y_pred = [1 if s < threshold else 0 for s in scores]
        
        # Text Report
        report = classification_report(y_true, y_pred)
        print("\n=== MODEL PERFORMANCE REPORT ===")
        print(report)
        print("================================")
        
        # Save Plots
        self._plot_results(y_true, y_pred, scores, threshold)

    def _plot_results(self, y_true, y_pred, scores, threshold):
        """Helper to draw the confusion matrix and anomaly score plot."""
        save_path = self.config['paths']['outputs']
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.close()
        
        # 2. Anomaly Scores
        plt.figure(figsize=(15, 6))
        # Invert scores so "Higher" = "More Anomalous"
        plt.plot(-scores, label='Anomaly Score', color='purple', alpha=0.6)
        plt.axhline(-threshold, color='orange', linestyle='--', label='Threshold')
        plt.title("Anomaly Scores vs Threshold")
        plt.legend()
        plt.savefig(os.path.join(save_path, 'anomaly_scores.png'))
        plt.close()
        
        self.logger.info(f"Plots saved to {save_path}")


