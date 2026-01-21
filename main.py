import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from src.utils import load_config, setup_logger
from src.data_loader import DataLoader as RawDataLoader
from src.features import FeatureEngineer
from src.model_vae import VAE
from src.trainer_vae import VAETrainer
from src.utils_deep import DeepLearningDataUtils

def main():
    config = load_config()
    logger = setup_logger("Main", config['paths']['logs'])
    logger.info("--- Deep Learning Pipeline Started (VAE) ---")

    try:
        # 1. Load Data
        loader = RawDataLoader(config)
        failures = loader.load_failures()
        engineer = FeatureEngineer(config)
        
        # ==========================================
        # PART 0: THE SANITY CHECK (Research Grade)
        # ==========================================
        train_assets = config['assets']['train_assets']
        failed_assets = failures['asset_id'].unique().tolist()
        
        logger.info(f"Verifying training assets {train_assets} against failure list...")
        for asset in train_assets:
            if asset in failed_assets:
                raise ValueError(f"CRITICAL ERROR: Asset {asset} is listed in event_info.csv! You cannot train on it.")
        logger.info("✅ Verification Passed: Training assets are clean.")

        # ==========================================
        # PART 1: TRAIN VAE ON HEALTHY FLEET
        # ==========================================
        logger.info("--- Preparing Training Data ---")
        train_dfs = []
        for asset_id in train_assets:
            df = loader.load_turbine(asset_id)
            df = engineer.create_rolling_features(df)
            train_dfs.append(df)
        
        df_train = pd.concat(train_dfs)
        df_train = df_train.fillna(0)
        
        # Convert to Tensors (Train/Val Split)
        dl_utils = DeepLearningDataUtils(config)
        train_loader, val_loader, feature_cols = dl_utils.prepare_train_val_loaders(df_train)
        
        input_dim = len(feature_cols)
        logger.info(f"Input Dimension: {input_dim} features")

        # Initialize & Train
        logger.info("Initializing Variational Autoencoder...")
        model = VAE(input_dim=input_dim, hidden_dim=64, latent_dim=10)
        
        trainer = VAETrainer(model, config)
        trainer.train(train_loader, val_loader)

        # ==========================================
        # PART 2: TEST ON TARGET ASSET (50)
        # ==========================================
        logger.info("--- Preparing Test Data (Asset 50) ---")
        df_test = loader.load_turbine(config['assets']['target_asset'])
        df_test = engineer.create_target(df_test, failures, config['assets']['target_asset'])
        df_test = engineer.create_rolling_features(df_test)
        df_test = df_test.fillna(0)
        
        # Prepare Test Loader (using the same scaler as training)
        test_loader = dl_utils.prepare_test_loader(df_test, feature_cols)

        # ==========================================
        # PART 3: INFERENCE (Reconstruction Error)
        # ==========================================
        from src.utils import get_device # Import here if needed
        device = get_device()
        
        logger.info(f"Running Inference on {device}...")
        model.eval() 
        reconstruction_errors = []
        
        with torch.no_grad():
            for data, _ in test_loader:
                # Move data to GPU
                data = data.to(device)
                
                recon, mu, log_var = model(data)
                
                # Calculate loss
                loss = torch.mean((data - recon) ** 2, dim=1)
                
                # Move result BACK to CPU for numpy/plotting
                reconstruction_errors.extend(loss.cpu().numpy())
        
        # ==========================================
        # PART 4: EVALUATION & RESULTS
        # ==========================================
        df_test['anomaly_score'] = reconstruction_errors
        
        # Dynamic Threshold: Mean + 2 Std Devs (Statistical Standard)
        threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
        df_test['pred'] = (df_test['anomaly_score'] > threshold).astype(int)
        
        # Check Lead Time
        danger_zone = df_test[df_test['target'] == 1]
        true_alarms = danger_zone[danger_zone['pred'] == 1]
        
        if not true_alarms.empty:
            first_alarm = true_alarms.index.min()
            failure_date = danger_zone.index.max()
            lead_time = failure_date - first_alarm
            
            logger.info("=== DEEP LEARNING RESULTS ===")
            logger.info(f"Failure Date:    {failure_date}")
            logger.info(f"First Detection: {first_alarm}")
            logger.info(f"LEAD TIME:     {lead_time}")
            logger.info("=============================")
        else:
            logger.warning("No alarms triggered in the failure window.")

        # Plotting
        os.makedirs(config['paths']['outputs'], exist_ok=True)
        plt.figure(figsize=(15, 6))
        plt.plot(df_test.index, df_test['anomaly_score'], label='Reconstruction Error (VAE)', color='blue', alpha=0.6)
        plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
        
        if not danger_zone.empty:
             plt.axvspan(danger_zone.index.min(), danger_zone.index.max(), color='red', alpha=0.2, label='Failure Window')

        plt.title(f"Deep Learning Anomaly Detection (VAE) - Asset {config['assets']['target_asset']}")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Reconstruction Error (MSE)")
        plt.savefig(os.path.join(config['paths']['outputs'], "vae_results.png"))
        logger.info("Results saved to outputs/vae_results.png")

        logger.info("--- Pipeline Finished Successfully ---")

    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()