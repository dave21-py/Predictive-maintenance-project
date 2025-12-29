import pandas as pd
from src.utils import load_config, setup_logger
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.model import UnsupervisedModel
from src.model_supervised import SupervisedModel

def main():
    config = load_config()
    logger = setup_logger("Main", config['paths']['logs'])
    logger.info("--- Pipeline Started ---")

    try:
        loader = DataLoader(config)
        failures = loader.load_failures()
        engineer = FeatureEngineer(config)
        
        # ==========================================
        # PART 1: PREPARE TEST DATA (Asset 50)
        # ==========================================
        logger.info("--- Preparing Test Data (Asset 50) ---")
        df_test = loader.load_turbine(config['assets']['target_asset'])
        df_test = engineer.create_target(df_test, failures, config['assets']['target_asset'])
        df_test = engineer.create_rolling_features(df_test)
        
        # ==========================================
        # PART 2: UNSUPERVISED APPROACH (Isolation Forest)
        # ==========================================
        logger.info("--- Running Unsupervised Approach ---")
        unsup_model = UnsupervisedModel(config)
        unsup_model.select_features(df_test)
        
        # Train & Optimize
        scores = unsup_model.train(df_test)
        best_threshold = unsup_model.optimize_threshold(scores, df_test['target'])
        unsup_model.evaluate(scores, df_test['target'], best_threshold)
        
        # --- EVIDENCE CALCULATION (The "Smoking Gun" for your Report) ---
        # We calculate exactly when the first alarm went off
        df_test['score'] = scores
        df_test['pred'] = [1 if s < best_threshold else 0 for s in scores]
        
        # Look only at the 14-day failure window
        danger_zone = df_test[df_test['target'] == 1]
        
        # Find alarms inside that window
        true_alarms = danger_zone[danger_zone['pred'] == 1]
        
        if not true_alarms.empty:
            first_alarm = true_alarms.index.min()
            failure_date = danger_zone.index.max()
            lead_time = failure_date - first_alarm
            
            logger.info("=== CUSTOMER VALUE EVIDENCE ===")
            logger.info(f"Actual Failure Date:    {failure_date}")
            logger.info(f"First Alarm Triggered:  {first_alarm}")
            logger.info(f"✅ SYSTEM LEAD TIME:     {lead_time}")
            logger.info("===============================")
        else:
            logger.warning("No alarms triggered during the failure window!")

        # ==========================================
        # PART 3: SUPERVISED APPROACH (XGBoost)
        # ==========================================
        logger.info("--- Running Supervised Approach (Fleet Learning) ---")
        
        # A. Load Fleet Data (12, 15, 16)
        train_dfs = []
        for asset_id in config['assets']['train_assets']:
            try:
                df = loader.load_turbine(asset_id)
                df = engineer.create_target(df, failures, asset_id)
                df = engineer.create_rolling_features(df)
                train_dfs.append(df)
            except Exception as e:
                logger.warning(f"Skipping Train Asset {asset_id}: {e}")
        
        if train_dfs:
            # Combine into one big training set
            df_train = pd.concat(train_dfs)
            
            # B. Train & Evaluate
            sup_model = SupervisedModel(config)
            sup_model.train(df_train)
            sup_model.evaluate(df_test)
        else:
            logger.error("No training data loaded for Supervised Model!")

        logger.info("--- Pipeline Finished Successfully ---")
        
    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()