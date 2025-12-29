from src.utils import load_config, setup_logger
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.model import UnsupervisedModel


def main():
    # Setup
    config = load_config()
    logger = setup_logger("Main", config['paths']['logs'])

    logger.info("Pipeline started")

    try: 
        # Load data
        loader = DataLoader(config)
        failures = loader.load_failures()
        
        # Load Asset 50
        df = loader.load_turbine(config['assets']['target_asset'])
        
        # Test Feature Engineering
        engineer = FeatureEngineer(config)
        
        # Test Target Creation
        df = engineer.create_target(df, failures, asset_id=config['assets']['target_asset'])
        
        # Test Rolling Features
        df = engineer.create_rolling_features(df)
        
        # Model Training
        model = UnsupervisedModel(config)
        model.select_features(df)
        scores = model.train(df)
        best_threshold = model.optimize_threshold(scores, df['target'])
        model.evaluate(scores, df['target'], best_threshold)
        
        logger.info("--- Pipeline Finished Successfully ---")

    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        # Print full error for debugging
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()