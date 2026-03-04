import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from src.utils import load_config, setup_logger, get_device
from src.data_loader import DataLoader as RawDataLoader
from src.features import FeatureEngineer
from src.trainer_wjepa import WJEPATrainer


def _fit_scaler_and_selector(
    df: pd.DataFrame,
    power_column: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Fits a StandardScaler + VarianceThreshold selector on the given dataframe
    and returns:
    - X_scaled: np.ndarray of shape (N, D_selected)
    - P: np.ndarray of shape (N,)
    - selected_feature_names: List[str] of column names used for X_scaled
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold

    drop_cols = ["time_stamp", "asset_id", "target", "train_test", "status_type_id"]
    features = [
        c
        for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[features].values
    selector = VarianceThreshold(threshold=0.0)
    X_selected = selector.fit_transform(X)

    support = selector.get_support(indices=True)
    selected_features = [features[i] for i in support]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    if power_column not in df.columns:
        raise ValueError(
            f"Configured power_column '{power_column}' not found in dataframe columns."
        )
    P = df[power_column].values

    # Persist scaler and selector to disk for later reuse (e.g., Colab / inference)
    os.makedirs("artifacts", exist_ok=True)
    import joblib

    joblib.dump(scaler, os.path.join("artifacts", "wjepa_scaler.pkl"))
    joblib.dump(selector, os.path.join("artifacts", "wjepa_selector.pkl"))
    joblib.dump(selected_features, os.path.join("artifacts", "wjepa_features.pkl"))

    return X_scaled, P, selected_features


def _transform_with_existing_scaler(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses previously saved scaler + selector + feature list
    to transform a new dataframe for inference / evaluation.
    """
    import joblib

    selector = joblib.load(os.path.join("artifacts", "wjepa_selector.pkl"))
    scaler = joblib.load(os.path.join("artifacts", "wjepa_scaler.pkl"))
    selected_features: List[str] = joblib.load(
        os.path.join("artifacts", "wjepa_features.pkl")
    )

    X = df[selected_features].values
    X_sel = selector.transform(X)
    X_scaled = scaler.transform(X_sel)
    return X_scaled, np.arange(len(df))


def main():
    config = load_config()
    logger = setup_logger("MainWJEPA", config["paths"]["logs"])
    logger.info("--- W-JEPA World Model Pipeline Started ---")

    device = get_device()
    logger.info(f"Using device: {device}")

    wj_cfg = config["models"]["wjepa"]
    power_column = wj_cfg.get("power_column", "power_2_avg")

    try:
        # --------------------------------------------
        # PART 0: Load failures and verify training assets
        # --------------------------------------------
        loader = RawDataLoader(config)
        failures = loader.load_failures()
        engineer = FeatureEngineer(config)

        train_assets = config["assets"]["train_assets"]
        failed_assets = failures["asset_id"].unique().tolist()

        logger.info(f"Verifying training assets {train_assets} against failure list...")
        for asset in train_assets:
            if asset in failed_assets:
                raise ValueError(
                    f"CRITICAL ERROR: Asset {asset} is listed in event_info.csv! "
                    "You cannot train on it."
                )
        logger.info("✅ Verification Passed: Training assets are clean.")

        # --------------------------------------------
        # PART 1: Build training dataframe (healthy fleet)
        # --------------------------------------------
        logger.info("--- Preparing Training Data for W-JEPA ---")
        train_dfs = []
        for asset_id in train_assets:
            df = loader.load_turbine(asset_id)
            df = engineer.create_rolling_features(df)
            df = df.fillna(0)
            train_dfs.append(df)

        df_train = pd.concat(train_dfs)

        # Fit scaler/selector and extract power column
        X_train_scaled, P_train, selected_features = _fit_scaler_and_selector(
            df_train,
            power_column=power_column,
        )

        # Simple 80/20 split in time order
        N = X_train_scaled.shape[0]
        split_idx = int(0.8 * N)
        X_train, X_val = X_train_scaled[:split_idx], X_train_scaled[split_idx:]
        P_train_split, P_val = P_train[:split_idx], P_train[split_idx:]

        # --------------------------------------------
        # PART 2: Train W-JEPA
        # --------------------------------------------
        logger.info("Initializing W-JEPA trainer...")
        trainer = WJEPATrainer(
            config=config,
            feature_names=selected_features,
            x_train=X_train,
            x_val=X_val,
            p_train=P_train_split,
            p_val=P_val,
        )

        wjepa_model = trainer.train()

        # Save model weights for Colab / long training runs
        os.makedirs("artifacts", exist_ok=True)
        torch.save(
            wjepa_model.state_dict(),
            os.path.join("artifacts", "wjepa_world_model.pt"),
        )
        logger.info("Saved W-JEPA world model to artifacts/wjepa_world_model.pt")

        logger.info("--- W-JEPA Pipeline Finished Successfully ---")

    except Exception as e:
        logger.error(f"W-JEPA Pipeline Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

