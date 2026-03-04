import os
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from src.utils import load_config, setup_logger, get_device
from src.data_loader import DataLoader as RawDataLoader
from src.features import FeatureEngineer
from src.model_wjepa import WJEPAWorldModel
from src.trainer_wjepa import build_subsystem_slices


def load_artifacts(artifacts_dir: str = "artifacts"):
    scaler = joblib.load(os.path.join(artifacts_dir, "wjepa_scaler.pkl"))
    selector = joblib.load(os.path.join(artifacts_dir, "wjepa_selector.pkl"))
    feature_names: List[str] = joblib.load(
        os.path.join(artifacts_dir, "wjepa_features.pkl")
    )
    state_dict = torch.load(os.path.join(artifacts_dir, "wjepa_world_model.pt"))
    return scaler, selector, feature_names, state_dict


def prepare_asset_dataframe(config, asset_id: int) -> pd.DataFrame:
    loader = RawDataLoader(config)
    failures = loader.load_failures()
    engineer = FeatureEngineer(config)

    df = loader.load_turbine(asset_id)
    df = engineer.create_target(df, failures, asset_id)
    df = engineer.create_rolling_features(df)
    df = df.fillna(0)
    return df


def transform_with_artifacts(df: pd.DataFrame, scaler, selector, feature_names: List[str]):
    X = df[feature_names].values
    X_sel = selector.transform(X)
    X_scaled = scaler.transform(X_sel)
    return X_scaled


def compute_surprise_scores(
    model: WJEPAWorldModel,
    X_scaled: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Computes a per-timestep surprise score based on latent prediction error:
    surprise_t+1 = || z_hat_{t+1} - z_{t+1}^{teacher} ||^2
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for t in range(len(X_scaled) - 1):
            x_t = torch.from_numpy(X_scaled[t]).float().unsqueeze(0).to(device)
            x_tp1 = torch.from_numpy(X_scaled[t + 1]).float().unsqueeze(0).to(device)

            z_t = model.encode_student(x_t)
            z_tp1 = model.encode_teacher(x_tp1)
            z_hat_tp1 = model.predict_next(z_t)

            surprise = torch.mean((z_hat_tp1 - z_tp1) ** 2, dim=1)
            scores.append(surprise.item())

    # Align length with df (first timestep has no surprise)
    scores = np.array(scores)
    scores = np.concatenate([[scores[0]], scores])
    return scores


def main():
    config = load_config()
    logger = setup_logger("EvalWJEPA", config["paths"]["logs"])
    logger.info("--- Evaluating W-JEPA on target asset ---")

    device = get_device()
    logger.info(f"Using device: {device}")

    target_asset = config["assets"]["target_asset"]

    try:
        scaler, selector, feature_names, state_dict = load_artifacts()

        df_test = prepare_asset_dataframe(config, target_asset)
        X_scaled = transform_with_artifacts(df_test, scaler, selector, feature_names)

        # Build model with same feature mapping as training
        wj_cfg = config["models"]["wjepa"]
        subsystem_slices = build_subsystem_slices(feature_names)

        model = WJEPAWorldModel(
            input_dim=X_scaled.shape[1],
            latent_dim=wj_cfg.get("latent_dim", 64),
            subsystem_slices=subsystem_slices,
            hidden_dim=wj_cfg.get("encoder_hidden_dim", 256),
            predictor_hidden_dim=wj_cfg.get("predictor_hidden_dim", 256),
        ).to(device)
        model.load_state_dict(state_dict)

        # Surprise scores
        surprise = compute_surprise_scores(model, X_scaled, device)
        df_test["wjepa_surprise"] = surprise

        # AUROC between healthy (target=0) vs anomaly window (target=1)
        if df_test["target"].nunique() >= 2:
            y_true = df_test["target"].values
            y_score = df_test["wjepa_surprise"].values
            auroc = roc_auc_score(y_true, y_score)
            logger.info(f"[W-JEPA] AUROC on Asset {target_asset}: {auroc:.4f}")
        else:
            logger.warning("Target labels for evaluation asset are not diverse; AUROC not computed.")
            auroc = None

        # Threshold for lead-time: mean + 2 * std on healthy region
        healthy = df_test[df_test["target"] == 0]["wjepa_surprise"]
        if not healthy.empty:
            threshold = healthy.mean() + 2 * healthy.std()
        else:
            threshold = df_test["wjepa_surprise"].mean() + 2 * df_test["wjepa_surprise"].std()
        df_test["wjepa_alarm"] = (df_test["wjepa_surprise"] > threshold).astype(int)

        # Lead time: first alarm vs last failure timestamp
        danger_zone = df_test[df_test["target"] == 1]
        if not danger_zone.empty:
            failure_date = danger_zone.index.max()
            alarm_indices = df_test[df_test["wjepa_alarm"] == 1].index
            alarm_indices = alarm_indices[alarm_indices <= failure_date]

            if len(alarm_indices) > 0:
                first_alarm = alarm_indices.min()
                lead_time = failure_date - first_alarm
                logger.info(f"[W-JEPA] Failure Date: {failure_date}")
                logger.info(f"[W-JEPA] First Detection: {first_alarm}")
                logger.info(f"[W-JEPA] Lead Time: {lead_time}")
            else:
                logger.warning("[W-JEPA] No alarms before failure window.")
        else:
            logger.warning("No failure window (target=1) found for target asset.")

        # Plot surprise over time with threshold and failure window
        os.makedirs(config["paths"]["outputs"], exist_ok=True)
        plt.figure(figsize=(15, 6))
        plt.plot(df_test.index, df_test["wjepa_surprise"], label="W-JEPA Surprise", color="blue", alpha=0.7)
        plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.4f})")

        if not danger_zone.empty:
            plt.axvspan(danger_zone.index.min(), danger_zone.index.max(), color="red", alpha=0.2, label="Failure Window")

        plt.title(f"W-JEPA Latent Surprise - Asset {target_asset}")
        plt.xlabel("Time")
        plt.ylabel("Latent Surprise")
        plt.legend()
        out_path = os.path.join(config["paths"]["outputs"], "wjepa_surprise_asset_target.png")
        plt.savefig(out_path)
        logger.info(f"[W-JEPA] Saved surprise plot to {out_path}")

        logger.info("--- W-JEPA Evaluation Finished ---")

    except Exception as e:
        logger.error(f"W-JEPA Evaluation Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

