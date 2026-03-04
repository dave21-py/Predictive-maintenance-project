import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from src.utils import get_device
from src.model_wjepa import WJEPAWorldModel
from src.losses_wjepa import vicreg_loss, physics_power_loss


class TimePairDataset(Dataset):
    """
    Dataset of consecutive time pairs (x_t, x_{t+1}) with corresponding power labels P_t.
    """

    def __init__(
        self,
        x_array: np.ndarray,
        power_array: np.ndarray,
    ):
        assert len(x_array) == len(power_array)
        # Build pairs (t, t+1)
        self.x_t = x_array[:-1]
        self.x_tp1 = x_array[1:]
        self.p_t = power_array[:-1]

    def __len__(self) -> int:
        return len(self.x_t)

    def __getitem__(self, idx: int):
        x_t = torch.from_numpy(self.x_t[idx]).float()
        x_tp1 = torch.from_numpy(self.x_tp1[idx]).float()
        p_t = torch.tensor(self.p_t[idx]).float()
        return x_t, x_tp1, p_t


def build_subsystem_slices(
    feature_names: List[str],
) -> Dict[str, List[int]]:
    """
    Heuristic mapping of raw feature names into physical subsystems.
    This can be refined later or moved into config if needed.
    """
    subsystems = {
        "thermal": [],
        "electrical": [],
        "dynamics": [],
        "control": [],
        "other": [],
    }

    for idx, name in enumerate(feature_names):
        n = name.lower()
        if "temp" in n or "heat" in n:
            subsystems["thermal"].append(idx)
        elif "power" in n or "voltage" in n or "current" in n:
            subsystems["electrical"].append(idx)
        elif "vib" in n or "speed" in n or "rpm" in n or "torque" in n:
            subsystems["dynamics"].append(idx)
        elif "pitch" in n or "yaw" in n or "control" in n:
            subsystems["control"].append(idx)
        else:
            subsystems["other"].append(idx)

    # Drop empty subsystems except "other"
    return {k: v for k, v in subsystems.items() if v}


class WJEPATrainer:
    def __init__(
        self,
        config: dict,
        feature_names: List[str],
        x_train: np.ndarray,
        x_val: np.ndarray,
        p_train: np.ndarray,
        p_val: np.ndarray,
    ):
        self.config = config
        self.logger = logging.getLogger("WJEPATrainer")
        self.device = get_device()
        self.logger.info(f"🚀 Training W-JEPA on device: {self.device}")

        wj_cfg = config["models"]["wjepa"]

        subsystem_slices = build_subsystem_slices(feature_names)
        self.logger.info(
            f"Subsystem mapping: "
            + ", ".join(f"{k}={len(v)}" for k, v in subsystem_slices.items())
        )

        input_dim = x_train.shape[1]
        latent_dim = wj_cfg.get("latent_dim", 64)

        self.model = WJEPAWorldModel(
            input_dim=input_dim,
            latent_dim=latent_dim,
            subsystem_slices=subsystem_slices,
            hidden_dim=wj_cfg.get("encoder_hidden_dim", 256),
            predictor_hidden_dim=wj_cfg.get("predictor_hidden_dim", 256),
        ).to(self.device)

        self.optimizer = Adam(
            self.model.student_encoder.parameters(),
            lr=wj_cfg.get("learning_rate", 1e-4),
        )

        batch_size = wj_cfg.get("batch_size", 256)
        self.epochs = wj_cfg.get("epochs", 50)
        self.patience = wj_cfg.get("patience", 10)
        self.ema_momentum = wj_cfg.get("ema_momentum", 0.99)

        # Loss weights
        self.inv_w = wj_cfg.get("invariance_weight", 25.0)
        self.var_w = wj_cfg.get("variance_weight", 25.0)
        self.cov_w = wj_cfg.get("covariance_weight", 1.0)
        self.gamma = wj_cfg.get("gamma", 1.0)
        self.lambda_power = wj_cfg.get("lambda_power", 15.0)

        # Build datasets and loaders
        train_dataset = TimePairDataset(x_train, p_train)
        val_dataset = TimePairDataset(x_val, p_val)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

    def _step_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t, x_tp1, p_t = batch
        x_t = x_t.to(self.device)
        x_tp1 = x_tp1.to(self.device)
        p_t = p_t.to(self.device)

        # Encode states
        z_t = self.model.encode_student(x_t)
        with torch.no_grad():
            z_tp1_teacher = self.model.encode_teacher(x_tp1)

        # Predict next latent
        z_hat_tp1 = self.model.predict_next(z_t)

        # Physics head on current state
        p_hat = self.model.predict_power(z_t)

        # Losses
        L_vicreg = vicreg_loss(
            z_hat_tp1,
            z_tp1_teacher,
            invariance_weight=self.inv_w,
            variance_weight=self.var_w,
            covariance_weight=self.cov_w,
            gamma=self.gamma,
        )

        L_phys = physics_power_loss(
            p_hat,
            p_t,
            lambda_power=self.lambda_power,
        )

        L_total = L_vicreg + L_phys
        return L_total, L_vicreg.detach(), L_phys.detach()

    def train(self):
        self.logger.info(f"Starting W-JEPA training for up to {self.epochs} epochs")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # ---------------------
            # Train
            # ---------------------
            self.model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                loss, _, _ = self._step_batch(batch)
                loss.backward()
                self.optimizer.step()

                # EMA update for teacher
                with torch.no_grad():
                    self.model.update_teacher(momentum=self.ema_momentum)

                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)

            # ---------------------
            # Validation
            # ---------------------
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    loss, _, _ = self._step_batch(batch)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_loader)

            self.logger.info(
                f"[W-JEPA] Epoch {epoch}: "
                f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}"
            )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {
                    "student_encoder": self.model.student_encoder.state_dict(),
                    "teacher_encoder": self.model.teacher_encoder.state_dict(),
                    "predictor": self.model.predictor.state_dict(),
                    "physics_head": self.model.physics_head.state_dict(),
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(
                        f"[W-JEPA] Early stopping triggered at epoch {epoch}"
                    )
                    break

        # Restore best weights
        if best_state:
            self.model.student_encoder.load_state_dict(best_state["student_encoder"])
            self.model.teacher_encoder.load_state_dict(best_state["teacher_encoder"])
            self.model.predictor.load_state_dict(best_state["predictor"])
            self.model.physics_head.load_state_dict(best_state["physics_head"])
            self.logger.info("[W-JEPA] Restored best model weights.")

        return self.model

