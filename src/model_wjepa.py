import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class WJEPAEncoder(nn.Module):
    """
    Physics-informed encoder for W-JEPA.

    - Groups features into physical subsystems (thermal, electrical, dynamics, control, ...).
    - Each subsystem gets its own linear projection.
    - All subsystem latents are concatenated to form the global latent state z.

    The exact feature indices per subsystem are provided at runtime.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        subsystem_slices: Dict[str, List[int]],
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.subsystem_slices = subsystem_slices
        self.subsystems = list(subsystem_slices.keys())

        # One small MLP per subsystem: R^{|S|} -> R^{latent_dim_per_sub}
        self.sub_encoders = nn.ModuleDict()
        for name, indices in subsystem_slices.items():
            in_dim = len(indices)
            if in_dim == 0:
                # Skip empty subsystems
                continue
            self.sub_encoders[name] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, latent_dim),
            )

        # No extra projection here; the concatenation of all subsystem latents is z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_dim)
        returns z: (batch_size, latent_dim * num_active_subsystems)
        """
        latents = []
        for name, indices in self.subsystem_slices.items():
            if name not in self.sub_encoders:
                continue
            if not indices:
                continue
            # Select subsystem features
            sub_x = x[:, indices]
            latents.append(self.sub_encoders[name](sub_x))

        if not latents:
            raise ValueError("No active subsystems found for WJEPAEncoder.")

        # Concatenate along feature dimension
        z = torch.cat(latents, dim=1)
        return z


class WJEPAWorldModel(nn.Module):
    """
    Core W-JEPA world model:

    - Student encoder E_s: encodes context x_t -> z_t.
    - Teacher encoder E_t: encodes target x_{t+1} -> z_{t+1} (updated via EMA in trainer).
    - Predictor: f(z_t) -> z_hat_{t+1}.
    - Physics head: maps z_t to Active Power P_hat.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        subsystem_slices: Dict[str, List[int]],
        hidden_dim: int = 256,
        predictor_hidden_dim: int = 256,
    ):
        super().__init__()

        self.student_encoder = WJEPAEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            subsystem_slices=subsystem_slices,
            hidden_dim=hidden_dim,
        )

        # Teacher has the same architecture; parameters will be kept in sync via EMA
        self.teacher_encoder = WJEPAEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            subsystem_slices=subsystem_slices,
            hidden_dim=hidden_dim,
        )

        # Freeze teacher by default; trainer will update via EMA
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        # Student predictor: z_t -> z_hat_{t+1}
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, predictor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(predictor_hidden_dim, self.latent_dim),
        )

        # Physics head: z_t -> P_hat (scalar power prediction)
        self.physics_head = nn.Sequential(
            nn.Linear(self.latent_dim, predictor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(predictor_hidden_dim, 1),
        )

    @property
    def latent_dim(self) -> int:
        # Effective latent dimension = per-subsystem latent_dim * number of active subsystems
        example_sub = next(iter(self.student_encoder.sub_encoders.values()))
        per_sub_dim = example_sub[-1].out_features
        num_subs = len(self.student_encoder.sub_encoders)
        return per_sub_dim * num_subs

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.99):
        """
        Exponential moving average (EMA) update for teacher parameters.
        """
        for student_param, teacher_param in zip(
            self.student_encoder.parameters(), self.teacher_encoder.parameters()
        ):
            teacher_param.data = (
                momentum * teacher_param.data + (1.0 - momentum) * student_param.data
            )

    def encode_student(self, x_t: torch.Tensor) -> torch.Tensor:
        return self.student_encoder(x_t)

    @torch.no_grad()
    def encode_teacher(self, x_tp1: torch.Tensor) -> torch.Tensor:
        return self.teacher_encoder(x_tp1)

    def predict_next(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Predict z_{t+1} from z_t using the student predictor.
        """
        return self.predictor(z_t)

    def predict_power(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Predict Active Power output P_hat from latent state z_t.
        """
        return self.physics_head(z_t).squeeze(-1)

