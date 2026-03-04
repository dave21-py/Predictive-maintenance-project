import torch


def vicreg_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    VICReg loss as in Bardes et al.:
    - Invariance: MSE between student prediction and teacher representation.
    - Variance: enforce per-dimension std >= gamma.
    - Covariance: penalize off-diagonal covariance entries.
    """
    # Invariance term: match z_pred and z_target
    invariance = torch.mean((z_pred - z_target) ** 2)

    # Center the embeddings
    z_pred_centered = z_pred - z_pred.mean(dim=0, keepdim=True)
    z_target_centered = z_target - z_target.mean(dim=0, keepdim=True)

    # Variance term (for both views)
    eps = 1e-4
    std_pred = torch.sqrt(z_pred_centered.var(dim=0) + eps)
    std_target = torch.sqrt(z_target_centered.var(dim=0) + eps)

    var_pred = torch.mean(torch.relu(gamma - std_pred))
    var_target = torch.mean(torch.relu(gamma - std_target))
    variance = var_pred + var_target

    # Covariance term (for both views)
    N, D = z_pred_centered.shape
    cov_pred = (z_pred_centered.T @ z_pred_centered) / (N - 1)
    cov_target = (z_target_centered.T @ z_target_centered) / (N - 1)

    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z_pred.device)
    cov_pred_off = cov_pred[off_diag_mask] ** 2
    cov_target_off = cov_target[off_diag_mask] ** 2
    covariance = cov_pred_off.mean() + cov_target_off.mean()

    loss = (
        invariance_weight * invariance
        + variance_weight * variance
        + covariance_weight * covariance
    )
    return loss


def physics_power_loss(
    p_hat: torch.Tensor,
    p_actual: torch.Tensor,
    lambda_power: float = 15.0,
) -> torch.Tensor:
    """
    Physics-informed anchor:
    Enforces that the latent space maintains information needed to predict Active Power P.
    """
    mse = torch.mean((p_hat - p_actual) ** 2)
    return lambda_power * mse

