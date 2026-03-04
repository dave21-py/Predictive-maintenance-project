### Anamoly Detection for Wind Turbines 

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

#### Objective
To detect catastrophic failures in turbines without labeled "failure" data.

#### Architectures

- **Variational Autoencoder (VAE)** baseline  
  Learns the physical manifold of healthy turbine operations.
  * **Compression:** 2,853+ sensor features → 10 latent variables.
  * **Inference:** Anomalies detected via high **Reconstruction Error** (MSE), indicating a deviation from learned physics (P ∝ v³).

- **W-JEPA (Physics-Informed Joint-Embedding Predictive Architecture)**  
  Self-supervised world model trained with VICReg and a physics-informed power prediction head.
  * **Modality Mapping:** Groups features into physical subsystems (thermal, electrical, dynamics, control).
  * **VICReg:** Variance–Invariance–Covariance Regularization on latent predictions ẑₜ₊₁ vs zₜ₊₁.
  * **Physics Anchor:** Secondary loss on Active Power output (P) to keep the latent space tied to turbine energy-conversion dynamics.
