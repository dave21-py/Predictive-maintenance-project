### Anamoly Detection for Wind Turbines 

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

#### Objective
To detect catastrophic failures in turbines without labeled "failure" data.

#### Architecture (Current)
We implemented a **Variational Autoencoder (VAE)** to learn the physical manifold of healthy turbine operations.
*   **Compression:** 2,853 sensor features $\to$ 10 latent variables.
*   **Inference:** Anomalies are detected via high **Reconstruction Error** (MSE), indicating a deviation from learned physics ($P \propto v^3$).
