### Anomaly Detection for Wind Turbines (VAE)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

#### Objective
To detect catastrophic failures in turbines without labeled "failure" data.

#### Architecture
We implemented a **Variational Autoencoder (VAE)** to learn the physical manifold of healthy turbine operations.
*   **Compression:** 2,853 sensor features $\to$ 10 latent variables.
*   **Inference:** Anomalies are detected via high **Reconstruction Error** (MSE), indicating a deviation from learned physics ($P \propto v^3$).

#### Results
*   **Lead Time:** Identified a clear precursor anomaly in September 2023, providing a 60-day warning before the total system crash in October.
*   **Signal:** Reconstruction error spiked to **60.0 MSE** (vs 1.7 threshold) during pre-failure chatter.

PLS SEE PDF FOR FULL DETAILS
![PDF](Poster.pdf)
