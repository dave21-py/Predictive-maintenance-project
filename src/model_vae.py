from turtle import hideturtle
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=10):
        super(VAE, self).__init__()

        # Compression (Encoder)
        # We shrink data: Input (300) -> Hidden (64) -> Mean(10) & LogVar(10)
        self.encoder_layer = nn.Linear(input_dim, hidden_dim)

        # Two seperate heads: One for Aim (Mean) & One for Uncertainity (Stddev)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim) # If we try to predict variance or stddev instead of log var, the output maybe negative, so by predicting log(var), the output can be negative, -5, e^-5 is positive number


        # Decoder
        # We expand data: Latent (10) -> Hidden (64) -> Output (300)
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def reparametrize(self, mu, log_var):
        """
        z = mu + sigma * epsilon
        """
        if self.training:
            # Calculate sigma from log variance, {Variance} = sigma^2, ln(sigma^2) = 2 * ln(sigma), e^{0.5 * {log_var}} = sigma
            std = torch.exp(0.5 * log_var)

            # generate random noise (epilson, i.e; wind)
            eps = torch.randn_like(std)

            # formula
            return mu + (eps * std)
        else:
            # for testing, return best possible guess, mean
            return mu

    def forward(self, x):
        """
        forward pass
        """

        # -- Encode --
        h = F.relu(self.encoder_layer(x))

        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        # -- Reparametrize
        z = self.reparametrize(mu, log_var)

        # -- Decode --
        h_decoded = F.relu(self.decoder_hidden(z))
        reconstruction = self.decoder_output(h_decoded)

        return reconstruction, mu, log_var






        


