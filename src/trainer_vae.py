import torch
import torch.nn.functional as F
from torch.optim import Adam
import logging
import numpy as np
import copy
from src.utils import get_device

class VAETrainer:
    def __init__(self, model, config):
        self.config = config
        self.logger = logging.getLogger("VAETrainer")
        self.device = get_device()
        self.logger.info(f"🚀 Training on Device: {self.device}")
        
        self.model = model.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=config['models']['vae']['learning_rate'])
        self.patience = config['models']['vae']['patience']

    def loss_function(self, recon_x, x, mu, log_var):
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def train(self, train_loader, val_loader):
        epochs = self.config['models']['vae']['epochs']
        self.logger.info(f"Starting training for max {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            # ---  LOGGING START ---
            if epoch == 0:
                self.logger.info("Epoch 0 started. Waiting for data loader to feed the GPU...")

            for batch_idx, (data, _) in enumerate(train_loader):
                # Check if this is the very first batch
                if epoch == 0 and batch_idx == 0:
                    self.logger.info("FIRST BATCH RECEIVED! GPU IS CRUNCHING NOW.")

                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, log_var = self.model(data)
                loss = self.loss_function(recon_batch, data, mu, log_var)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                # Print progress every 10 batches for the first epoch only
                # This proves it is not frozen
                if epoch == 0 and batch_idx % 10 == 0:
                     self.logger.info(f"Batch {batch_idx} processed...")
            
            # --- VERBOSE LOGGING END ---

            avg_train_loss = train_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(self.device)
                    recon_batch, mu, log_var = self.model(data)
                    loss = self.loss_function(recon_batch, data, mu, log_var)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader.dataset)

            # Log EVERY epoch so we see speed
            self.logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"Early Stopping triggered at Epoch {epoch}!")
                    break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.logger.info("Restored best model weights.")