import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from typing import Tuple
import sys
from log import Logger

class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        
        self.unconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())
        self.unconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
                                     )

        self.expectation   = nn.Linear(in_features=128 * 8 * 8, out_features=latent_dim)
        self.log_variance  = nn.Linear(in_features=128 * 8 * 8, out_features=latent_dim)
        self.decoder_input = nn.Linear(in_features=latent_dim, out_features=128 * 8 * 8)

        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, unflattened_size=(128, 8, 8))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        
        return self.expectation(x), self.log_variance(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.decoder_input(z)
        z = self.unflatten(z)
        z = self.unconv1(z)
        z = self.unconv2(z)

        return torch.sigmoid(z)
        
    def reparametrize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var) # log(std^2) = 1/2 * log(std)
        eps = torch.randn_like(std) # eps ~ N(0, 1)

        return mean + std * eps
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode(x)
        z = self.reparametrize(mean, log_var)

        return self.decode(z), mean, log_var
    

# MSE + Dkl
def loss_func(orig_img: torch.Tensor, sampled_img: torch.Tensor, log_var: torch.Tensor, expectation: torch.Tensor) -> torch.Tensor:
    reconstruction_loss = nn.functional.mse_loss(sampled_img, orig_img, reduction="sum")
    kullback_leibler_loss = -0.5 * torch.sum(1 + log_var - expectation.pow(2) - log_var.exp())

    return (reconstruction_loss + kullback_leibler_loss) / sampled_img.size(0)

def train_vae(n_epoch: int, model: VAE, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    optimizer = Adam(model.parameters(), lr=1e-3)
    best_loss = float('inf')
    patience  = 20
    counter = 0
    
    with open("models/vae_training_log.txt", "w", encoding="utf-8") as f:
        sys.stdout = Logger(sys.stdout, f)

        for epoch in range(n_epoch):

            if counter > patience:
                print(f"Early stopping {epoch + 1}/{n_epoch} with best val loss: {best_loss:.4f}")
                return model
            
            model.train()

            total = 0
            train_loss = 0.0
            for img, _ in train_loader:
                output, mean, log_var = model(img)

                loss = loss_func(img, output, log_var, mean)
                total += img.size(0)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss /= total
            
            model.eval()
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for img, _ in val_loader:
                    output, _, _ = model(img)
                    loss = loss_func(img, output, log_var, mean)

                    val_loss += loss.item()
                    total += img.size(0)

            val_loss /= total

            if best_loss > val_loss:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
            
            print(f"Epoch {epoch + 1}/{n_epoch} | Val loss: {val_loss:.4f} | Train loss: {train_loss:.4f}")
    
    return model