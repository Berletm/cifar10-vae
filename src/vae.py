import torch.nn as nn
import torch

from typing import Tuple

# MSE + Dkl
def loss_func(orig_img: torch.Tensor, sampled_img: torch.Tensor, log_var: torch.Tensor, expectation: torch.Tensor) -> torch.Tensor:
    pass

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
        
        self.unconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())
        self.unconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU())

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
        eps = torch.rand_like(std) # eps ~ N(0, 1)

        return mean + std * eps
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode(x)
        z = self.reparametrize(mean, log_var)

        return self.decode(z), mean, log_var
    

