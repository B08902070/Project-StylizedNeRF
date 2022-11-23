import torch
import torch.nn as nn
import numpy as np

class VAE_encoder(nn.Module):
    def __init__(self, data_dim, latent_dim, W=512, D=4):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.W = W     # intermidiate layer dimension
        self.D = D     # number of intermidiate layers

        # set fully connected layers
        self.fc_layers = []
        cur_dim = data_dim
        for i in range(D-1):
            self.fc_layers.append(nn.Linear(cur_dim, W))
            cur_dim = W
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.mu_layer = nn.Linear(cur_dim, latent_dim)
        self.sigma_layer = nn.Linear(cur_dim, latent_dim)

    def forward(self, x):
        for layer in self.fc_layers:
            x = nn.ReLU(layer(x))
        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x)

        return mu, sigma

class VAE_decoder(nn.Module):
    def __init__(self, data_dim, latent_dim, W=512, D=4):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.W = W     # intermidiate layer dimension
        self.D = D     # number of intermidiate layers

        # set fully connected layers
        self.fc_layers = []
        cur_dim = latent_dim
        for i in range(D-1):
            self.fc_layers.append(nn.Linear(cur_dim, W))
            cur_dim = W
        self.fc_layers.append(nn.Linear(cur_dim, data_dim))
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, x):
        for layer in self.fc_layers:
            x = nn.ReLU(layer(x))
        return x


class VAE(nn.Module):
    def __init__(self, data_dim, latent_dim, W=512, D=4, kl_lamda =0.1):
        super().__init__()
        self.encoder = VAE_encoder(data_dim, latent_dim, W, D)
        self.decoder = VAE_decoder(data_dim, latent_dim, W, D)
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.W = W
        self.D = D
        self.kl_lamda = kl_lamda
        self.mse_loss = nn.MSELoss()

        # set fully connected layers
        self.fc_layers = []
        cur_dim = latent_dim
        for _ in range(D-1):
            self.fc_layers.append(nn.Linear(cur_dim, W))
            cur_dim = W
        self.fc_layers.append(nn.Linear(cur_dim, data_dim))
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def reparameterize(self, mu, sigma):
        e = torch.randn_like(sigma)
        return e * torch.exp(0.5*sigma) + mu

    def encode(self, x):
        mu, sigma = self.encoder(x)
        output = self.reparameterize(mu, sigma)
        return output, mu, sigma

    def decode(self, x):
        output = self.decoder(x)
        return output

    def forward(self, x):
        encoded_x, mu, sigma = self.encode(x)
        decoded_x = self.decode(encoded_x)
        return decoded_x, mu, sigma
    
    def loss(self, x, decoded_x, mu, sigma):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim=1), dim=0)
        recon_loss = self.mse_loss(x, decoded_x)
        return kl_loss*self.kl_lamda + recon_loss

