import torch
import torch.nn as nn
from torch.autograd import Variable

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
            x = torch.relu(layer(x))
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
            x = torch.relu(layer(x))
        return x

def reparameterize(mu, sigma):
    e = torch.randn_like(sigma)
    return e * torch.exp(0.5*sigma) + mu

class VAE(nn.Module):
    def __init__(self, data_dim, latent_dim, W=512, D=4, kl_lambda =0.1):
        super().__init__()
        self.encoder = VAE_encoder(data_dim, latent_dim, W, D)
        self.decoder = VAE_decoder(data_dim, latent_dim, W, D)
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.W = W
        self.D = D
        self.kl_lambda = kl_lambda
        self.mse_loss = nn.MSELoss()

    def encode(self, x):
        mu, sigma = self.encoder(x)
        output = reparameterize(mu, sigma)
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
        loss = kl_loss*self.kl_lambda + recon_loss
        return loss, recon_loss, kl_loss


class Learnable_Latents(nn.Module):
    def __init__(self, style_num, frame_num, latent_dim):
        super().__init__()
        self.style_num = style_num
        self.frame_num = frame_num
        self.latent_dim = latent_dim

        self.latents = torch.randn(self.style_num, self.frame_num, self.latent_dim)
        self.latents_mu = torch.randn(self.style_num, self.latent_dim)
        self.latents_sigma = torch.randn(self.style_num, self.latent_dim)

        self.sigma_scale = 1.0
        self.set_requires_grad()
        self.latents_optim = None

    def set_requires_grad(self):
        self.latents.requires_grad = True
        self.latents_mu.requires_grad = False
        self.latents_sigma.requires_grad=False

    def forward(self, style_ids, frame_ids):
        flat_ids= style_ids * self.frame_num + frame_ids
        frame_latents = self.latents.reshape(-1, self.latent_dim)[flat_ids]
        latents_mu = self.latents_mu[style_ids]
        return (frame_latents-latents_mu) * self.sigma_scale + latents_mu
    
    def loss(self, style_ids, frame_ids):
        latents = self(style_ids, frame_ids)
        mu = self.latents_mu[style_ids]
        sigma = self.latents_sigma[style_ids]
        eps = 1e-3
        loss = torch.mean(torch.sum((latents-mu.detach())**2 / (torch.exp(0.5 * sigma.detach()) + eps), -1))
        return loss

    def set_latents(self, latents_mu, latents_sigma):
        self.latents_mu, self.latents_sigma = latents_mu, latents_sigma
        all_latents_mu = latents_mu.unsqueeze(1).expand(list(self.latents.shape))
        all_latents_sigma = latents_sigma.unsqueeze(1).expand(list(self.latents.shape))
        self.latents = reparameterize(all_latents_mu, all_latents_sigma)
        self.set_requires_grad()

    def rescale_sigma(self, sigma_scale):
        self.sigma_scale = sigma_scale

    def set_latents_optim(self):
        self.latents_optim = torch.optim.Adam([self.latents], lr=1e-3)

    def optimize(self):
        self.latents_optim.step()
        self.latents_optim.zero_grad()

