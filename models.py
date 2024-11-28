import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple


class VarianceScheduler:
    def __init__(self, beta_start: float = 0.0001, beta_end: float = 0.02, num_steps: int = 1000,
                 interpolation: str = 'linear') -> None:
        self.num_steps = num_steps

        # Interpolate beta values, result is an evenly spaced set of values from beta_start to beta_end
        if interpolation == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif interpolation == 'quadratic':
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                                        num_steps) ** 2  # Normalize the vector of betas
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')

        # Precompute statistics
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x: torch.Tensor, time_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Adds noise to all image tensors in 'x' given the time step
        :param x: Image tensors to add noise to (time step, color channel, height, width)
            input image tensor at time step zero = x[0]
        :param time_step: Time steps tensors
        :return:
            noisy_input: Noisy version of the input images at the given time steps (time step, color channel, height, width)
            noise: Noise sample from a normal distribution with the same size as the input image 'x'
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Extract alpha_bar[t] for each time step in the batch
        alpha_bars_t = self.alpha_bars.to(device)[time_step].view(-1, 1, 1, 1)

        # Sample random noise
        noise = torch.randn_like(x, device=device)

        # Compute noisy image
        noisy_input = torch.sqrt(alpha_bars_t) * x + torch.sqrt(1 - alpha_bars_t) * noise

        return noisy_input, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal positional embeddings for diffusion timesteps.
        Args:
            time (torch.Tensor): Tensor of timesteps (batch_size,).
        Returns:
            torch.Tensor: Sinusoidal positional embeddings (batch_size, dim).
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        half_dim = self.dim // 2  # split
        # Scaling factors based on the formula
        div_term = torch.exp(-torch.arange(half_dim, device=device) * (2 * torch.log(torch.tensor(10000.0)) / self.dim))
        # Compute sine and cosine embeddings
        embeddings = torch.cat([
            torch.sin(time[:, None] * div_term),
            torch.cos(time[:, None] * div_term)
        ], dim=-1)
        return embeddings


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 down_channels: List = [64, 128, 128, 128, 128],
                 up_channels: List = [128, 128, 128, 128, 64],
                 time_emb_dim: int = 128,
                 num_classes: int = 10) -> None:
        super().__init__()

        # NOTE: You can change the arguments received by the UNet if you want, but keep the num_classes argument

        self.num_classes = num_classes

        # Time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Label embedding layer
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # Downsampling layers
        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else down_channels[i - 1], down_channels[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(down_channels[i]),
                nn.ReLU()
            )
            for i in range(len(down_channels))
        ])

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(down_channels[-1], down_channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(down_channels[-1]),
            nn.ReLU()
        )

        # Upsampling layers
        self.ups = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(down_channels[-1] * 2 if i == 0 else up_channels[i - 1] + down_channels[-(i + 1)],
                                   up_channels[i],
                                   kernel_size=3,
                                   padding=1),
                nn.BatchNorm2d(up_channels[i]),
                nn.ReLU()
            )
            for i in range(len(up_channels))
        ])

        # Final layer
        self.final_conv = nn.Conv2d(up_channels[-1], in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Embed time and labels
        t = self.time_mlp(timestep)  # Time embeddings
        l = self.class_emb(label) if self.num_classes > 0 else 0  # Label embeddings

        # Combine embeddings into a conditioning vector
        conditioning = t + l  # Broadcast addition

        # Pass through downsampling layers
        downs = []
        for down in self.downs:
            x = down(x)
            downs.append(x)

        # Pass through bottleneck layer
        x = self.bottleneck(x)

        # Pass through upsampling layers with skip connections
        for i, up in enumerate(self.ups):
            skip = downs[-(i + 1)]
            x = up(torch.cat([x, skip], dim=1))  # Concatenate along channel dimension

        # Final layer to reconstruct the input space
        out = self.final_conv(x)

        return out


class VAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 height: int = 32,
                 width: int = 32,
                 mid_channels: List = [32, 32, 32],
                 latent_dim: int = 32,
                 num_classes: int = 10) -> None:

        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # NOTE: self.mid_size specifies the size of the image [C, H, W] in the bottleneck of the network
        self.mid_size = [mid_channels[-1], height // (2 ** (len(mid_channels) - 1)),
                         width // (2 ** (len(mid_channels) - 1))]

        # NOTE: You can change the arguments of the VAE as you please, but always define self.latent_dim, self.num_classes, self.mid_size

        # TODO: handle the label embedding here
        self.class_emb = ...

        # TODO: define the encoder part of your network
        self.encoder = ...

        # TODO: define the network/layer for estimating the mean
        self.mean_net = ...

        # TODO: define the networklayer for estimating the log variance
        self.logvar_net = ...

        # TODO: define the decoder part of your network
        self.decoder = ...

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: compute the output of the network encoder
        out = ...

        # TODO: estimating mean and logvar
        mean = self.mean_net(out)
        logvar = self.logvar_net(out)

        # TODO: computing a sample from the latent distribution
        sample = self.reparameterize(mean, logvar)

        # TODO: decoding the sample
        out = self.decode(sample, label)

        return out, mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: implement the reparameterization trick: sample = noise * std + mean
        sample = ...

        return sample

    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')

        return loss

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: compute the KL divergence
        kl_div = -.5 * (logvar.flatten(start_dim=1) + 1 - torch.exp(logvar.flatten(start_dim=1)) - mean.flatten(
            start_dim=1).pow(2)).sum()

        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor = None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            # randomly consider some labels
            labels = torch.randint(0, self.num_classes, [num_samples, ], device=device)

        # TODO: sample from standard Normal distrubution
        noise = ...

        # TODO: decode the noise based on the given labels
        out = ...

        return out

    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: use you decoder to decode a given sample and their corresponding labels
        out = ...

        return out


class LDDPM(nn.Module):
    def __init__(self, network: nn.Module, vae: VAE, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.vae = vae
        self.network = network

        # freeze vae
        self.vae.requires_grad_(False)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: uniformly sample as many timesteps as the batch size
        t = ...

        # TODO: generate the noisy input
        noisy_input, noise = ...

        # TODO: estimate the noise
        estimated_noise = ...

        # compute the loss (either L1 or L2 loss)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor,
                       timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        sample = ...

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'),
                        labels: torch.Tensor = None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.vae.num_classes, [num_samples, ], device=device)

        # TODO: using the diffusion model generate a sample inside the latent space of the vae
        # NOTE: you need to recover the dimensions of the image in the latent space of your VAE
        sample = ...

        sample = self.vae.decode(sample, labels)

        return sample


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Uniformly sample as many timesteps as the batch size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = x.shape[0]
        t = torch.randint(low=0,
                          high=self.var_scheduler.num_steps,
                          size=(batch_size,),
                          device=device)

        # Generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # Estimate the noise using the UNet
        estimated_noise = self.network(x=noisy_input,
                                       timestep=t,
                                       label=label)

        # Compute the L2 loss
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor,
                       timestep: torch.Tensor) -> torch.Tensor:
        '''
        return x_t-1 given x_t and estimated noise
        :param noisy_sample: The current sample in the backwards process
        :param estimated_noise: The noise predicted by the UNet
        :param timestep: current timestep t
        :return: Denoised version of the sample at step t-1 (x_t-1)
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # TODO: implement the sample recovery strategy of the DDPM

        # Get precomputed alphas and alpha bars
        alpha_t = self.var_scheduler.alphas.to(device)[timestep].view(-1, 1, 1, 1)
        alpha_bar_t = self.var_scheduler.alpha_bars.to(device)[timestep].view(-1, 1, 1, 1)

        # Calculate x_t-1
        x_prev = (1 / torch.sqrt(alpha_t)) * (noisy_sample - torch.sqrt(1 - alpha_t) * estimated_noise)
        return x_prev

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'),
                        labels: torch.Tensor = None):
        '''

        :param num_samples:
        :param device:
        :param labels:
        :return:
        '''
        shape = (num_samples, self.network.downs[0][0].in_channels, 32, 32)  # Example image shape (adjust accordingly)
        samples = torch.randn(shape, device=device)  # Start from pure noise

        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples, ], device=device)
        else:
            labels = None

        # TODO: apply the iterative sample generation of the DDPM
        for t in reversed(range(self.var_scheduler.num_steps)):
            timestep = torch.tensor([t] * num_samples, device=device)
            estimated_noise = self.network(samples, timestep, labels)
            if t > 0:
                samples = self.recover_sample(samples, estimated_noise, timestep)
            else:
                # Final denoised sample
                alpha_bar_t = self.var_scheduler.alpha_bars.to(device)[timestep].view(-1, 1, 1, 1)
                samples = (1 / torch.sqrt(alpha_bar_t)) * (samples - torch.sqrt(1 - alpha_bar_t) * estimated_noise)

        return samples


class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        t = ...

        # TODO: generate the noisy input
        noisy_input, noise = ...

        # TODO: estimate the noise
        estimated_noise = ...

        # TODO: compute the loss
        loss = F.l1_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor,
                       timestep: torch.Tensor) -> torch.Tensor:
        # TODO: apply the sample recovery strategy of the DDIM
        sample = ...

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'),
                        labels: torch.Tensor = None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples, ], device=device)
        else:
            labels = None
        # TODO: apply the iterative sample generation of DDIM (similar to DDPM)
        sample = ...

        return sample

