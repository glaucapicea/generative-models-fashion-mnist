import torch
from torch import nn

import torch.nn.functional as F
from typing import List, Tuple


# from labml_nn.diffusion.ddpm.utils import gather

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
        self.sigma2 = self.betas

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


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 n_downs=2,
                 time_emb_dim: int = 128,
                 num_classes: int = 10) -> None:
        super().__init__()

        # NOTE: You can change the arguments received by the UNet if you want, but keep the num_classes argument
        self.num_classes = num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.time_emb_dim = time_emb_dim

        # Label embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.class_emb = nn.Embedding(num_classes, self.time_emb_dim)

        # Initial input resizing layer
        self.inc = DoubleConv(in_channels, 32)

        # Downsampling layers
        self.down1 = Down(32, 64)
        self.sa1 = SelfAttention(64, 16)
        self.down2 = Down(64, 128)
        self.sa2 = SelfAttention(128, 8)
        self.down3 = Down(128, 128)
        self.sa3 = SelfAttention(128, 4)

        # Bottleneck layers
        self.bot1 = DoubleConv(128, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 128)

        # Upsampling layers
        self.up1 = Up(256, 64)
        self.sa4 = SelfAttention(64, 8)
        self.up2 = Up(128, 32)
        self.sa5 = SelfAttention(32, 16)
        self.up3 = Up(64, 32)
        self.sa6 = SelfAttention(32, 32)
        self.outc = nn.Conv2d(32, in_channels, kernel_size=1)

    def pos_encoding(self, time):
        dim = self.time_emb_dim
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, dim, 2, device=self.device).float() / dim)
        )
        sine_half = torch.sin(time.repeat(1, dim // 2) * inv_freq)
        cos_half = torch.cos(time.repeat(1, dim // 2) * inv_freq)
        pos_enc = torch.cat([sine_half, cos_half], dim=-1)
        return pos_enc

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(timestep)
        # t = SinusoidalPositionEmbeddings(timestep)
        # t = t.unsqueeze(-1).type(torch.float)
        # t = self.pos_encoding(t)
        if label is not None:
            t += self.class_emb(label)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        output = self.outc(x)

        return output


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
        # Sample recovery strategy of the DDPM

        # Get precomputed alphas and alpha bars
        betas = self.var_scheduler.betas.to(device)[timestep].view(-1, 1, 1, 1)
        alpha_t = self.var_scheduler.alphas.to(device)[timestep].view(-1, 1, 1, 1)
        alpha_bar_t = self.var_scheduler.alpha_bars.to(device)[timestep].view(-1, 1, 1, 1)
        noise = torch.randn_like(noisy_sample, device=device) if timestep.max() > 0 else 0.0

        # Calculate x_t-1
        x_prev = (1 / torch.sqrt(alpha_t)) * (noisy_sample - ( betas / torch.sqrt(1 - alpha_bar_t) ) * estimated_noise) + torch.sqrt(betas) * noise
        return x_prev

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'),
                        labels: torch.Tensor = None):
        '''
        Iteratively generate samples from pure noise
        :param num_samples: Number of samples to generate
        :param device:      Device
        :param labels:      Labels for which the sample are to be generated for
        :return:
        '''
        shape = (num_samples, 1, 32, 32)  # Example image shape (n_samples, in_channels, H, W)
        samples = torch.randn(shape, device=device)  # Start from pure noise

        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples, ], device=device)
        else:
            labels = None

        # Iterative sample generation of the DDPM
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
            Recover sample using DDIM sampling strategy.
            :param noisy_sample: The current sample in the backwards process (x_t)
            :param estimated_noise: The noise predicted by the UNet (epsilon_theta)
            :param timestep: The current timestep t
            :return: Denoised version of the sample at step t-1 (x_t-1)
            '''
        device = noisy_sample.device

        # Precomputed statistics
        alpha_t = self.var_scheduler.alphas.to(device)[timestep].view(-1, 1, 1, 1)
        alpha_bar_t = self.var_scheduler.alpha_bars.to(device)[timestep].view(-1, 1, 1, 1)

        # For DDIM: Compute x_0 (clean image estimate)
        pred_x0 = (1 / torch.sqrt(alpha_bar_t)) * (noisy_sample - torch.sqrt(1 - alpha_bar_t) * estimated_noise)

        # Compute alpha_bar for t-1
        t_next = timestep - 1
        alpha_bar_t_next = self.var_scheduler.alpha_bars.to(device)[t_next].view(-1, 1, 1, 1)

        # Interpolate x_t-1
        x_t_next = torch.sqrt(alpha_bar_t_next) * pred_x0 + torch.sqrt(1 - alpha_bar_t_next) * estimated_noise

        return x_t_next

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device = torch.device('cuda'),
                        labels: torch.Tensor = None):
        '''
        Iteratively generate samples from pure noise
        :param num_samples: Number of samples to generate
        :param device:      Device
        :param labels:      Labels for which the sample are to be generated for
        :return:
        '''
        shape = (num_samples, 1, 32, 32)  # Example image shape (n_samples, in_channels, H, W)
        samples = torch.randn(shape, device=device)  # Start from pure noise

        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples, ], device=device)
        else:
            labels = None

        # Iterative sample generation of the DDPM
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

