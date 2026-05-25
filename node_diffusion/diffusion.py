import torch
import numpy as np


class GaussianDiffusion:
    """
    DDPM with x0-prediction (direct clean data prediction).
    Linear beta schedule, T=1000 timesteps.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = timesteps

        betas           = torch.linspace(beta_start, beta_end, timesteps)
        alphas          = 1.0 - betas
        alphas_bar      = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])

        self.betas                    = betas
        self.alphas                   = alphas
        self.alphas_bar               = alphas_bar
        self.alphas_bar_prev          = alphas_bar_prev
        self.sqrt_alphas_bar          = alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1 - alphas_bar).sqrt()

        # posterior variance for reverse step
        self.posterior_variance = (
            betas * (1 - alphas_bar_prev) / (1 - alphas_bar)
        ).clamp(min=1e-20)

    def _to(self, device):
        for attr in ['betas', 'alphas', 'alphas_bar', 'alphas_bar_prev',
                     'sqrt_alphas_bar', 'sqrt_one_minus_alphas_bar',
                     'posterior_variance']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, x0, t, noise=None):
        """Add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        return s1 * x0 + s2 * noise, noise

    def training_losses(self, model, x0, t, model_kwargs, snr_gamma=5.0):
        """
        x0-prediction MSE loss with Min-SNR weighting.
        weight(t) = min(SNR(t), gamma), SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
        Focuses training on low-noise timesteps, reduces blur from high-noise steps.
        """
        self._to(x0.device)
        x0    = x0.float()
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)

        pred_x0 = model(xt, t, **model_kwargs)              # [B, 2, 40]

        # per-sample Min-SNR weight
        ab  = self.alphas_bar[t]                            # [B]
        snr = ab / (1 - ab)                                 # [B]
        w   = snr.clamp(max=snr_gamma)                      # [B]
        w   = w.view(-1, 1, 1)                              # [B, 1, 1]

        loss = w * (pred_x0 - x0) ** 2                     # [B, 2, 40]

        # mask out padding nodes: node_mask [B, 40] → [B, 1, 40]
        mask = model_kwargs['node_mask'].float().unsqueeze(1)
        loss = (loss * mask).sum() / (mask.sum() * 2 + 1e-8)
        return loss

    @torch.no_grad()
    def p_sample_loop(self, model, shape, model_kwargs, device, clamp=200.0):
        """
        Full DDPM reverse diffusion from T → 0.
        Returns x0 with shape `shape`.
        """
        self._to(device)
        model.eval()

        x = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            ts = torch.full((shape[0],), t, device=device, dtype=torch.long)

            x0_pred = model(x, ts, **model_kwargs)
            x0_pred = x0_pred.clamp(-clamp, clamp)

            alpha_bar      = self.alphas_bar[t]
            alpha_bar_prev = self.alphas_bar_prev[t]
            beta           = self.betas[t]
            alpha          = self.alphas[t]

            # posterior mean from predicted x0
            mean = (
                alpha_bar_prev.sqrt() * beta / (1 - alpha_bar) * x0_pred
                + alpha.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar) * x
            )

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + self.posterior_variance[t].sqrt() * noise
            else:
                x = mean

        return x
