import torch


class GaussianDiffusion:
    """
    DDPM with epsilon-prediction over node coordinates only.
    Linear noise schedule (matches pretrained checkpoint).
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = timesteps

        betas           = torch.linspace(beta_start, beta_end, timesteps)
        alphas          = 1.0 - betas
        alphas_bar      = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])

        self.betas                     = betas
        self.alphas                    = alphas
        self.alphas_bar                = alphas_bar
        self.alphas_bar_prev           = alphas_bar_prev
        self.sqrt_alphas_bar           = alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1 - alphas_bar).sqrt()
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
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        return s1 * x0 + s2 * noise, noise

    def training_losses(self, model, x0, t, model_kwargs):
        self._to(x0.device)
        x0 = x0.float()

        coord_noise = torch.randn_like(x0)
        xt, _       = self.q_sample(x0, t, coord_noise)

        pred_coord_noise = model(xt, t, **model_kwargs)

        node_mask  = model_kwargs['node_mask'].float()
        coord_mask = node_mask.unsqueeze(1)                          # [B, 1, N]

        coord_loss = (
            (pred_coord_noise - coord_noise) ** 2 * coord_mask
        ).sum() / (coord_mask.sum() * 2 + 1e-8)

        with torch.no_grad():
            s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
            s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
            pred_x0    = (xt - s2 * pred_coord_noise) / s1
            raw_mse    = ((pred_x0 - x0) ** 2 * coord_mask).sum() / (coord_mask.sum() * 2 + 1e-8)
            coord_rmse = raw_mse.sqrt().item()

        return coord_loss, coord_rmse
