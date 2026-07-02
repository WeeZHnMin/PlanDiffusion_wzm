import math

import torch


class GaussianDiffusion:
    """
    DDPM with epsilon-prediction over node coordinates only.
    Cosine noise schedule (Nichol & Dhariwal 2021).
    """

    def __init__(self, timesteps=1000):
        self.T = timesteps

        # cosine schedule
        t          = torch.arange(timesteps + 1) / timesteps
        f          = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_bar = f / f[0]
        betas      = (1 - alphas_bar[1:] / alphas_bar[:-1]).clamp(max=0.999)
        alphas_bar = alphas_bar[1:]          # drop the prepended 1.0

        alphas          = 1.0 - betas
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

    T_HIGH = 700  # t > T_HIGH: ring-centroid supervision; t <= T_HIGH: per-node supervision

    def training_losses(self, model, x0, t, model_kwargs, step=None):
        self._to(x0.device)
        x0 = x0.float()

        coord_noise = torch.randn_like(x0)
        xt, _       = self.q_sample(x0, t, coord_noise)

        pred_coord_noise = model(xt, t, **model_kwargs)

        node_mask  = model_kwargs['node_mask'].float()
        coord_mask = node_mask.unsqueeze(1)                          # [B, 1, N]

        s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)

        # ── build centroid ε-target ──────────────────────────────────────────
        membership     = model_kwargs['room_membership'].float()     # [B, N, MAX_ROOMS]
        ring_sizes     = membership.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, MAX_ROOMS]
        ring_cent_x0   = torch.bmm(x0, membership) / ring_sizes     # [B, 2, MAX_ROOMS]
        node_n_rings   = membership.sum(dim=2, keepdim=True).clamp(min=1).transpose(1, 2)  # [B, 1, N]
        node_cent_x0   = torch.bmm(ring_cent_x0, membership.transpose(1, 2)) / node_n_rings  # [B, 2, N]
        eps_target_cent = (xt - s1 * node_cent_x0) / s2.clamp(min=1e-3)  # ε that → centroid

        # ── per-sample switch: high-t uses centroid target ───────────────────
        high_t = (t > self.T_HIGH).float().view(-1, 1, 1)           # [B, 1, 1]
        eps_target = high_t * eps_target_cent + (1 - high_t) * coord_noise

        loss = (
            (pred_coord_noise - eps_target) ** 2 * coord_mask
        ).sum() / (coord_mask.sum() * 2 + 1e-8)

        # ── bookkeeping losses (for logging) ─────────────────────────────────
        coord_loss = (
            (pred_coord_noise - coord_noise) ** 2 * coord_mask
        ).sum() / (coord_mask.sum() * 2 + 1e-8)
        centroid_loss = (
            (pred_coord_noise - eps_target_cent) ** 2 * coord_mask * high_t
        ).sum() / ((coord_mask * high_t).sum() * 2 + 1e-8)

        with torch.no_grad():
            pred_x0    = (xt - s2 * pred_coord_noise) / s1.clamp(min=1e-3)
            raw_mse    = ((pred_x0 - x0) ** 2 * coord_mask).sum() / (coord_mask.sum() * 2 + 1e-8)
            coord_rmse = raw_mse.sqrt().item()

        return loss, coord_loss, centroid_loss, coord_rmse
