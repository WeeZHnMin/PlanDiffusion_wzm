import math

import torch


def compute_x_centroid(x, room_membership):
    """
    각 노드를 자신이 속한 환(ring)들의 질심 평균으로 투영.

    x              : [B, 2, N]  float
    room_membership: [B, N, R]  float (0/1)
    returns        : [B, 2, N]  — 소속 환이 없는 노드는 0
    """
    x_T = x.permute(0, 2, 1).float()           # [B, N, 2]
    M   = room_membership.float()               # [B, N, R]

    ring_size = M.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, R]
    M_col     = M / ring_size                             # [B, N, R] col-norm
    centroids = torch.bmm(M_col.transpose(1, 2), x_T)   # [B, R, 2]

    n_rings = M.sum(dim=2, keepdim=True).clamp(min=1)   # [B, N, 1]
    M_row   = M / n_rings                                # [B, N, R] row-norm
    x_cen   = torch.bmm(M_row, centroids)                # [B, N, 2]

    return x_cen.permute(0, 2, 1)  # [B, 2, N]


class GaussianDiffusion:
    """
    Residual DDPM: diffuse z_0 = x_0 - x_centroid instead of x_0.

    Training : compute x_centroid from GT coords + room_membership,
               noise z_0, model predicts eps on z_t.
    Inference: at each step recompute x_centroid from current x estimate,
               work in z-space, reconstruct x = z + x_centroid.
    """

    def __init__(self, timesteps=1000):
        self.T = timesteps

        t          = torch.arange(timesteps + 1) / timesteps
        f          = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_bar = f / f[0]
        betas      = (1 - alphas_bar[1:] / alphas_bar[:-1]).clamp(max=0.999)
        alphas_bar = alphas_bar[1:]

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

    def training_losses(self, model, x0, t, model_kwargs):
        self._to(x0.device)
        x0 = x0.float()

        # GT 질심으로 잔차 계산
        x_centroid = compute_x_centroid(x0, model_kwargs['room_membership'])
        z0         = x0 - x_centroid

        coord_noise = torch.randn_like(z0)
        zt, _       = self.q_sample(z0, t, coord_noise)

        pred_coord_noise = model(zt, t, **model_kwargs)

        node_mask  = model_kwargs['node_mask'].float()
        coord_mask = node_mask.unsqueeze(1)   # [B, 1, N]

        coord_loss = (
            (pred_coord_noise - coord_noise) ** 2 * coord_mask
        ).sum() / (coord_mask.sum() * 2 + 1e-8)

        with torch.no_grad():
            s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
            s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
            z0_pred    = (zt - s2 * pred_coord_noise) / s1.clamp(min=1e-3)
            pred_x0    = z0_pred + x_centroid   # 절대 좌표 복원
            raw_mse    = ((pred_x0 - x0) ** 2 * coord_mask).sum() / (coord_mask.sum() * 2 + 1e-8)
            coord_rmse = raw_mse.sqrt().item()

        return coord_loss, coord_rmse
