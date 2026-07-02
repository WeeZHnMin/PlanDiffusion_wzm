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

    def training_losses(self, model, x0, t, model_kwargs, step=None, inpaint_prob=0.0):
        self._to(x0.device)
        x0 = x0.float()

        node_mask = model_kwargs['node_mask'].float()   # [B, N]

        # ── Inpainting 模式：随机固定部分节点作为锚点 ─────────────────────────
        if inpaint_prob > 0.0 and torch.rand(1).item() < inpaint_prob:
            ratio      = torch.rand(1).item() * 0.4 + 0.3              # 30~70%
            rand_vals  = torch.rand_like(node_mask)
            fixed_mask = ((rand_vals < ratio) & node_mask.bool()).float()  # [B, N]
        else:
            fixed_mask = torch.zeros_like(node_mask)                   # 全量去噪模式

        coord_noise = torch.randn_like(x0)
        xt, _       = self.q_sample(x0, t, coord_noise)

        # 固定节点：用 gt 坐标替换，不加噪
        fixed_coord = fixed_mask.unsqueeze(1)                          # [B, 1, N]
        xt = xt * (1 - fixed_coord) + x0 * fixed_coord

        model_kwargs = dict(model_kwargs, fixed_mask=fixed_mask)
        pred_coord_noise = model(xt, t, **model_kwargs)

        # Loss 只算噪声节点（非固定）
        noisy_mask = node_mask.unsqueeze(1) * (1 - fixed_coord)        # [B, 1, N]
        # 如果全部节点都被固定（极端情况），回退到全量 mask
        if noisy_mask.sum() < 1:
            noisy_mask = node_mask.unsqueeze(1)

        coord_loss = (
            (pred_coord_noise - coord_noise) ** 2 * noisy_mask
        ).sum() / (noisy_mask.sum() * 2 + 1e-8)

        loss = coord_loss
        centroid_loss = torch.tensor(0.0, device=x0.device)

        s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        with torch.no_grad():
            pred_x0    = (xt - s2 * pred_coord_noise) / s1.clamp(min=1e-3)
            coord_mask = node_mask.unsqueeze(1)
            raw_mse    = ((pred_x0 - x0) ** 2 * coord_mask).sum() / (coord_mask.sum() * 2 + 1e-8)
            coord_rmse = raw_mse.sqrt().item()

        return loss, coord_loss, centroid_loss, coord_rmse
