import torch


class GaussianDiffusion:
    """
    DDPM with epsilon-prediction over both coordinates AND node type embeddings.

    Both coords and types are noised and denoised jointly:
    - Coord noise: added to raw (x,y) coordinates.
    - Type noise : added in the type embedding space (model_channels-dim).
    At inference, denoised type embeddings are decoded to class IDs via
    nearest-neighbor lookup in model.type_embed.weight.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 type_loss_weight=1.0):
        self.T = timesteps
        self.type_loss_weight = type_loss_weight

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
        """加噪，支持 [B,2,N] 和 [B,N,d] 两种形状。"""
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        return s1 * x0 + s2 * noise, noise

    def training_losses(self, model, x0, t, model_kwargs):
        """
        Joint epsilon-prediction loss for coordinates and type embeddings.

        model_kwargs must contain 'node_types' [B, N] with values 0-32.
        """
        self._to(x0.device)
        x0 = x0.float()

        # ── 坐标加噪 ──────────────────────────────────────────────────────
        coord_noise = torch.randn_like(x0)
        xt, _       = self.q_sample(x0, t, coord_noise)             # [B, 2, N]

        # ── 类型嵌入加噪 ──────────────────────────────────────────────────
        node_types  = model_kwargs['node_types'].long()              # [B, N]
        raw_model   = model.module if hasattr(model, 'module') else model
        type_emb_0  = raw_model.type_embed(node_types).float()      # [B, N, d]
        type_noise  = torch.randn_like(type_emb_0)
        type_xt, _  = self.q_sample(type_emb_0, t, type_noise)      # [B, N, d]

        # ── 前向传播 ──────────────────────────────────────────────────────
        pred_coord_noise, pred_type_noise = model(
            xt, type_xt, t, **model_kwargs
        )

        node_mask  = model_kwargs['node_mask'].float()
        coord_mask = node_mask.unsqueeze(1)                          # [B, 1, N]
        type_mask  = node_mask.unsqueeze(-1)                         # [B, N, 1]

        # ── 坐标 loss ─────────────────────────────────────────────────────
        coord_loss = (
            (pred_coord_noise - coord_noise) ** 2 * coord_mask
        ).sum() / (coord_mask.sum() * 2 + 1e-8)

        # ── 类型 loss（embedding 空间 MSE，padding 节点 mask 掉）──────────
        type_loss = (
            (pred_type_noise - type_noise) ** 2 * type_mask
        ).sum() / (type_mask.sum() * type_emb_0.shape[-1] + 1e-8)

        loss = coord_loss + self.type_loss_weight * type_loss

        # ── 监控指标 ──────────────────────────────────────────────────────
        with torch.no_grad():
            s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
            s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)

            # 坐标 RMSE
            pred_x0    = (xt - s2 * pred_coord_noise) / s1
            raw_mse    = ((pred_x0 - x0) ** 2 * coord_mask).sum() / (coord_mask.sum() * 2 + 1e-8)
            coord_rmse = raw_mse.sqrt().item()

            # 类型准确率：还原预测的 type embedding，最近邻找类别
            pred_type_emb_0 = (type_xt - s2 * pred_type_noise.float()) / s1  # [B, N, d]
            all_embs        = raw_model.type_embed.weight.float()         # [33, d]
            B_, N_ = node_types.shape
            dists       = torch.cdist(
                pred_type_emb_0.reshape(B_ * N_, -1), all_embs
            ).view(B_, N_, -1)
            pred_types  = dists.argmin(-1)                               # [B, N]
            valid       = node_types.ne(0)
            type_acc    = (pred_types.eq(node_types) & valid).sum().item() / (valid.sum().item() + 1e-8)

        return loss, coord_rmse, type_acc
