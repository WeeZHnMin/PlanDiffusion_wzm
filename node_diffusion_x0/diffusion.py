import math

import torch
import torch.nn.functional as F

TYPE_LOSS_WEIGHT = 0.1   # λ：类型损失权重，可调


class GaussianDiffusion:
    """
    DDPM with x0-prediction over node coordinates.
    Joint training: coordinate x0-prediction + node type classification.
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

    def training_losses(self, model, x0, t, model_kwargs):
        self._to(x0.device)
        x0 = x0.float()

        # 前向加噪
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)

        # 分离 node_types，不传给模型前向（模型 forward 不接收类型标签）
        node_types = model_kwargs.pop('node_types', None)  # [B, N], 1-32, 0=padding

        pred_x0, type_logits = model(xt, t, **model_kwargs)  # [B,2,N], [B,N,33]

        # 还原 model_kwargs（避免修改调用方的 dict）
        if node_types is not None:
            model_kwargs['node_types'] = node_types

        node_mask  = model_kwargs['node_mask'].float()
        coord_mask = node_mask.unsqueeze(1)   # [B, 1, N]

        # 坐标损失：x0-prediction MSE
        coord_loss = (
            (pred_x0 - x0) ** 2 * coord_mask
        ).sum() / (coord_mask.sum() * 2 + 1e-8)

        coord_rmse = (
            ((pred_x0 - x0) ** 2 * coord_mask).sum() / (coord_mask.sum() * 2 + 1e-8)
        ).sqrt().item()

        # 类型损失：交叉熵（ignore_index=0 忽略 padding 节点）
        type_loss = torch.tensor(0.0, device=x0.device)
        type_acc  = 0.0
        if node_types is not None and type_logits is not None:
            B, N, C = type_logits.shape
            type_logits_flat = type_logits.view(B * N, C)
            node_types_flat  = node_types.view(B * N).long()
            type_loss = F.cross_entropy(type_logits_flat, node_types_flat, ignore_index=0)

            with torch.no_grad():
                pred_types = type_logits_flat[:, 1:].argmax(dim=-1) + 1  # 预测范围 1-32
                valid      = node_types_flat != 0
                if valid.any():
                    type_acc = (pred_types[valid] == node_types_flat[valid]).float().mean().item()

        total_loss = coord_loss + TYPE_LOSS_WEIGHT * type_loss

        return total_loss, coord_loss.item(), type_loss.item(), coord_rmse, type_acc

    def p_sample(self, model, xt, t_scalar, model_kwargs):
        """单步逆采样，返回 x_{t-1}、当前步预测的 pred_x0、type_logits。"""
        self._to(xt.device)
        B = xt.shape[0]
        t = torch.full((B,), t_scalar, device=xt.device, dtype=torch.long)

        node_types = model_kwargs.pop('node_types', None)
        pred_x0, type_logits = model(xt, t, **model_kwargs)
        if node_types is not None:
            model_kwargs['node_types'] = node_types

        sqrt_ab     = self.sqrt_alphas_bar[t_scalar]
        sqrt_one_ab = self.sqrt_one_minus_alphas_bar[t_scalar]

        # 从 pred_x0 推算 epsilon，再走标准 DDPM 后验
        pred_eps = (xt - sqrt_ab * pred_x0) / sqrt_one_ab.clamp(min=1e-3)

        if t_scalar == 0:
            return pred_x0, pred_x0, type_logits

        alpha    = self.alphas[t_scalar]
        ab       = self.alphas_bar[t_scalar]
        ab_prev  = self.alphas_bar_prev[t_scalar]
        post_var = self.posterior_variance[t_scalar]

        mu = (ab_prev.sqrt() * self.betas[t_scalar] / (1 - ab) * pred_x0
              + alpha.sqrt() * (1 - ab_prev) / (1 - ab) * xt)
        noise  = torch.randn_like(xt)
        x_prev = mu + post_var.sqrt() * noise
        return x_prev, pred_x0, type_logits
