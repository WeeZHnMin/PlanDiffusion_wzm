import torch
import torch.nn.functional as F


class GaussianDiffusion:
    """
    DDPM with epsilon-prediction (predicting added noise).
    Linear beta schedule, T=1000 timesteps.
    Also predicts node types via auxiliary cross-entropy loss.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 type_loss_weight=1.0):
        self.T = timesteps
        self.type_loss_weight = type_loss_weight

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
        """
        Epsilon-prediction MSE loss + node type cross-entropy loss.
        model returns (epsilon [B,2,40], type_logits [B,40,33])
        model_kwargs must contain 'node_types' [B,40] with values 0-32.
        """
        self._to(x0.device)
        x0    = x0.float()
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)

        pred_noise, type_logits = model(xt, t, **model_kwargs)

        node_mask = model_kwargs['node_mask'].float()
        mask = node_mask.unsqueeze(1)

        # 坐标 loss（只算有效节点）
        coord_loss = ((pred_noise - noise) ** 2 * mask).sum() / (mask.sum() * 2 + 1e-8)

        # 类型 loss（只算有效节点，ignore_index=0 跳过 padding）
        node_types = model_kwargs['node_types'].long()  # [B, 40]
        B, N, C = type_logits.shape
        type_loss = F.cross_entropy(
            type_logits.reshape(B * N, C),
            node_types.reshape(B * N),
            ignore_index=0,
        )

        loss = coord_loss + self.type_loss_weight * type_loss

        with torch.no_grad():
            s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
            s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
            pred_x0    = (xt - s2 * pred_noise) / s1
            raw_mse    = ((pred_x0 - x0) ** 2 * mask).sum() / (mask.sum() * 2 + 1e-8)
            coord_rmse = raw_mse.sqrt().item() * 160.0

            # 类型预测准确率（有效节点）
            pred_types = type_logits.argmax(dim=-1)  # [B, 40]
            valid      = node_types.ne(0)
            type_acc   = (pred_types.eq(node_types) & valid).sum().item() / (valid.sum().item() + 1e-8)

        return loss, coord_rmse, type_acc
