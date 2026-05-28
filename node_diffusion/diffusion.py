import torch


class GaussianDiffusion:
    """
    DDPM with epsilon-prediction (predicting added noise).
    Linear beta schedule, T=1000 timesteps.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.alphas_bar_prev = alphas_bar_prev
        self.sqrt_alphas_bar = alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1 - alphas_bar).sqrt()

        self.posterior_variance = (
            betas * (1 - alphas_bar_prev) / (1 - alphas_bar)
        ).clamp(min=1e-20)

    def _to(self, device):
        for attr in [
            "betas",
            "alphas",
            "alphas_bar",
            "alphas_bar_prev",
            "sqrt_alphas_bar",
            "sqrt_one_minus_alphas_bar",
            "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, x0, t, noise=None):
        """Add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
        return s1 * x0 + s2 * noise, noise

    def training_losses(self, model, x0, t, model_kwargs):
        """Standard epsilon-prediction MSE loss."""
        self._to(x0.device)
        x0 = x0.float()
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)

        pred_noise = model(xt, t, **model_kwargs)

        mask = model_kwargs["node_mask"].float().unsqueeze(1)
        loss = ((pred_noise - noise) ** 2 * mask).sum() / (mask.sum() * 2 + 1e-8)

        with torch.no_grad():
            s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1)
            s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1)
            pred_x0 = (xt - s2 * pred_noise) / s1
            raw_mse = ((pred_x0 - x0) ** 2 * mask).sum() / (mask.sum() * 2 + 1e-8)
            coord_rmse = raw_mse.sqrt().item() * 160.0

        return loss, coord_rmse

    @torch.no_grad()
    def p_sample_loop(self, model, shape, model_kwargs, device, clamp=200.0):
        """
        Full DDPM reverse diffusion from T -> 0.
        Returns x0 with shape `shape`.
        """
        self._to(device)
        model.eval()

        x = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            ts = torch.full((shape[0],), t, device=device, dtype=torch.long)

            eps_pred = model(x, ts, **model_kwargs)
            alpha_bar = self.alphas_bar[t]
            x0_pred = (x - (1 - alpha_bar).sqrt() * eps_pred) / alpha_bar.sqrt()
            x0_pred = x0_pred.clamp(-clamp, clamp)

            alpha_bar_prev = self.alphas_bar_prev[t]
            beta = self.betas[t]
            alpha = self.alphas[t]

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
