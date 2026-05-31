"""
离散均匀扩散过程（DiGress 风格）。

前向过程（加噪）：
  Q_t     = (1 - β_t) * I  +  β_t/K * 1^T       （单步转移矩阵）
  Q̄_t    = ᾱ_t * I  +  (1 - ᾱ_t)/K * 1^T       （累积转移矩阵）
  x_t ~ Categorical(x_0 @ Q̄_t)                   （直接从 t=0 采样到任意 t）

反向过程（去噪）：
  q(x_{t-1}|x_t, x_0) ∝ (x_0 @ Q̄_{t-1}) * (Q_t 对应列)
  模型预测 x_0，再从后验采样 x_{t-1}
"""

import torch
import torch.nn.functional as F
import numpy as np


def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    steps = np.arange(T + 1)
    f     = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return np.clip(betas, 0.0, 0.999)


class DiscreteUniformTransition:
    def __init__(self, x_classes: int, e_classes: int):
        self.x_classes = x_classes
        self.e_classes = e_classes

    def get_Qt(self, beta_t: torch.Tensor, device):
        """单步转移矩阵 Q_t，shape (bs, K, K)"""
        b = beta_t.view(-1, 1, 1).to(device)
        Kx = self.x_classes
        Ke = self.e_classes
        qt_x = (1 - b) * torch.eye(Kx, device=device) + b / Kx
        qt_e = (1 - b) * torch.eye(Ke, device=device) + b / Ke
        return qt_x, qt_e   # (bs, Kx, Kx), (bs, Ke, Ke)

    def get_Qt_bar(self, alpha_bar_t: torch.Tensor, device):
        """累积转移矩阵 Q̄_t"""
        a = alpha_bar_t.view(-1, 1, 1).to(device)
        Kx = self.x_classes
        Ke = self.e_classes
        qt_bar_x = a * torch.eye(Kx, device=device) + (1 - a) / Kx
        qt_bar_e = a * torch.eye(Ke, device=device) + (1 - a) / Ke
        return qt_bar_x, qt_bar_e


class GaussianNoiseSchedule:
    def __init__(self, T: int):
        self.T = T
        betas      = torch.from_numpy(cosine_beta_schedule(T)).float()
        alphas     = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]])

        self.register = {
            'betas':           betas,
            'alphas':          alphas,
            'alphas_bar':      alphas_bar,
            'alphas_bar_prev': alphas_bar_prev,
        }

    def _to(self, device):
        for k in self.register:
            self.register[k] = self.register[k].to(device)

    def get_beta(self, t_int):
        return self.register['betas'][t_int]

    def get_alpha_bar(self, t_int):
        return self.register['alphas_bar'][t_int]

    def get_alpha_bar_prev(self, t_int):
        return self.register['alphas_bar_prev'][t_int]


def sample_discrete(prob: torch.Tensor) -> torch.Tensor:
    """从 categorical 分布采样。prob: (..., K)，返回 (...) 类别索引。"""
    flat = prob.reshape(-1, prob.shape[-1])
    idx  = torch.multinomial(flat.clamp(min=1e-8), num_samples=1).squeeze(-1)
    return idx.reshape(prob.shape[:-1])


def apply_noise(X: torch.Tensor, E: torch.Tensor, node_mask: torch.Tensor,
                t_int: torch.Tensor, schedule: GaussianNoiseSchedule,
                transition: DiscreteUniformTransition, device):
    """
    给 (X, E) 在时间步 t 加噪。
    X: (B, N, Kx) one-hot，E: (B, N, N, Ke) one-hot
    返回 noisy (Xt, Et)，以及所需的转移参数。
    """
    schedule._to(device)
    B = X.shape[0]

    alpha_bar_t = schedule.get_alpha_bar(t_int)      # (B,)
    beta_t      = schedule.get_beta(t_int)            # (B,)

    Qt_bar_x, Qt_bar_e = transition.get_Qt_bar(alpha_bar_t, device)

    # X_t ~ Categorical(X_0 @ Q̄_t)
    prob_x = torch.bmm(X.reshape(B * X.shape[1], 1, X.shape[2]),
                       Qt_bar_x.unsqueeze(1).expand(B, X.shape[1], -1, -1)
                       .reshape(B * X.shape[1], Qt_bar_x.shape[-2], Qt_bar_x.shape[-1]))
    prob_x = prob_x.squeeze(1).reshape(B, X.shape[1], -1)    # (B, N, Kx)
    Xt = F.one_hot(sample_discrete(prob_x), num_classes=X.shape[-1]).float()

    # E_t ~ Categorical(E_0 @ Q̄_t)，对每条边独立
    N = E.shape[1]
    prob_e = torch.bmm(E.reshape(B * N * N, 1, E.shape[-1]),
                       Qt_bar_e.unsqueeze(1).expand(B, N * N, -1, -1)
                       .reshape(B * N * N, Qt_bar_e.shape[-2], Qt_bar_e.shape[-1]))
    prob_e = prob_e.squeeze(1).reshape(B, N, N, -1)           # (B, N, N, Ke)
    Et = F.one_hot(sample_discrete(prob_e), num_classes=E.shape[-1]).float()

    # 保证对称（无向图）
    Et = (Et + Et.permute(0, 2, 1, 3)) / 2

    # 应用 node_mask
    Xt = Xt * node_mask.unsqueeze(-1).float()
    Et = Et * (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).unsqueeze(-1).float()

    return Xt, Et, {
        'beta_t':           beta_t,
        'alpha_bar_t':      alpha_bar_t,
        'alpha_bar_prev_t': schedule.get_alpha_bar_prev(t_int),
    }


def posterior_sample(Xt: torch.Tensor, pred_X0: torch.Tensor,
                     Et: torch.Tensor, pred_E0: torch.Tensor,
                     node_mask: torch.Tensor, params: dict,
                     transition: DiscreteUniformTransition, device):
    """
    从后验 q(x_{t-1}|x_t, x_0_pred) 采样一步。
    pred_X0, pred_E0 是模型预测的 softmax 概率。
    """
    beta_t  = params['beta_t'].view(-1, 1, 1).to(device)
    a_bar_t = params['alpha_bar_t'].view(-1, 1, 1).to(device)
    a_bar_s = params['alpha_bar_prev_t'].view(-1, 1, 1).to(device)
    B, N, Kx = pred_X0.shape
    Ke = pred_E0.shape[-1]

    # ── 节点 ──────────────────────────────────────────────────
    # q(x_{t-1}|x_t, x_0) ∝ (x_0 @ Q̄_{t-1}) * Q_t[x_{t-1}, x_t]
    # 用均匀转移矩阵的解析形式简化
    # P(x_{t-1}=k | x_t, x_0) ∝ P(x_t | x_{t-1}=k) * P(x_{t-1}=k | x_0)
    # = Q_t[k, x_t] * (Q̄_{t-1} @ x_0)[k]

    # P(x_{t-1}=k | x_0): (B, N, Kx)
    p_xprev_given_x0 = a_bar_s * pred_X0 + (1 - a_bar_s) / Kx

    # Q_t[k, x_t]: 第 k 类对应当前 x_t 的转移概率
    # Q_t[k, x_t] = (1 - beta) if k==x_t else beta/K
    x_t_idx = Xt.argmax(-1)                                  # (B, N)
    q_xt    = beta_t / Kx + (1 - beta_t - beta_t / Kx) * Xt # (B, N, Kx)

    unnorm_x = p_xprev_given_x0 * q_xt
    prob_x   = unnorm_x / (unnorm_x.sum(-1, keepdim=True) + 1e-8)
    Xs       = F.one_hot(sample_discrete(prob_x), num_classes=Kx).float()
    Xs       = Xs * node_mask.unsqueeze(-1).float()

    # ── 边 ────────────────────────────────────────────────────
    p_eprev_given_e0 = a_bar_s.unsqueeze(-1) * pred_E0 + (1 - a_bar_s.unsqueeze(-1)) / Ke
    q_et = (beta_t.unsqueeze(-1) / Ke
            + (1 - beta_t.unsqueeze(-1) - beta_t.unsqueeze(-1) / Ke) * Et)

    unnorm_e = p_eprev_given_e0 * q_et
    prob_e   = unnorm_e / (unnorm_e.sum(-1, keepdim=True) + 1e-8)
    Es       = F.one_hot(sample_discrete(prob_e), num_classes=Ke).float()
    Es       = (Es + Es.permute(0, 2, 1, 3)) / 2
    edge_mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).unsqueeze(-1).float()
    Es       = Es * edge_mask

    return Xs, Es
