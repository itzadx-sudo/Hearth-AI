from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import VITALS, N_FEATURES, NUM_CLASSES


# weighted focal loss — penalises easy Healthy samples, pushes harder on Critical
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            raw_alpha = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32)
            self.alpha = raw_alpha / raw_alpha.sum()
        else:
            self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha           = self.alpha.to(logits.device)
        probs           = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        p_t             = (probs * targets_one_hot).sum(dim=1)
        alpha_t         = alpha[targets]
        focal_weight    = (1.0 - p_t).pow(self.gamma)
        ce_loss         = -torch.log(p_t + 1e-8)
        return (alpha_t * focal_weight * ce_loss).mean()


# splits batch into virtual chunks so BN stats stay stable at large batch sizes
class GhostBatchNorm(nn.Module):
    def __init__(self, n_features: int, virtual_batch_size: int = 128, momentum: float = 0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features, momentum=momentum)
        self.virtual_batch_size = virtual_batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) > self.virtual_batch_size:
            chunks = x.split(self.virtual_batch_size)
            return torch.cat([self.bn(chunk) for chunk in chunks], dim=0)
        return self.bn(x)


# gated linear unit — acts as a learnable feature selector per step
class GLUBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, virtual_batch_size: int = 128):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2, bias=False)
        self.bn = GhostBatchNorm(out_features * 2, virtual_batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.fc(x))
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


# per-step soft mask over input features; output is the attention vector used for XAI
class AttentiveTransformer(nn.Module):
    def __init__(self, in_features: int, out_features: int, virtual_batch_size: int = 128):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn = GhostBatchNorm(out_features, virtual_batch_size)

    def forward(self, x: torch.Tensor, prior_scales: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.fc(x)) * prior_scales
        return F.softmax(x, dim=-1)


# shared processing block with residual GLU layers inside each TabNet step
class FeatureTransformer(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int,
                 n_independent: int = 2, virtual_batch_size: int = 128):
        super().__init__()
        self.initial_fc = nn.Linear(in_features, hidden_dim)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)
        self.layers = nn.ModuleList([
            GLUBlock(hidden_dim, hidden_dim, virtual_batch_size)
            for _ in range(n_independent)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_bn(self.initial_fc(x))
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = (x + residual) * math.sqrt(0.5)
        return x


# dual-head TabNet: status (3-class) + risk (scalar); return_attention exposes XAI weights
class TabNet(nn.Module):
    def __init__(self, input_dim: int = N_FEATURES, output_dim: int = NUM_CLASSES,
                 n_d: int = 32, n_a: int = 32, n_steps: int = 5, gamma: float = 1.5,
                 n_independent: int = 2, virtual_batch_size: int = 256):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.n_steps    = n_steps
        self.gamma      = gamma
        self.n_d        = n_d
        self.n_a        = n_a
        hidden_dim      = n_d + n_a

        self.initial_bn        = nn.BatchNorm1d(input_dim)
        self.initial_embedding = nn.Linear(input_dim, hidden_dim)
        self.step_transformers = nn.ModuleList([
            FeatureTransformer(input_dim, hidden_dim, n_independent, virtual_batch_size)
            for _ in range(n_steps)
        ])
        self.attention_transformers = nn.ModuleList([
            AttentiveTransformer(n_a, input_dim, virtual_batch_size)
            for _ in range(n_steps)
        ])
        self.status_head = nn.Linear(n_d, output_dim)
        self.risk_head   = nn.Sequential(
            nn.Linear(n_d, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size    = x.size(0)
        x_normalized  = self.initial_bn(x)
        prior_scales  = torch.ones(batch_size, self.input_dim, device=x.device)
        output_agg    = torch.zeros(batch_size, self.n_d, device=x.device)
        attention_agg = torch.zeros(batch_size, self.input_dim, device=x.device)
        current_input = x_normalized

        # each step picks which features to focus on via attention mask
        for step in range(self.n_steps):
            h    = self.step_transformers[step](current_input)
            d, a = h[:, :self.n_d], h[:, self.n_d:]
            output_agg = output_agg + F.relu(d)
            mask          = self.attention_transformers[step](a, prior_scales)
            prior_scales  = prior_scales * (self.gamma - mask)
            current_input = mask * x_normalized
            attention_agg = attention_agg + mask

        status_logits = self.status_head(output_agg)
        risk_logits   = self.risk_head(output_agg)
        if return_attention:
            attention_agg = attention_agg / max(1, self.n_steps)
            return status_logits, risk_logits, attention_agg
        return status_logits, risk_logits, None

    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            _, _, attention = self.forward(x, return_attention=True)
            avg_attention = attention.mean(dim=0).cpu().numpy()
        feature_names = VITALS + ["activity", "delta_hr", "delta_spo2"]
        return {name: float(avg_attention[i]) for i, name in enumerate(feature_names)}
