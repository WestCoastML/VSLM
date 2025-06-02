import torch
import torch.nn.functional as F
import torch.nn as nn
from common import gumbel_softmax


class GumbelSamplingLayer(nn.Module):
    def __init__(self, in_dim, full_out_dim, sample_dim, temperature=1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, full_out_dim)
        self.logits = nn.Parameter(
            torch.randn(sample_dim, full_out_dim)
        )  # learnable selector
        self.sample_dim = sample_dim
        self.temperature = temperature

    def forward(self, x):
        # x: (batch_size, in_dim)
        full_out = self.linear(x)  # (batch_size, full_out_dim)

        # Sample selection mask: (sample_dim, full_out_dim)
        mask_weights = gumbel_softmax(
            self.logits, temperature=self.temperature, hard=True
        )

        # Weighted sum: project from full_out_dim -> sample_dim
        # Output shape: (batch_size, sample_dim)
        sampled_out = torch.matmul(mask_weights, full_out.T).T
        return sampled_out
