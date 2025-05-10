import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Optional, Tuple

class AdaptiveLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, adapt_dim: int
    ):
        super(AdaptiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.randn(out_features))
        self.adapt = nn.Linear(adapt_dim, out_features * in_features)

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        adapt_weight = self.adapt(adapt_input).view(
            self.out_features, self.in_features
        )
        weight = self.weight + adapt_weight
        return F.linear(x, weight, self.bias)
    
class TokenMixing(nn.Module):
    def __init__(self, token_dim: int, adapt_dim: int):
        super(TokenMixing, self).__init__()
        self.token_mixing = AdaptiveLinear(
            token_dim, token_dim, adapt_dim
        )

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_length, embed_dim = x.shape
        x = x.view(
            batch_size * seq_length, embed_dim
        )
        x_mixed = self.token_mixing(x, adapt_input)
        return x_mixed.view(batch_size, seq_length, embed_dim)
    
class ChannelMixing(nn.Module):
    def __init__(self, channel_dim: int, adapt_dim: int):
        super(ChannelMixing, self).__init__()
        self.channel_mixing = AdaptiveLinear(
            channel_dim, channel_dim, adapt_dim
        )

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        return self.channel_mixing(x, adapt_input)
    
class MixtureOfExperts(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)