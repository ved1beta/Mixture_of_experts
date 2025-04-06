import torch
import torch.nn as nn
from typing import Dict, Tuple

from .moe import MixtureOfExperts

class MoETransformerBlock(nn.Module):
    """
    Transformer block that uses MoE for the feed-forward network layer.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        num_experts: int,
        k: int = 2,
        dropout: float = 0.1,
        capacity_factor: float = 1.5,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # MoE layer instead of standard feed-forward
        self.moe = MixtureOfExperts(
            input_dim=hidden_dim,
            hidden_dim=ff_dim,
            output_dim=hidden_dim,
            num_experts=num_experts,
            k=k,
            capacity_factor=capacity_factor,
            dropout=dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, is_training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through transformer block with MoE.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            is_training: Whether in training mode
            
        Returns:
            Tuple of:
            - output: Processed tensor of shape [batch_size, seq_len, hidden_dim]
            - metrics: Dictionary of metrics from MoE execution
        """
        # Self-attention block (pre-LayerNorm style)
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = x + residual
        
        # MoE block (pre-LayerNorm style)
        residual = x
        x = self.norm2(x)
        moe_output, moe_metrics = self.moe(x, is_training=is_training)
        x = self.dropout(moe_output)
        x = x + residual
        
        return x, moe_metrics 