import torch
import torch.nn as nn
from typing import Dict, Tuple

from .router import MoERouter
from .experts import ExpertLayer

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer.
    Routes tokens to different experts and combines their outputs.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        k: int = 2,
        capacity_factor: float = 1.5,
        dropout: float = 0.1,
        router_jitter: bool = True,
        use_aux_loss: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        
        # Create router
        self.router = MoERouter(
            input_dim=input_dim,
            num_experts=num_experts,
            k=k,
            capacity_factor=capacity_factor,
            router_jitter=router_jitter,
            use_aux_loss=use_aux_loss
        )
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor, is_training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            is_training: Whether in training mode
            
        Returns:
            Tuple of:
            - output: Output tensor of shape [batch_size, seq_len, output_dim]
            - metrics: Dictionary of metrics from router and expert execution
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Get routing assignments
        router_outputs = self.router(x, is_training=is_training)
        
        # Extract dispatch and combine tensors
        dispatch_tensor = router_outputs["dispatch_tensor"]  # [batch_size, seq_len, num_experts, k]
        combine_tensor = router_outputs["combine_tensor"]    # [batch_size, seq_len, num_experts, k]
        
        # Process each expert
        expert_outputs = torch.zeros(
            batch_size, seq_len, self.num_experts, self.output_dim,
            device=x.device, dtype=x.dtype
        )
        
        # Iterate through experts
        for expert_idx, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert
            # Sum over k dimension to get a binary mask
            expert_mask = dispatch_tensor[:, :, expert_idx].sum(dim=-1) > 0  # [batch_size, seq_len]
            
            if expert_mask.sum() > 0:
                # Select tokens for this expert
                # Need to reshape for the expert
                expert_input = x[expert_mask]  # [num_tokens, input_dim]
                
                # Process selected tokens
                processed = expert(expert_input)  # [num_tokens, output_dim]
                
                # Place results back into output tensor
                # We need to expand the mask to match output dimensions
                expanded_mask = expert_mask.unsqueeze(-1).expand(-1, -1, self.output_dim)
                expert_outputs[:, :, expert_idx][expanded_mask] = processed.flatten()
        
        # Combine expert outputs using weights from router
        # First, reshape to align dimensions
        expert_outputs = expert_outputs.view(batch_size, seq_len, self.num_experts, self.output_dim)
        combine_tensor = combine_tensor.unsqueeze(-1)  # [batch_size, seq_len, num_experts, k, 1]
        
        # Weighted sum of expert outputs
        # Sum over expert and k dimensions
        combined_output = torch.sum(
            expert_outputs.unsqueeze(3) * combine_tensor,  # [batch_size, seq_len, num_experts, k, output_dim]
            dim=[2, 3]  # Sum over expert and k dimensions
        )
        
        # Return output and metrics
        return combined_output, {
            "router": router_outputs["expert_metrics"],
            "aux_loss": router_outputs["aux_loss"]
        } 