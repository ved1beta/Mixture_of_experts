import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class MoERouter(nn.Module):
    """
    Router module for Mixture-of-Experts that handles token-to-expert assignment.
    
    This implementation includes:
    - Top-k routing
    - Load balancing
    - Expert capacity limiting
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        k: int = 2,
        capacity_factor: float = 1.5,
        router_jitter: bool = True,
        use_aux_loss: bool = True
    ):
        """
        Args:
            input_dim: Dimension of input features
            num_experts: Number of experts in the MoE layer
            k: Number of experts to route to for each token
            capacity_factor: Factor to determine expert capacity (higher = more flexibility but more computation)
            router_jitter: Whether to add noise during routing (training only)
            use_aux_loss: Whether to use auxiliary load balancing loss
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.router_jitter = router_jitter
        self.use_aux_loss = use_aux_loss
        
        # Router projection - maps input tokens to expert scores
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Initialize with small weights for more balanced routing at start
        nn.init.normal_(self.router.weight, mean=0, std=0.01)
    
    def forward(self, x: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Route input tokens to their top-k experts.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            is_training: Whether in training mode (affects routing jitter)
            
        Returns:
            Dictionary with routing results including:
            - dispatch_tensor: Binary tensor indicating which tokens go to which experts
            - combine_tensor: Weight tensor for combining expert outputs
            - expert_metrics: Dictionary of metrics about expert assignment
            - aux_loss: Load balancing auxiliary loss (if enabled)
        """
        batch_size, seq_len, _ = x.shape
        
        # Calculate expert capacity - how many tokens each expert can process
        # Use ceiling to ensure it's an integer and there's enough capacity
        tokens_per_batch = batch_size * seq_len
        tokens_per_expert = tokens_per_batch / self.num_experts
        capacity = int(tokens_per_expert * self.capacity_factor)
        
        # Get router logits
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # Add jitter during training if enabled
        if is_training and self.router_jitter:
            # Add small amount of noise to encourage exploration
            router_logits += torch.randn_like(router_logits) * 1e-2
        
        # Get routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # Calculate auxiliary load balancing loss if enabled
        aux_loss = None
        if is_training and self.use_aux_loss:
            # Compute the auxiliary loss to encourage balanced expert assignment
            # We want router assignment to be uniform across experts
            # Loss is minimized when each expert gets same number of tokens
            router_prob_per_expert = router_probs.mean(dim=[0, 1])  # [num_experts]
            
            # Ideal distribution would be uniform across experts
            target_distribution = torch.ones_like(router_prob_per_expert) / self.num_experts
            
            # We want to minimize the difference between actual and target distribution
            # Using mean squared error as the loss
            aux_loss = F.mse_loss(router_prob_per_expert, target_distribution)
        
        # Find top-k experts for each token
        # Shape: [batch_size, seq_len, k]
        top_k_probs, top_k_indices = torch.topk(router_probs, self.k, dim=-1)
        
        # Normalize probabilities of selected experts
        # This ensures weights sum to 1 for each token
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Create dispatch and combine tensors for efficient computation
        # Dispatch tensor is binary - just indicates which expert to use
        # Combine tensor contains the weights for combining expert outputs
        
        # Initialize with zeros
        dispatch_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, self.k,
            device=x.device, dtype=x.dtype
        )
        combine_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, self.k,
            device=x.device, dtype=x.dtype
        )
        
        # Track expert assignment counts for capacity limiting
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.int32)
        
        # Track if tokens were dropped due to capacity limits
        dropped_tokens = torch.zeros(batch_size, seq_len, self.k, device=x.device, dtype=torch.bool)
        
        # Fill dispatch and combine tensors while respecting capacity constraints
        for b in range(batch_size):
            for s in range(seq_len):
                for i in range(self.k):
                    expert_idx = top_k_indices[b, s, i].item()
                    
                    # Check if expert has capacity
                    if expert_counts[expert_idx] < capacity:
                        # Route token to this expert
                        dispatch_tensor[b, s, expert_idx, i] = 1.0
                        combine_tensor[b, s, expert_idx, i] = top_k_probs[b, s, i]
                        expert_counts[expert_idx] += 1
                    else:
                        # Expert is at capacity, token is dropped for this expert
                        dropped_tokens[b, s, i] = True
        
        # Calculate expert assignment metrics
        total_tokens = batch_size * seq_len * self.k
        dropped_token_count = dropped_tokens.sum().item()
        dropped_token_rate = dropped_token_count / total_tokens
        
        # Calculate expert utilization
        expert_utilization = expert_counts / capacity
        
        # Calculate load balance metrics
        if expert_counts.sum() > 0:
            fraction_per_expert = expert_counts / expert_counts.sum()
            # Calculate entropy of expert assignment (higher = more balanced)
            # Add small epsilon to prevent log(0)
            entropy = -(fraction_per_expert * torch.log(fraction_per_expert + 1e-10)).sum()
            # Normalize by log(num_experts) so max entropy is 1.0
            normalized_entropy = entropy / torch.log(torch.tensor(self.num_experts, dtype=torch.float32))
        else:
            normalized_entropy = torch.tensor(0.0, device=x.device)
        
        # Gather all expert metrics
        expert_metrics = {
            "dropped_token_rate": dropped_token_rate,
            "expert_utilization": expert_utilization,
            "expert_counts": expert_counts,
            "normalized_entropy": normalized_entropy.item(),
        }
        
        return {
            "dispatch_tensor": dispatch_tensor,
            "combine_tensor": combine_tensor,
            "router_probs": router_probs,
            "top_k_indices": top_k_indices,
            "top_k_probs": top_k_probs,
            "expert_metrics": expert_metrics,
            "aux_loss": aux_loss
        }
