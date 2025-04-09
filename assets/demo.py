import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List, Optional, Union


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


class ExpertLayer(nn.Module):
    """
    Standard feed-forward network used as an expert in MoE.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard FFN
        x = self.w1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


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


def benchmark_moe_router(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    num_experts: int,
    k: int = 2,
    capacity_factor: float = 1.5,
    num_runs: int = 10,
    device: str = "cuda"
):
    """
    Benchmark MoE router performance.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        input_dim: Input dimension
        num_experts: Number of experts
        k: Number of experts per token
        capacity_factor: Expert capacity factor
        num_runs: Number of runs for timing
        device: Device to run on ("cuda" or "cpu")
        
    Returns:
        Dictionary of performance metrics
    """
    # Create random input
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    # Create router
    router = MoERouter(
        input_dim=input_dim,
        num_experts=num_experts,
        k=k,
        capacity_factor=capacity_factor
    ).to(device)
    
    # Warmup
    for _ in range(5):
        _ = router(x)
    
    # Wait for GPU
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timing
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = router(x)
        if device == "cuda":
            torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    tokens_per_second = (batch_size * seq_len) / avg_time
    
    return {
        "avg_time_ms": avg_time * 1000,
        "tokens_per_second": tokens_per_second,
        "total_time_s": total_time
    }


def benchmark_moe_layer(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_experts: int,
    k: int = 2,
    capacity_factor: float = 1.5,
    num_runs: int = 10,
    device: str = "cuda"
):
    """
    Benchmark full MoE layer performance.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        input_dim: Input dimension
        hidden_dim: Hidden dimension in expert FFN
        output_dim: Output dimension
        num_experts: Number of experts
        k: Number of experts per token
        capacity_factor: Expert capacity factor
        num_runs: Number of runs for timing
        device: Device to run on ("cuda" or "cpu")
        
    Returns:
        Dictionary of performance metrics
    """
    # Create random input
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    # Create MoE layer
    moe = MixtureOfExperts(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        k=k,
        capacity_factor=capacity_factor
    ).to(device)
    
    # Warmup
    for _ in range(5):
        _, _ = moe(x)
    
    # Wait for GPU
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timing
    start_time = time.time()
    
    for _ in range(num_runs):
        _, _ = moe(x)
        if device == "cuda":
            torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    tokens_per_second = (batch_size * seq_len) / avg_time
    
    return {
        "avg_time_ms": avg_time * 1000,
        "tokens_per_second": tokens_per_second,
        "total_time_s": total_time
    }


def run_benchmark_suite():
    """
    Run a suite of benchmarks for MoE router and layer.
    """
    print("Starting MoE Benchmark Suite...")
    
    # Configuration sets to test
    configs = [
        # batch_size, seq_len, input_dim, hidden_dim, output_dim, num_experts, k
        (1, 1024, 1024, 4096, 1024, 8, 2),
        (4, 1024, 1024, 4096, 1024, 8, 2),
        (8, 1024, 1024, 4096, 1024, 8, 2),
        (4, 1024, 1024, 4096, 1024, 16, 2),
        (4, 1024, 1024, 4096, 1024, 32, 2),
        (4, 1024, 1024, 4096, 1024, 8, 4),
    ]
    
    # Results storage
    router_results = {}
    layer_results = {}
    
    for config in configs:
        batch_size, seq_len, input_dim, hidden_dim, output_dim, num_experts, k = config
        config_name = f"bs{batch_size}_seq{seq_len}_d{input_dim}_e{num_experts}_k{k}"
        
        print(f"\nTesting configuration: {config_name}")
        
        # Router benchmark
        print("  Benchmarking router...")
        router_res = benchmark_moe_router(
            batch_size=batch_size,
            seq_len=seq_len,
            input_dim=input_dim,
            num_experts=num_experts,
            k=k,
            num_runs=20
        )
        router_results[config_name] = router_res
        print(f"  Router: {router_res['avg_time_ms']:.2f} ms, {router_res['tokens_per_second']:.2f} tokens/s")
        
        # Full MoE layer benchmark
        print("  Benchmarking full MoE layer...")
        layer_res = benchmark_moe_layer(
            batch_size=batch_size,
            seq_len=seq_len,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            k=k,
            num_runs=10
        )
        layer_results[config_name] = layer_res
        print(f"  Full MoE: {layer_res['avg_time_ms']:.2f} ms, {layer_res['tokens_per_second']:.2f} tokens/s")
    
    # Print summary
    print("\n--- SUMMARY ---")
    print("\nRouter Performance:")
    for config, res in router_results.items():
        print(f"{config}: {res['avg_time_ms']:.2f} ms, {res['tokens_per_second']:.2f} tokens/s")
    
    print("\nFull MoE Layer Performance:")
    for config, res in layer_results.items():
        print(f"{config}: {res['avg_time_ms']:.2f} ms, {res['tokens_per_second']:.2f} tokens/s")
    
    # Plot results
    plot_benchmark_results(router_results, layer_results)


def plot_benchmark_results(router_results, layer_results):
    """
    Plot benchmark results.
    """
    # Extract data for plotting
    configs = list(router_results.keys())
    router_times = [router_results[c]["avg_time_ms"] for c in configs]
    layer_times = [layer_results[c]["avg_time_ms"] for c in configs]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot times
    x = np.arange(len(configs))
    width = 0.35
    
    ax1.bar(x - width/2, router_times, width, label='Router')
    ax1.bar(x + width/2, layer_times, width, label='Full MoE')
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time by Configuration')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    
    # Plot throughput
    router_throughput = [router_results[c]["tokens_per_second"] / 1000 for c in configs]  # Convert to K tokens/s
    layer_throughput = [layer_results[c]["tokens_per_second"] / 1000 for c in configs]
    
    ax2.bar(x - width/2, router_throughput, width, label='Router')
    ax2.bar(x + width/2, layer_throughput, width, label='Full MoE')
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Throughput (K tokens/s)')
    ax2.set_title('Throughput by Configuration')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('moe_benchmark_results.png')
    plt.close()


def analyze_expert_utilization(
    batch_size: int = 8,
    seq_len: int = 512,
    input_dim: int = 768,
    num_experts: int = 16,
    k: int = 2,
    capacity_factor: float = 1.5,
    device: str = "cuda"
):
    """
    Analyze expert utilization with different routing parameters.
    """
    # Create random input
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    # Create router
    router = MoERouter(
        input_dim=input_dim,
        num_experts=num_experts,
        k=k,
        capacity_factor=capacity_factor,
        router_jitter=False  # Disable jitter for consistent results
    ).to(device)
    
    # Get routing assignments
    with torch.no_grad():
        router_outputs = router(x, is_training=False)
    
    # Extract metrics
    expert_counts = router_outputs["expert_metrics"]["expert_counts"].cpu().numpy()
    
    # Plot expert utilization
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_experts), expert_counts)
    plt.xlabel('Expert Index')
    plt.ylabel('Token Count')
    plt.title(f'Expert Utilization (capacity_factor={capacity_factor}, k={k})')
    plt.axhline(y=expert_counts.mean(), color='r', linestyle='--', label=f'Average: {expert_counts.mean():.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'expert_utilization_k{k}_cf{capacity_factor}.png')
    plt.close()
    
    # Print statistics
    print(f"Expert utilization statistics (k={k}, capacity_factor={capacity_factor}):")
    print(f"  Mean tokens per expert: {expert_counts.mean():.2f}")
    print(f"  Std dev: {expert_counts.std():.2f}")
    print(f"  Min: {expert_counts.min():.2f}, Max: {expert_counts.max():.2f}")
    print(f"  Imbalance ratio (max/min): {expert_counts.max() / (expert_counts.min() + 1e-10):.2f}")
    
    return expert_counts


def profile_moe_with_different_parameters():
    """
    Profile MoE performance with different parameters.
    """
    print("Profiling MoE with different parameters...")
    
    # Base configuration
    batch_size = 4
    seq_len = 1024
    input_dim = 1024
    hidden_dim = 4096
    output_dim = 1024
    num_experts = 8
    k = 2
    capacity_factor = 1.5
    
    # Parameters to vary
    k_values = [1, 2, 4, 8]
    capacity_factors = [1.0, 1.5, 2.0]
    
    # Results storage
    results = {}
    
    # Test different k values
    print("\nTesting different k values:")
    for k_val in k_values:
        config_name = f"k{k_val}"
        print(f"  Testing {config_name}...")
        
        res = benchmark_moe_layer(
            batch_size=batch_size,
            seq_len=seq_len,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            k=k_val,
            capacity_factor=capacity_factor,
            num_runs=10
        )
        results[config_name] = res
        print(f"  {config_name}: {res['avg_time_ms']:.2f} ms, {res['tokens_per_second']:.2f} tokens/s")
    
    # Test different capacity factors
    print("\nTesting different capacity factors:")
    for cf in capacity_factors:
        config_name = f"cf{cf}"
        print(f"  Testing {config_name}...")
        
        res = benchmark_moe_layer(
            batch_size=batch_size,
            seq_len=seq_len,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            k=k,
            capacity_factor=cf,
            num_runs=10
        )
        results[config_name] = res
        print(f"  {config_name}: {res['avg_time_ms']:.2f} ms, {res['tokens_per_second']:.2f} tokens/s")
    
    # Plot results
    plot_parameter_comparison(results)


def plot_parameter_comparison(results):
    """
    Plot parameter comparison results.
    """
    # Extract data for plotting
    k_configs = [c for c in results.keys() if c.startswith('k')]
    cf_configs = [c for c in results.keys() if c.startswith('cf')]
    
    k_times = [results[c]["avg_time_ms"] for c in k_configs]
    cf_times = [results[c]["avg_time_ms"] for c in cf_configs]
    
    k_throughput = [results[c]["tokens_per_second"] / 1000 for c in k_configs]  # Convert to K tokens/s
    cf_throughput = [results[c]["tokens_per_second"] / 1000 for c in cf_configs]
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot k comparison
    ax1.bar(k_configs, k_times)
    ax1.set_xlabel('k value')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time by k value')
    
    ax2.bar(k_configs, k_throughput)
    ax2.set_xlabel('k value')
    ax2.set_ylabel('Throughput (K tokens/s)')
    ax2.set_title('Throughput by k value')
    
    # Plot capacity factor comparison
    ax3.bar(cf_configs, cf_times)
    ax3.set_xlabel('Capacity Factor')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Execution Time by Capacity Factor')
    
    ax4.bar(cf_configs, cf_throughput)
    ax4.set_xlabel('Capacity Factor')
    ax4.set_ylabel('Throughput (K tokens/s)')
    ax4.set_title('Throughput by Capacity Factor')
    
    plt.tight_layout()
    plt.savefig('parameter_comparison.png')
    plt.close()


if __name__ == "__main__":
    # Check if CUDA is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run benchmark suite
    print("\nRunning benchmark suite...")
    run_benchmark_suite()
    
    # Run expert utilization analysis
    print("\nAnalyzing expert utilization...")
    analyze_expert_utilization(device=device)
    
    # Profile MoE with different parameters
    print("\nProfiling MoE with different parameters...")
    profile_moe_with_different_parameters()
    
    print("\nBenchmarks completed. Check the generated PNG files for visualization results.")