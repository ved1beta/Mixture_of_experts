import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

from src.models.router import MoERouter
from src.models.moe import MixtureOfExperts

def benchmark_moe_router(
    batch_size: int,
    seq_len: int,
    input_dim: int,
    num_experts: int,
    k: int = 2,
    capacity_factor: float = 1.5,
    num_runs: int = 10,
    device: str = "cuda"
) -> Dict[str, float]:
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
) -> Dict[str, float]:
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

def run_benchmark_suite(
    batch_sizes: List[int] = [1, 4, 8],
    seq_len: int = 1024,
    input_dim: int = 1024,
    hidden_dim: int = 4096,
    output_dim: int = 1024,
    expert_counts: List[int] = [8, 16, 32],
    k_values: List[int] = [1, 2, 4],
    device: str = "cuda"
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Run a suite of benchmarks for MoE router and layer.
    
    Returns:
        Tuple of (router_results, layer_results)
    """
    print("Starting MoE Benchmark Suite...")
    
    # Results storage
    router_results = {}
    layer_results = {}
    
    for batch_size in batch_sizes:
        for num_experts in expert_counts:
            for k in k_values:
                config_name = f"bs{batch_size}_seq{seq_len}_d{input_dim}_e{num_experts}_k{k}"
                
                print(f"\nTesting configuration: {config_name}")
                
                # Router benchmark
                try:
                    print("  Benchmarking router...")
                    router_res = benchmark_moe_router(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        input_dim=input_dim,
                        num_experts=num_experts,
                        k=k,
                        num_runs=20,
                        device=device
                    )
                    router_results[config_name] = router_res
                    print(f"  Router: {router_res['avg_time_ms']:.2f} ms, {router_res['tokens_per_second']:.2f} tokens/s")
                except Exception as e:
                    print(f"  Router benchmarking failed: {str(e)}")
                    print("  Skipping to next configuration...")
                    continue
                
                # Full MoE layer benchmark
                try:
                    print("  Benchmarking full MoE layer...")
                    layer_res = benchmark_moe_layer(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        num_experts=num_experts,
                        k=k,
                        num_runs=10,
                        device=device
                    )
                    layer_results[config_name] = layer_res
                    print(f"  Full MoE: {layer_res['avg_time_ms']:.2f} ms, {layer_res['tokens_per_second']:.2f} tokens/s")
                except Exception as e:
                    print(f"  MoE layer benchmarking failed: {str(e)}")
                    print("  Skipping to next configuration...")
                    continue
    
    # Print summary of successful tests
    print("\n--- SUMMARY ---")
    
    if router_results:
        print("\nRouter Performance:")
        for config, res in router_results.items():
            print(f"{config}: {res['avg_time_ms']:.2f} ms, {res['tokens_per_second']:.2f} tokens/s")
    else:
        print("\nNo successful router benchmarks completed.")
    
    if layer_results:
        print("\nFull MoE Layer Performance:")
        for config, res in layer_results.items():
            print(f"{config}: {res['avg_time_ms']:.2f} ms, {res['tokens_per_second']:.2f} tokens/s")
    else:
        print("\nNo successful MoE layer benchmarks completed.")
    
    return router_results, layer_results

def analyze_expert_utilization(
    batch_size: int = 8,
    seq_len: int = 512,
    input_dim: int = 768,
    num_experts: int = 16,
    k: int = 2,
    capacity_factor: float = 1.5,
    device: str = "cuda"
) -> np.ndarray:
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

def profile_moe_with_different_parameters(
    batch_size: int = 4,
    seq_len: int = 1024,
    input_dim: int = 1024,
    hidden_dim: int = 4096,
    output_dim: int = 1024,
    num_experts: int = 8,
    k: int = 2,
    capacity_factor: float = 1.5,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Profile MoE performance with different parameters.
    """
    print("Profiling MoE with different parameters...")
    
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
            num_runs=10,
            device=device
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
            num_runs=10,
            device=device
        )
        results[config_name] = res
        print(f"  {config_name}: {res['avg_time_ms']:.2f} ms, {res['tokens_per_second']:.2f} tokens/s")
    
    return results 