import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.benchmarking import (
    run_benchmark_suite,
    analyze_expert_utilization,
    profile_moe_with_different_parameters
)

def main():
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Run benchmark suite
    print("\nRunning benchmark suite...")
    router_results, layer_results = run_benchmark_suite(
        batch_sizes=[1, 4, 8],
        seq_len=1024,
        input_dim=1024,
        hidden_dim=4096,
        output_dim=1024,
        expert_counts=[8, 16, 32],
        k_values=[1, 2, 4],
        device=device
    )
    
    # Analyze expert utilization
    print("\nAnalyzing expert utilization...")
    expert_counts = analyze_expert_utilization(
        batch_size=8,
        seq_len=512,
        input_dim=768,
        num_experts=16,
        k=2,
        capacity_factor=1.5,
        device=device
    )
    
    # Profile with different parameters
    print("\nProfiling with different parameters...")
    profile_results = profile_moe_with_different_parameters(
        batch_size=4,
        seq_len=1024,
        input_dim=1024,
        hidden_dim=4096,
        output_dim=1024,
        num_experts=8,
        k=2,
        capacity_factor=1.5,
        device=device
    )

if __name__ == "__main__":
    main() 