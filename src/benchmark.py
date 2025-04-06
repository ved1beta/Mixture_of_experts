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
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Run benchmark suite with smaller configurations to avoid OOM
    print("\nRunning benchmark suite...")
    router_results, layer_results = run_benchmark_suite(
        batch_sizes=[1, 2, 4],  # Reduced from [1, 4, 8]
        seq_len=512,            # Reduced from 1024
        input_dim=768,          # Reduced from 1024
        hidden_dim=2048,        # Reduced from 4096
        output_dim=768,         # Reduced from 1024
        expert_counts=[4, 8, 16],  # Reduced from [8, 16, 32]
        k_values=[1, 2],        # Reduced from [1, 2, 4]
        device=device
    )
    
    # Analyze expert utilization with smaller configuration
    print("\nAnalyzing expert utilization...")
    expert_counts = analyze_expert_utilization(
        batch_size=4,           # Reduced from 8
        seq_len=256,            # Reduced from 512
        input_dim=512,          # Reduced from 768
        num_experts=8,          # Reduced from 16
        k=2,
        capacity_factor=1.5,
        device=device
    )
    
    # Profile with different parameters using smaller config
    print("\nProfiling with different parameters...")
    profile_results = profile_moe_with_different_parameters(
        batch_size=2,           # Reduced from 4
        seq_len=256,            # Reduced from 1024
        input_dim=512,          # Reduced from 1024
        hidden_dim=1024,        # Reduced from 4096
        output_dim=512,         # Reduced from 1024
        num_experts=4,          # Reduced from 8
        k=2,
        capacity_factor=1.5,
        device=device
    )

if __name__ == "__main__":
    main() 