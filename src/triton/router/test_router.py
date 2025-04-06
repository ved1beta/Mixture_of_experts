import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import numpy as np
import time
from tabulate import tabulate

from src.models.router import MoERouter
from src.triton.router.top_k_router import TritonMoERouter


def test_correctness(batch_size=2, seq_len=128, input_dim=768, num_experts=8, k=2, capacity_factor=1.5):
    """Test the correctness of the Triton implementation compared to PyTorch."""
    print("\n===== Testing Correctness =====")
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, input_dim, device='cuda')
    
    # Create PyTorch router
    torch_router = MoERouter(
        input_dim=input_dim,
        num_experts=num_experts,
        k=k,
        capacity_factor=capacity_factor,
        router_jitter=False  # Disable jitter for deterministic results
    ).cuda()
    
    # Create Triton router
    triton_router = TritonMoERouter(
        input_dim=input_dim,
        num_experts=num_experts,
        k=k,
        capacity_factor=capacity_factor,
        router_jitter=False  # Disable jitter for deterministic results
    )
    
    # Match the weights for fair comparison
    triton_router.router_weights = torch_router.router_weights
    triton_router.router_bias = torch_router.router_bias
    
    # Run both routers
    print("Running PyTorch router...")
    torch_output = torch_router(x)
    
    print("Running Triton router...")
    triton_output = triton_router.forward(x)
    
    # Compare outputs
    print("\nComparing outputs:")
    
    # Check keys match
    torch_keys = set(torch_output.keys())
    triton_keys = set(triton_output.keys())
    
    print(f"PyTorch keys: {torch_keys}")
    print(f"Triton keys: {triton_keys}")
    
    # Since Triton implementation is incomplete, this will show differences
    # When fully implemented, the outputs should match closely
    
    # Compare a few key tensors when implementation is complete
    # For now, just show shapes to confirm interface compatibility
    print("\nOutput shapes:")
    comparison_table = []
    
    for key in torch_output.keys():
        if key in triton_output:
            if isinstance(torch_output[key], torch.Tensor) and isinstance(triton_output[key], torch.Tensor):
                torch_shape = tuple(torch_output[key].shape)
                triton_shape = tuple(triton_output[key].shape)
                comparison_table.append([key, torch_shape, triton_shape, torch_shape == triton_shape])
    
    print(tabulate(comparison_table, headers=["Tensor", "PyTorch Shape", "Triton Shape", "Shapes Match"]))
    
    # Return True if implementation is complete and outputs match
    # For now, just return False since implementation is incomplete
    return False


def benchmark_performance(batch_sizes=[1, 2, 4, 8], seq_len=128, input_dim=768, num_experts=8, k=2, capacity_factor=1.5, num_runs=100):
    """Benchmark the performance of the Triton implementation compared to PyTorch."""
    print("\n===== Benchmarking Performance =====")
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, input_dim, device='cuda')
        
        # Create PyTorch router
        torch_router = MoERouter(
            input_dim=input_dim,
            num_experts=num_experts,
            k=k,
            capacity_factor=capacity_factor,
            router_jitter=False  # Disable jitter for deterministic results
        ).cuda()
        
        # Create Triton router
        triton_router = TritonMoERouter(
            input_dim=input_dim,
            num_experts=num_experts,
            k=k,
            capacity_factor=capacity_factor,
            router_jitter=False  # Disable jitter for deterministic results
        )
        
        # Match the weights for fair comparison
        triton_router.router_weights = torch_router.router_weights
        triton_router.router_bias = torch_router.router_bias
        
        # Warmup
        for _ in range(10):
            _ = torch_router(x)
            _ = triton_router.forward(x)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        torch_start = time.time()
        for _ in range(num_runs):
            _ = torch_router(x)
        torch.cuda.synchronize()
        torch_end = time.time()
        torch_time = (torch_end - torch_start) / num_runs * 1000  # ms
        
        # Benchmark Triton
        triton_start = time.time()
        for _ in range(num_runs):
            _ = triton_router.forward(x)
        torch.cuda.synchronize()
        triton_end = time.time()
        triton_time = (triton_end - triton_start) / num_runs * 1000  # ms
        
        # Calculate speedup
        speedup = torch_time / triton_time if triton_time > 0 else float('inf')
        
        # Add to results
        results.append({
            'batch_size': batch_size,
            'seq_len': seq_len,
            'torch_time_ms': torch_time,
            'triton_time_ms': triton_time,
            'speedup': speedup
        })
        
        print(f"PyTorch: {torch_time:.3f} ms")
        print(f"Triton: {triton_time:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")
    
    # Print results table
    print("\nPerformance Results:")
    table = [[r['batch_size'], r['seq_len'], r['torch_time_ms'], r['triton_time_ms'], r['speedup']] for r in results]
    print(tabulate(table, headers=["Batch Size", "Seq Len", "PyTorch (ms)", "Triton (ms)", "Speedup"]))
    
    return results


def main():
    """Run all tests and benchmarks."""
    # Make sure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run tests.")
        return
    
    # Test correctness
    success = test_correctness()
    if not success:
        print("\nWarning: Triton implementation is incomplete or incorrect.")
    
    # Benchmark performance
    results = benchmark_performance()
    
    # Save results
    with open("triton_router_benchmark_results.txt", "w") as f:
        f.write("Triton MoE Router Benchmark Results\n")
        f.write("=================================\n\n")
        
        # Hardware info
        f.write("Hardware Information:\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"CUDA Version: {torch.version.cuda}\n\n")
        
        # Performance results
        f.write("Performance Results:\n")
        table = [[r['batch_size'], r['seq_len'], r['torch_time_ms'], r['triton_time_ms'], r['speedup']] for r in results]
        f.write(tabulate(table, headers=["Batch Size", "Seq Len", "PyTorch (ms)", "Triton (ms)", "Speedup"]))
        f.write("\n")


if __name__ == "__main__":
    main() 