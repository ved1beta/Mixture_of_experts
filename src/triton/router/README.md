# Triton MoE Router Implementation

This directory contains the Triton implementation of the Mixture of Experts (MoE) router.

## Overview

The goal of this implementation is to optimize the router used in MoE models by implementing key operations in Triton. The router is responsible for:

1. Computing routing probabilities for each token across experts
2. Selecting the top-k experts for each token
3. Creating dispatch and combine tensors for efficient computation

## Components

- `router/top_k_router.py`: Main implementation of the Triton-based router
- `router/test_router.py`: Testing and benchmarking utilities

## Implementation Details

### Key Kernels

The implementation consists of several key kernels:

1. `_compute_routing_probabilities_kernel`: Computes the router logits by applying a linear layer to the input
2. `_top_k_gating_kernel`: Finds the top-k experts for each token
3. `_create_dispatch_tensor_kernel`: Creates the dispatch and combine tensors for efficient routing

### Optimizations

The Triton implementation focuses on several key optimizations:

1. **Memory-efficient routing**: Reduce memory usage compared to the PyTorch baseline
2. **Efficient top-k selection**: Optimize the algorithm for selecting top-k experts
3. **Reduced thread divergence**: Minimize thread divergence in CUDA kernels
4. **Optimized memory access patterns**: Use tiling and other techniques to improve memory locality

## Requirements

- PyTorch >= 2.0.0
- Triton >= 2.0.0
- CUDA >= 11.7

## Usage

```python
from src.triton.router.top_k_router import TritonMoERouter

# Create router
router = TritonMoERouter(
    input_dim=768,
    num_experts=8,
    k=2,
    capacity_factor=1.5
)

# Route tokens to experts
output = router.forward(input_tensor)
```

## Benchmarking

To run benchmarks comparing the Triton implementation to the PyTorch baseline:

```bash
python -m src.triton.router.test_router
```

The benchmark results will be saved to `triton_router_benchmark_results.txt`.

## Development Status

- [x] Interface definition
- [x] Basic structure
- [ ] Core kernels implementation
- [ ] Optimization and tuning
- [ ] Multi-GPU support

## Next Steps

1. Implement the core kernels
2. Benchmark and optimize for different GPU architectures
3. Add support for quantization
4. Integrate with the full MoE layer 

# MoE Triton Implementation Todo List

## 1. Core Router Implementation

- [ ] Create Triton kernels for router components
  - [ ] Implement `compute_router_logits_kernel` to replace the linear projection
  - [ ] Implement `add_router_jitter_kernel` for efficient noise addition during training
  - [ ] Implement `compute_router_probs_kernel` for softmax computation
  - [ ] Implement `top_k_routing_kernel` for efficient top-k selection
  - [ ] Implement `dispatch_tensor_kernel` to create dispatch tensors with capacity limiting
  - [ ] Implement `combine_tensor_kernel` to create combine tensors with proper normalization

- [ ] Create TritonMoERouter class that matches the PyTorch MoERouter interface
  - [ ] Implement proper parameter initialization
  - [ ] Add support for router jitter
  - [ ] Add support for capacity limiting
  - [ ] Add support for auxiliary load balancing loss calculation

## 2. Expert Layer Implementation

- [ ] Implement expert network in Triton
  - [ ] Create `expert_ffn_kernel` to handle the feed-forward computation
  - [ ] Implement efficient expert selection and processing

## 3. Full MoE Implementation

- [ ] Create TritonMixtureOfExperts class
  - [ ] Integrate Triton router implementation
  - [ ] Implement efficient expert dispatch and combine operations
  - [ ] Ensure API compatibility with PyTorch implementation

## 4. Benchmarking Framework

- [ ] Extend benchmarking utilities to support Triton implementations
  - [ ] Add `benchmark_triton_moe_router` function
  - [ ] Add `benchmark_triton_moe_layer` function
  - [ ] Add comparative benchmark for PyTorch vs Triton

- [ ] Add performance metrics collection
  - [ ] Memory usage comparison
  - [ ] Throughput (tokens/second)
  - [ ] Latency (ms per batch)
  - [ ] GPU utilization

## 5. Optimization and Testing

- [ ] Implement unit tests for Triton kernels
  - [ ] Test for numerical accuracy compared to PyTorch
  - [ ] Test for edge cases (small batch sizes, large expert counts)

- [ ] Optimize kernels for different hardware configurations
  - [ ] Add tunable parameters for different GPU architectures
  - [ ] Explore tiling strategies for better memory access patterns

## 6. Documentation and Examples

- [ ] Document Triton implementation details
  - [ ] Kernel designs and optimizations
  - [ ] Performance characteristics
  - [ ] Usage examples

- [ ] Create benchmark result visualizations
  - [ ] Comparative graphs showing speedup factors
  - [ ] Memory usage reduction metrics