# Mixture of Experts (MoE) Implementation

This project implements a Mixture of Experts (MoE) model with a focus on efficient routing and expert utilization. The implementation includes:

1. A flexible MoE router that can handle different numbers of experts and routing parameters
2. Expert layers implemented as feed-forward networks
3. A complete MoE layer that combines routing and expert computation
4. A transformer block that uses MoE for the feed-forward layer
5. Comprehensive benchmarking utilities

## Project Structure

```
src/
├── models/
│   ├── __init__.py
│   ├── router.py      # MoE router implementation
│   ├── experts.py     # Expert layer implementation
│   ├── moe.py         # Main MoE layer
│   └── transformer.py # Transformer block with MoE
├── utils/
│   ├── __init__.py
│   └── benchmarking.py # Benchmarking utilities
├── triton/
│   ├── README.md      # Triton implementation documentation
│   └── router/        # Triton router implementation
│       ├── top_k_router.py # Triton MoE router
│       └── test_router.py  # Testing utilities
├── benchmark.py       # Main benchmark script
├── profile_moe.py     # Profiling script for memory and performance analysis
└── profiling.py       # General profiling utilities
```

## Features

- Efficient routing algorithm with support for:
  - Top-k expert selection
  - Capacity factor for load balancing
  - Optional router jitter for improved training
- Expert layers with configurable dimensions
- Comprehensive benchmarking suite that measures:
  - Router performance
  - Full MoE layer performance
  - Expert utilization
  - Impact of different parameters (k, capacity factor)
- Advanced profiling tools for:
  - Memory usage analysis
  - Performance bottleneck identification
  - Detailed timing of different components
- Optimized Triton implementation (in progress):
  - Memory-efficient routing
  - Optimized top-k selection
  - Reduced thread divergence

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Benchmarks

Run the benchmark suite:
```bash
python src/benchmark.py
```

This will:
1. Run a comprehensive benchmark suite with different configurations
2. Analyze expert utilization and generate plots
3. Profile performance with different parameters

### Profiling the Implementation

Run the profiling script:
```bash
python src/profile_moe.py
```

This will:
1. Profile the MoE router, MoE layer, and transformer block
2. Generate detailed reports on memory usage and performance
3. Identify key bottlenecks for optimization
4. Update the baseline profiling documentation

### Testing the Triton Implementation

Run the Triton router tests:
```bash
python -m src.triton.router.test_router
```

This will:
1. Test the correctness of the Triton implementation against PyTorch
2. Benchmark the performance of both implementations
3. Generate a report with the results

## Results

The benchmarking and profiling results will be saved as:
- Expert utilization plots: `expert_utilization_k{k}_cf{capacity_factor}.png`
- Profiling reports: `docs/router_profiling_report.md`, `docs/moe_layer_profiling_report.md`, etc.
- Consolidated profiling report: `docs/consolidated_profiling_report.md`
- Baseline profiling documentation: `docs/baseline_profiling.md`
- Triton benchmark results: `triton_router_benchmark_results.txt`

## Project Roadmap

1. ✅ Baseline PyTorch Implementation
   - ✅ Router implementation
   - ✅ Expert implementation
   - ✅ MoE layer implementation
   - ✅ Transformer integration

2. ✅ Evaluation Framework
   - ✅ Benchmarking utilities
   - ✅ Expert utilization analysis
   - ✅ Performance profiling

3. 🔄 Triton Implementation (In Progress)
   - ✅ Interface definition
   - ✅ Basic structure
   - ❌ Core kernels implementation
   - ❌ Optimization and tuning

4. ❌ Advanced Optimizations (Planned)
   - ❌ Memory optimization
   - ❌ Quantization support
   - ❌ Multi-GPU support

## Contributing

Feel free to submit issues and enhancement requests!
