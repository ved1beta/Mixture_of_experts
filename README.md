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
└── benchmark.py       # Main benchmark script
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

Run the benchmark suite:
```bash
python src/benchmark.py
```

This will:
1. Run a comprehensive benchmark suite with different configurations
2. Analyze expert utilization and generate plots
3. Profile performance with different parameters

## Benchmarking

The benchmark suite tests:
- Different batch sizes (1, 4, 8)
- Different sequence lengths (1024)
- Different expert counts (8, 16, 32)
- Different k values (1, 2, 4)
- Different capacity factors (1.0, 1.5, 2.0)

Results include:
- Average processing time per batch
- Tokens processed per second
- Expert utilization statistics
- Performance impact of different parameters

## Results

The benchmarking results will be saved as:
- Expert utilization plots: `expert_utilization_k{k}_cf{capacity_factor}.png`
- Console output with detailed performance metrics

## Contributing

Feel free to submit issues and enhancement requests!
