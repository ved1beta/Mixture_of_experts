import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.profiling import PerformanceProfiler
from src.models.router import MoERouter
from src.models.moe import MixtureOfExperts
from src.models.transformer import MoETransformerBlock

def profile_router():
    """Profile the MoE router."""
    print("\n===== Profiling MoE Router =====\n")
    
    # Create router
    router = MoERouter(
        input_dim=768,
        num_experts=8,
        k=2,
        capacity_factor=1.5
    )
    
    # Create sample input
    sample_input = torch.randn(4, 256, 768)  # [batch_size, seq_len, hidden_dim]
    
    # Create profiler
    profiler = PerformanceProfiler(router)
    
    # Profile memory usage
    profiler.profile_memory(
        sample_input, 
        batch_sizes=[1, 2, 4, 8]
    )
    
    # Profile performance
    profiler.profile_performance(
        sample_input,
        batch_size=4,
        iterations=50
    )
    
    # Generate report
    report_file = profiler.generate_markdown_report("docs/router_profiling_report.md")
    print(f"\nRouter profiling report saved to: {report_file}")
    
    return profiler

def profile_moe_layer():
    """Profile the full MoE layer."""
    print("\n===== Profiling MoE Layer =====\n")
    
    # Create MoE layer
    moe = MixtureOfExperts(
        input_dim=768,
        hidden_dim=2048,
        output_dim=768,
        num_experts=8,
        k=2,
        capacity_factor=1.5
    )
    
    # Create sample input
    sample_input = torch.randn(4, 256, 768)  # [batch_size, seq_len, hidden_dim]
    
    # Create profiler
    profiler = PerformanceProfiler(moe)
    
    # Profile memory usage
    profiler.profile_memory(
        sample_input, 
        batch_sizes=[1, 2, 4]  # Smaller batch sizes to avoid OOM
    )
    
    # Profile performance
    profiler.profile_performance(
        sample_input,
        batch_size=2,  # Smaller batch size to avoid OOM
        iterations=20
    )
    
    # Generate report
    report_file = profiler.generate_markdown_report("docs/moe_layer_profiling_report.md")
    print(f"\nMoE layer profiling report saved to: {report_file}")
    
    return profiler

def profile_transformer_block():
    """Profile the MoE transformer block."""
    print("\n===== Profiling MoE Transformer Block =====\n")
    
    # Create transformer block
    transformer = MoETransformerBlock(
        hidden_dim=768,
        num_heads=12,
        ff_dim=2048,
        num_experts=8,
        k=2,
        dropout=0.1,
        capacity_factor=1.5
    )
    
    # Create sample input
    sample_input = torch.randn(2, 128, 768)  # [batch_size, seq_len, hidden_dim]
    
    # Create profiler
    profiler = PerformanceProfiler(transformer)
    
    # Profile memory usage
    profiler.profile_memory(
        sample_input, 
        batch_sizes=[1, 2]  # Small batch sizes to avoid OOM
    )
    
    # Profile performance
    profiler.profile_performance(
        sample_input,
        batch_size=1,  # Small batch size to avoid OOM
        iterations=10
    )
    
    # Generate report
    report_file = profiler.generate_markdown_report("docs/transformer_profiling_report.md")
    print(f"\nTransformer block profiling report saved to: {report_file}")
    
    return profiler

def generate_consolidated_report(router_profiler, moe_profiler, transformer_profiler):
    """Generate a consolidated profiling report."""
    print("\n===== Generating Consolidated Report =====\n")
    
    with open("docs/consolidated_profiling_report.md", "w") as f:
        # Title
        f.write("# MoE Implementation Profiling Report\n\n")
        f.write(f"Generated on: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System info
        f.write("## System Information\n\n")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        f.write(f"* Device: {device}\n")
        f.write(f"* PyTorch version: {torch.__version__}\n")
        if device.type == 'cuda':
            f.write(f"* CUDA version: {torch.version.cuda}\n")
            f.write(f"* GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"* GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
        f.write("\n")
        
        # Summary of findings
        f.write("## Summary of Findings\n\n")
        
        f.write("### Memory Usage\n\n")
        f.write("| Component | Batch Size | Peak Memory Usage (MB) |\n")
        f.write("|-----------|------------|------------------------|\n")
        
        # Router memory
        if router_profiler.results['memory']:
            router_mem = router_profiler.results['memory'][-1]['peak_memory']
            if device.type == 'cuda':
                f.write(f"| Router | {router_profiler.results['memory'][-1]['batch_size']} | {router_mem['gpu_peak_mb']:.2f} |\n")
            else:
                f.write(f"| Router | {router_profiler.results['memory'][-1]['batch_size']} | {router_mem['cpu_memory_mb']:.2f} |\n")
        
        # MoE layer memory
        if moe_profiler.results['memory']:
            moe_mem = moe_profiler.results['memory'][-1]['peak_memory']
            if device.type == 'cuda':
                f.write(f"| MoE Layer | {moe_profiler.results['memory'][-1]['batch_size']} | {moe_mem['gpu_peak_mb']:.2f} |\n")
            else:
                f.write(f"| MoE Layer | {moe_profiler.results['memory'][-1]['batch_size']} | {moe_mem['cpu_memory_mb']:.2f} |\n")
        
        # Transformer memory
        if transformer_profiler.results['memory']:
            transformer_mem = transformer_profiler.results['memory'][-1]['peak_memory']
            if device.type == 'cuda':
                f.write(f"| Transformer | {transformer_profiler.results['memory'][-1]['batch_size']} | {transformer_mem['gpu_peak_mb']:.2f} |\n")
            else:
                f.write(f"| Transformer | {transformer_profiler.results['memory'][-1]['batch_size']} | {transformer_mem['cpu_memory_mb']:.2f} |\n")
        
        f.write("\n")
        
        # Performance summary
        f.write("### Performance\n\n")
        f.write("| Component | Execution Time (ms) |\n")
        f.write("|-----------|----------------------|\n")
        
        # Router performance
        if router_profiler.results['timing']:
            f.write(f"| Router | {router_profiler.results['timing'][0]['avg_time_ms']:.2f} |\n")
        
        # MoE layer performance
        if moe_profiler.results['timing']:
            f.write(f"| MoE Layer | {moe_profiler.results['timing'][0]['avg_time_ms']:.2f} |\n")
        
        # Transformer performance
        if transformer_profiler.results['timing']:
            f.write(f"| Transformer | {transformer_profiler.results['timing'][0]['avg_time_ms']:.2f} |\n")
        
        f.write("\n")
        
        # Bottlenecks
        f.write("## Main Bottlenecks\n\n")
        
        # Router bottlenecks
        f.write("### Router Bottlenecks\n\n")
        if router_profiler.results['bottlenecks']:
            f.write("| Operation | Self CPU Time (ms) | Self CUDA Time (ms) |\n")
            f.write("|-----------|-------------------|--------------------|\n")
            for bottleneck in router_profiler.results['bottlenecks'][:3]:
                f.write(f"| {bottleneck['name']} | {bottleneck['self_cpu_time_ms']:.2f} | {bottleneck['self_cuda_time_ms']:.2f} |\n")
        
        f.write("\n")
        
        # MoE layer bottlenecks
        f.write("### MoE Layer Bottlenecks\n\n")
        if moe_profiler.results['bottlenecks']:
            f.write("| Operation | Self CPU Time (ms) | Self CUDA Time (ms) |\n")
            f.write("|-----------|-------------------|--------------------|\n")
            for bottleneck in moe_profiler.results['bottlenecks'][:3]:
                f.write(f"| {bottleneck['name']} | {bottleneck['self_cpu_time_ms']:.2f} | {bottleneck['self_cuda_time_ms']:.2f} |\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("## Optimization Recommendations\n\n")
        
        # Extract top bottlenecks across all components
        all_bottlenecks = []
        if router_profiler.results['bottlenecks']:
            all_bottlenecks.extend(router_profiler.results['bottlenecks'][:2])
        if moe_profiler.results['bottlenecks']:
            all_bottlenecks.extend(moe_profiler.results['bottlenecks'][:2])
        if transformer_profiler.results['bottlenecks']:
            all_bottlenecks.extend(transformer_profiler.results['bottlenecks'][:2])
        
        # Sort by self CUDA time if available, otherwise CPU time
        if device.type == 'cuda':
            all_bottlenecks.sort(key=lambda x: x.get('self_cuda_time_ms', 0), reverse=True)
        else:
            all_bottlenecks.sort(key=lambda x: x['self_cpu_time_ms'], reverse=True)
        
        # Make recommendations
        if all_bottlenecks:
            f.write("Based on profiling, the following areas should be prioritized for optimization in Triton:\n\n")
            for i, bottleneck in enumerate(all_bottlenecks[:5]):
                f.write(f"{i+1}. Optimize `{bottleneck['name']}` operation\n")
        
        f.write("\n")
        
        # Memory recommendations
        f.write("### Memory Optimization Recommendations\n\n")
        f.write("1. Implement more efficient token-to-expert routing to reduce memory footprint\n")
        f.write("2. Optimize the combine operation that currently requires large intermediate tensors\n")
        f.write("3. Consider quantization to reduce overall memory usage\n")
        f.write("4. Use more efficient data layouts in the Triton implementation\n")
        
        f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("The baseline PyTorch implementation provides a solid foundation but has clear performance bottlenecks, ")
        f.write("particularly in the routing and expert combining operations. ")
        f.write("The Triton implementation should focus on optimizing these operations first, ")
        f.write("with particular attention to memory usage patterns identified in this profiling report.\n\n")
        
        # Footer
        f.write("---\n")
        f.write("*This report was automatically generated as part of the MoE Router Optimization Project.*\n")
    
    print(f"Consolidated report saved to: docs/consolidated_profiling_report.md")
    return "docs/consolidated_profiling_report.md"

def main():
    # Ensure docs directory exists
    os.makedirs("docs", exist_ok=True)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Profile router
    router_profiler = profile_router()
    
    # Profile MoE layer
    moe_profiler = profile_moe_layer()
    
    # Profile transformer block
    transformer_profiler = profile_transformer_block()
    
    # Generate consolidated report
    consolidated_report = generate_consolidated_report(
        router_profiler, 
        moe_profiler,
        transformer_profiler
    )
    
    print("\nProfiling complete. See reports in the docs/ directory.")
    print(f"Main report: {consolidated_report}")
    
    # Update baseline_profiling.md with findings
    update_baseline_profiling_doc(router_profiler, moe_profiler, transformer_profiler)
    
    return router_profiler, moe_profiler, transformer_profiler

def update_baseline_profiling_doc(router_profiler, moe_profiler, transformer_profiler):
    """Update the baseline_profiling.md document with findings from profiling."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open("docs/baseline_profiling.md", "r") as f:
        content = f.read()
    
    # Update hardware configuration
    content = content.replace(
        "- GPU: [To be filled after running benchmark]",
        f"- GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A (CPU)'}"
    )
    content = content.replace(
        "- CUDA Version: [To be filled after running benchmark]",
        f"- CUDA Version: {torch.version.cuda if device.type == 'cuda' else 'N/A (CPU)'}"
    )
    content = content.replace(
        "- Available Memory: [To be filled after running benchmark]",
        f"- Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if device.type == 'cuda' 
        else "- Available Memory: N/A (CPU)"
    )
    
    # Extract performance findings
    router_perf = ""
    if router_profiler.results['timing']:
        avg_time = router_profiler.results['timing'][0]['avg_time_ms']
        router_perf += f"- Average execution time: {avg_time:.2f} ms for batch size {router_profiler.results['timing'][0]['batch_size']}\n"
    
    if router_profiler.results['bottlenecks']:
        bottleneck = router_profiler.results['bottlenecks'][0]
        router_perf += f"- Main bottleneck: {bottleneck['name']} operation\n"
    
    # Update router performance
    content = content.replace(
        "- [To be filled after running benchmark]\n- [To be filled after running benchmark]",
        router_perf if router_perf else "- No data available"
    )
    
    # Extract MoE layer findings
    moe_perf = ""
    if moe_profiler.results['timing']:
        avg_time = moe_profiler.results['timing'][0]['avg_time_ms']
        moe_perf += f"- Average execution time: {avg_time:.2f} ms for batch size {moe_profiler.results['timing'][0]['batch_size']}\n"
    
    if moe_profiler.results['bottlenecks']:
        bottleneck = moe_profiler.results['bottlenecks'][0]
        moe_perf += f"- Main bottleneck: {bottleneck['name']} operation\n"
    
    # Update MoE layer performance
    if "### Full MoE Layer Performance" in content:
        key_findings_start = content.find("Key findings:", content.find("### Full MoE Layer Performance"))
        if key_findings_start > 0:
            key_findings_end = content.find("\n\n", key_findings_start)
            if key_findings_end > 0:
                new_content = content[:key_findings_start] + "Key findings:\n" + moe_perf + content[key_findings_end:]
                content = new_content
    
    # Extract router memory usage
    router_mem = ""
    if router_profiler.results['memory']:
        for result in router_profiler.results['memory']:
            if device.type == 'cuda':
                router_mem += f"- Batch size {result['batch_size']}: {result['peak_memory']['gpu_peak_mb']:.2f} MB peak GPU memory\n"
            else:
                router_mem += f"- Batch size {result['batch_size']}: {result['peak_memory']['cpu_memory_mb']:.2f} MB CPU memory\n"
    
    # Update router memory usage
    if "### Router Memory Usage" in content:
        mem_start = content.find("### Router Memory Usage")
        mem_end = content.find("\n\n", mem_start)
        if mem_end > 0:
            dash_lines = content[mem_start:mem_end].count("- ")
            replace_text = "\n".join(["- [To be filled after running benchmark]"] * dash_lines)
            new_content = content[:mem_start] + "### Router Memory Usage\n" + router_mem + content[mem_end:]
            content = new_content
    
    # Write updated content
    with open("docs/baseline_profiling.md", "w") as f:
        f.write(content)
    
    print("\nUpdated docs/baseline_profiling.md with profiling findings")

if __name__ == "__main__":
    main() 