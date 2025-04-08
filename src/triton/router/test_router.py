import torch
import time
from router import MoERouter
import triton
import triton.language as tl

def test_router():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running test on device: {device}")
    
    # Model parameters
    d_model = 512
    num_experts = 8
    top_k = 2
    
    # Create router
    router = MoERouter(d_model=d_model, num_experts=num_experts, top_k=top_k).to(device)
    print(f"Created MoERouter with {num_experts} experts, top-k={top_k}")
    
    # Create sample input
    batch_size, seq_len = 4, 128
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    print(f"Input shape: {x.shape}")
    
    # Reference implementation using PyTorch for comparison
    def router_pytorch(x, router_weights):
        # [batch_size, seq_len, d_model] @ [num_experts, d_model].T -> [batch_size, seq_len, num_experts]
        return torch.matmul(x, router_weights.transpose(0, 1))
        
    # Time Triton kernel
    start_time = time.time()
    indices, gates, logits = router(x)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time
    
    # Time PyTorch reference
    start_time = time.time()
    ref_logits = router_pytorch(x, router.router_weights)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    # Verify shapes
    print(f"Logits shape: {logits.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Gates shape: {gates.shape}")
    
    # Verify that top-k indices and gates make sense
    top_k_ref_values, top_k_ref_indices = torch.topk(ref_logits, k=top_k, dim=-1)
    
    # Verify sum of gates is close to 1
    gates_sum = gates.sum(dim=-1)
    print(f"Average sum of gates: {gates_sum.mean().item()}")
    
    # Check if indices match between PyTorch and Triton
    match_count = 0
    total_count = batch_size * seq_len * top_k
    for b in range(batch_size):
        for s in range(seq_len):
            triton_indices = set(indices[b, s].tolist())
            pytorch_indices = set(top_k_ref_indices[b, s].tolist())
            match_count += len(triton_indices.intersection(pytorch_indices))
    
    match_percent = 100 * match_count / total_count
    print(f"Index match percentage: {match_percent:.2f}%")
    
    # Calculate error between logits
    error = torch.abs(logits - ref_logits).mean().item()
    print(f"Mean absolute error between logits: {error:.6f}")
    
    # Compare performance
    print(f"Triton time: {triton_time*1000:.2f} ms")
    print(f"PyTorch time: {pytorch_time*1000:.2f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")
    
    return True

if __name__ == "__main__":
    test_result = test_router()
    print(f"Test {'passed' if test_result else 'failed'}") 