import torch
import triton
import triton.language as tl

@triton.jit
def router_kernel(
    x_ptr,                          # [batch_size, seq_len, d_model]
    router_weights_ptr,             # [num_experts, d_model]
    logits_ptr,                     # [batch_size, seq_len, num_experts]
    batch_size,
    seq_len,
    d_model,
    num_experts: tl.constexpr,      # Make this a compile-time constant
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute which batch and sequence position this program is responsible for
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Return if we're out of bounds
    if batch_idx >= batch_size:
        return
    
    # Compute the memory offsets for this batch and sequence element
    x_offset = batch_idx * seq_len * d_model + seq_idx * d_model
    logits_offset = batch_idx * seq_len * num_experts + seq_idx * num_experts
    
    # Initialize accumulators for the logits
    logits = tl.zeros([num_experts], dtype=tl.float32)
    
    # Process the input in blocks
    for d_start in range(0, d_model, BLOCK_SIZE):
        # Create offsets and mask for this block
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = d_start + offsets < d_model
        
        # Load input features for this token
        x_block = tl.load(x_ptr + x_offset + d_start + offsets, mask=mask, other=0.0)
        
        # Process each expert
        for expert_idx in range(num_experts):
            # Load weight block for this expert
            weight_offset = expert_idx * d_model + d_start
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = d_start + offsets < d_model
            weight_block = tl.load(router_weights_ptr + weight_offset + offsets, mask=mask, other=0.0)
            
            # Compute partial dot product for this expert
            partial_dot = tl.sum(x_block * weight_block, axis=0)
            # Accumulate to logits for this expert
            logits[expert_idx] += partial_dot
    
    # Store the computed logits
    for expert_idx in range(num_experts):
        tl.store(logits_ptr + logits_offset + expert_idx, logits[expert_idx])

@triton.jit
def top_k_gating(
    logits_ptr,              # [batch_size, seq_len, num_experts]
    indices_ptr,             # [batch_size, seq_len, top_k]
    gates_ptr,               # [batch_size, seq_len, top_k]
    batch_size,
    seq_len, 
    num_experts,
    top_k,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute which batch and sequence position this program is responsible for
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Return if we're out of bounds
    if batch_idx >= batch_size:
        return
    
    # Compute memory offsets
    logits_offset = batch_idx * seq_len * num_experts + seq_idx * num_experts
    indices_offset = batch_idx * seq_len * top_k + seq_idx * top_k
    gates_offset = batch_idx * seq_len * top_k + seq_idx * top_k
    
    # Load the logits for this token
    logits = tl.zeros([num_experts], dtype=tl.float32)
    for i in range(num_experts):
        logits[i] = tl.load(logits_ptr + logits_offset + i)
    
    # Find top-k indices (simple approach - can be optimized further)
    top_k_values = tl.zeros([top_k], dtype=tl.float32) - float('inf')
    top_k_indices = tl.zeros([top_k], dtype=tl.int32)
    
    for i in range(num_experts):
        val = logits[i]
        # Find insertion position
        for k in range(top_k):
            # If current value is larger than what we have, insert it
            mask = val > top_k_values[k]
            if mask:
                # Shift everything down
                for j in range(top_k-1, k, -1):
                    top_k_values[j] = top_k_values[j-1]
                    top_k_indices[j] = top_k_indices[j-1]
                # Insert new value
                top_k_values[k] = val
                top_k_indices[k] = i
                break
    
    # Apply softmax to just the top-k values
    max_val = tl.max(top_k_values)
    exp_vals = tl.exp(top_k_values - max_val)
    sum_exp = tl.sum(exp_vals)
    softmax_vals = exp_vals / sum_exp
    
    # Store the top-k indices and gates
    for k in range(top_k):
        tl.store(indices_ptr + indices_offset + k, top_k_indices[k])
        tl.store(gates_ptr + gates_offset + k, softmax_vals[k])

class MoERouter(torch.nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Initialize router weights
        self.router_weights = torch.nn.Parameter(torch.empty(num_experts, d_model))
        torch.nn.init.normal_(self.router_weights, mean=0.0, std=0.02)
        
        # Constants for the Triton kernels
        self.BLOCK_SIZE = 128  # Can be tuned for performance
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # Allocate memory for the outputs
        logits = torch.empty((batch_size, seq_len, self.num_experts), 
                            device=x.device, dtype=x.dtype)
        indices = torch.empty((batch_size, seq_len, self.top_k), 
                             device=x.device, dtype=torch.int32)
        gates = torch.empty((batch_size, seq_len, self.top_k), 
                           device=x.device, dtype=x.dtype)
        
        # Launch the router kernel
        grid = lambda meta: (batch_size * seq_len,)
        router_kernel[grid](
            x_ptr=x.contiguous().data_ptr(),
            router_weights_ptr=self.router_weights.contiguous().data_ptr(),
            logits_ptr=logits.contiguous().data_ptr(),
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            num_experts=self.num_experts,
            BLOCK_SIZE=self.BLOCK_SIZE
        )
        
        # Launch the top-k gating kernel
        top_k_gating[grid](
            logits_ptr=logits.contiguous().data_ptr(),
            indices_ptr=indices.contiguous().data_ptr(),
            gates_ptr=gates.contiguous().data_ptr(),
            batch_size=batch_size,
            seq_len=seq_len,
            num_experts=self.num_experts,
            top_k=self.top_k,
            BLOCK_SIZE=self.BLOCK_SIZE
        )
        
        return indices, gates, logits