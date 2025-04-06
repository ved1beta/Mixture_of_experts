import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # drop CUDA if no GPU
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("my_model_run"):
        x = torch.randn(1024, 1024).cuda()
        y = torch.matmul(x, x)

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
