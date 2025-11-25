import torch
import time
from src.l_fep.attention import LogicalAttention

def run_attention_benchmark():
    print("Starting Phase 4: Logical Attention Benchmark")
    
    embed_dim = 64
    num_heads = 4
    batch_size = 16
    
    model = LogicalAttention(embed_dim, num_heads)
    
    seq_lens = [100, 200, 400, 800, 1600]
    
    for seq_len in seq_lens:
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        start_time = time.time()
        for _ in range(10):
            _ = model(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"Seq Len: {seq_len}, Avg Time: {avg_time:.6f}s")
        
    print("Phase 4 Completed.")

if __name__ == '__main__':
    run_attention_benchmark()
