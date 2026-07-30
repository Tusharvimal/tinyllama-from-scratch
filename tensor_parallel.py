import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    first_half = x[..., :half]
    second_half = x[..., half:]
    new_first_half = first_half * cos - second_half * sin
    new_second_half = first_half * sin + second_half * cos
    out = torch.cat([new_first_half, new_second_half], dim=-1)
    return out

class GroupedQueryAttentionSplit(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads,world_size, rank):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.group_size = num_q_heads // num_kv_heads
        self.num_q_heads_local = num_q_heads // world_size
        self.num_kv_heads_local = num_kv_heads // world_size
        self.rank = rank

        self.W_q = nn.Linear(d_model, self.num_q_heads_local * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.num_kv_heads_local * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.num_kv_heads_local * self.head_dim, bias=False)
        self.W_o = nn.Linear(self.num_q_heads_local * self.head_dim, d_model, bias=False)

        i = torch.arange(0, self.head_dim, 2)
        # self.freqs = 1.0 / (10000 ** (i / self.head_dim))
        self.register_buffer("freqs", 1.0 / (10000 ** (i / self.head_dim)), persistent=False)

    def forward(self, x):
        seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self. W_k(x)
        V = self.W_v(x)

        Q = Q.view(seq_len, self.num_q_heads_local, self.head_dim).transpose(0,1)
        K = K.view(seq_len, self.num_kv_heads_local, self.head_dim).transpose(0,1)
        V = V.view(seq_len, self.num_kv_heads_local, self.head_dim).transpose(0,1)

        K = torch.repeat_interleave(K, self.group_size, dim = 0)
        V = torch.repeat_interleave(V, self.group_size, dim = 0)

        positions = torch.arange(seq_len, device= x.device)
        angles = positions[:,None] * self.freqs[None,:]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        mask = torch.triu(torch.ones(seq_len, seq_len, device = x.device),
                          diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim = -1)
        output = attn_weights @ V
        output = output.transpose(0, 1).reshape(seq_len, self.num_q_heads_local * self.head_dim)
        output = self.W_o(output)
        dist.all_reduce(output, op = dist.ReduceOp.SUM)
        return output

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.group_size = num_q_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False)

        i = torch.arange(0, self.head_dim, 2)
        # self.freqs = 1.0 / (10000 ** (i / self.head_dim))
        self.register_buffer("freqs", 1.0 / (10000 ** (i / self.head_dim)), persistent=False)

    def forward(self, x):
        seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(seq_len, self.num_q_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)

        K = torch.repeat_interleave(K, self.group_size, dim=0)
        V = torch.repeat_interleave(V, self.group_size, dim=0)

        positions = torch.arange(seq_len, device=x.device)
        angles = positions[:, None] * self.freqs[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        mask = torch.triu(torch.ones(seq_len, seq_len, device = x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ V
        output = output.transpose(0, 1).reshape(seq_len, self.num_q_heads * self.head_dim)
        output = self.W_o(output)
        return output

class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        fused = F.silu(gate) * up
        output = self.down_proj(fused)
        return output


class MLPSplit(nn.Module):
    def __init__(self, d_model, d_ff, world_size, rank):
        super().__init__()
        self.d_ff_local = d_ff // world_size
        self.rank = rank
        self.gate_proj = nn.Linear(d_model, self.d_ff_local, bias=False)
        self.up_proj = nn.Linear(d_model, self.d_ff_local, bias=False)
        self.down_proj = nn.Linear(self.d_ff_local, d_model, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        fused = F.silu(gate) * up
        output = self.down_proj(fused)
        dist.all_reduce(output, op = dist.ReduceOp.SUM)
        return output


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = (x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))
        output = rms * self.weight
        return output


class TransformerBlockSplit(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads, d_ff, world_size, rank):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttentionSplit(d_model, num_q_heads, num_kv_heads, world_size, rank)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLPSplit(d_model, d_ff, world_size, rank)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TinyLlamaModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_q_heads, num_kv_heads, d_ff) for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

class TinyLlamaModelSplit(nn.Module):
    def __init__(self, vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers, world_size, rank):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlockSplit(d_model, num_q_heads, num_kv_heads, d_ff, world_size, rank) for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

def load_pretrained_weights(model, hf_model, num_layers):
    hf_state = hf_model.state_dict()
    my_state = model.state_dict()

    with torch.no_grad():
        my_state["embed.weight"].copy_(hf_state["model.embed_tokens.weight"])
        my_state["final_norm.weight"].copy_(hf_state["model.norm.weight"])
        my_state["lm_head.weight"].copy_(hf_state["lm_head.weight"])

        for i in range(num_layers):
            pairs = [
                (f"blocks.{i}.attn.W_q.weight", f"model.layers.{i}.self_attn.q_proj.weight"),
                (f"blocks.{i}.attn.W_k.weight", f"model.layers.{i}.self_attn.k_proj.weight"),
                (f"blocks.{i}.attn.W_v.weight", f"model.layers.{i}.self_attn.v_proj.weight"),
                (f"blocks.{i}.attn.W_o.weight", f"model.layers.{i}.self_attn.o_proj.weight"),
                (f"blocks.{i}.mlp.gate_proj.weight", f"model.layers.{i}.mlp.gate_proj.weight"),
                (f"blocks.{i}.mlp.up_proj.weight", f"model.layers.{i}.mlp.up_proj.weight"),
                (f"blocks.{i}.mlp.down_proj.weight", f"model.layers.{i}.mlp.down_proj.weight"),
                (f"blocks.{i}.norm1.weight", f"model.layers.{i}.input_layernorm.weight"),
                (f"blocks.{i}.norm2.weight", f"model.layers.{i}.post_attention_layernorm.weight"),
            ]
            for my_key, hf_key in pairs:
                my_state[my_key].copy_(hf_state[hf_key])

    return model
    
def load_pretrained_weights_split(model, hf_model, num_layers):
    hf_state = hf_model.state_dict()
    my_state = model.state_dict()

    with torch.no_grad():
        my_state["embed.weight"].copy_(hf_state["model.embed_tokens.weight"])
        my_state["final_norm.weight"].copy_(hf_state["model.norm.weight"])
        my_state["lm_head.weight"].copy_(hf_state["lm_head.weight"])

        for i in range(num_layers):
            attn = model.blocks[i].attn
            mlp = model.blocks[i].mlp
            rank = attn.rank
            head_dim = attn.head_dim
            nq_local = attn.num_q_heads_local
            nkv_local = attn.num_kv_heads_local
            d_ff_local = mlp.d_ff_local
            pairs = [
                # (f"blocks.{i}.attn.W_q.weight", f"model.layers.{i}.self_attn.q_proj.weight"),
                # (f"blocks.{i}.attn.W_k.weight", f"model.layers.{i}.self_attn.k_proj.weight"),
                # (f"blocks.{i}.attn.W_v.weight", f"model.layers.{i}.self_attn.v_proj.weight"),
                # (f"blocks.{i}.attn.W_o.weight", f"model.layers.{i}.self_attn.o_proj.weight"),
                # (f"blocks.{i}.mlp.gate_proj.weight", f"model.layers.{i}.mlp.gate_proj.weight"),
                # (f"blocks.{i}.mlp.up_proj.weight", f"model.layers.{i}.mlp.up_proj.weight"),
                # (f"blocks.{i}.mlp.down_proj.weight", f"model.layers.{i}.mlp.down_proj.weight"),
                (f"blocks.{i}.norm1.weight", f"model.layers.{i}.input_layernorm.weight"),
                (f"blocks.{i}.norm2.weight", f"model.layers.{i}.post_attention_layernorm.weight"),
            ]
            for my_key, hf_key in pairs:
                # assert my_state[my_key].shape == hf_state[hf_key].shape, f"{my_key} {my_state[my_key].shape} vs {hf_key} {hf_state[hf_key].shape}"
                my_state[my_key].copy_(hf_state[hf_key])

            start_q = rank * nq_local * head_dim 
            end_q = (rank + 1) * nq_local * head_dim
            my_state[f"blocks.{i}.attn.W_q.weight"].copy_(
                hf_state[f"model.layers.{i}.self_attn.q_proj.weight"][start_q:end_q, :]
            )

            start_kv = rank * nkv_local * head_dim
            end_kv = (rank + 1) * nkv_local * head_dim
            my_state[f"blocks.{i}.attn.W_k.weight"].copy_(
                hf_state[f"model.layers.{i}.self_attn.k_proj.weight"][start_kv:end_kv, :]
            )
            my_state[f"blocks.{i}.attn.W_v.weight"].copy_(
                hf_state[f"model.layers.{i}.self_attn.v_proj.weight"][start_kv:end_kv, :]
            )
            my_state[f"blocks.{i}.attn.W_o.weight"].copy_(
                hf_state[f"model.layers.{i}.self_attn.o_proj.weight"][:, start_q:end_q]
            )

            start_ff = rank * d_ff_local
            end_ff = (rank + 1) * d_ff_local 
            my_state[f"blocks.{i}.mlp.gate_proj.weight"].copy_(
                        hf_state[f"model.layers.{i}.mlp.gate_proj.weight"][start_ff:end_ff,:])
            my_state[f"blocks.{i}.mlp.up_proj.weight"].copy_(
                        hf_state[f"model.layers.{i}.mlp.up_proj.weight"][start_ff:end_ff,:])
            my_state[f"blocks.{i}.mlp.down_proj.weight"].copy_(
                        hf_state[f"model.layers.{i}.mlp.down_proj.weight"][:, start_ff:end_ff])

    return model

import time

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{rank}")

    vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers = 32000, 2048, 32, 4, 5632, 22

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, attn_implementation="eager")
    hf_model = hf_model.to(device)

    # Build the SPLIT model for this rank
    split_model = TinyLlamaModelSplit(vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers, world_size, rank)
    split_model = split_model.to(device)
    load_pretrained_weights_split(split_model, hf_model, num_layers)

    # Same input on every rank
    seq_len = 1000
    torch.manual_seed(42)
    token_ids = torch.randint(0, vocab_size, (seq_len,), device=device)

    # Warm-up pass (first CUDA call always has extra overhead, don't time it)
    with torch.no_grad():
        _ = split_model(token_ids)
    torch.cuda.synchronize()

    # Timed split model forward pass
    start = time.time()
    with torch.no_grad():
        split_logits = split_model(token_ids)
    torch.cuda.synchronize()
    split_elapsed = time.time() - start

    if rank == 0:
        print(f"Split model forward pass: {split_elapsed*1000:.2f} ms")

        # Build the ORIGINAL (unsplit) model only on rank 0 for comparison
        original_model = TinyLlamaModel(vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers)
        original_model = original_model.to(device)
        load_pretrained_weights(original_model, hf_model, num_layers)

        # Warm-up pass for original too
        with torch.no_grad():
            _ = original_model(token_ids)
        torch.cuda.synchronize()

        # Timed original model forward pass
        start = time.time()
        with torch.no_grad():
            original_logits = original_model(token_ids)
        torch.cuda.synchronize()
        original_elapsed = time.time() - start
        print(f"Original model forward pass: {original_elapsed*1000:.2f} ms")

        # Correctness check
        cos_sim = F.cosine_similarity(split_logits.flatten(), original_logits.flatten(), dim=0)
        max_diff = (split_logits - original_logits).abs().max().item()
        print(f"Cosine similarity: {cos_sim.item():.6f}")
        print(f"Max abs diff: {max_diff:.8f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()