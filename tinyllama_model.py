import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    first_half = x[..., :half]
    second_half = x[..., half:]
    new_first_half = first_half * cos - second_half * sin
    new_second_half = first_half * sin + second_half * cos
    out = torch.cat([new_first_half, new_second_half], dim=-1)
    return out


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


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = (x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))
        output = rms * self.weight
        return output


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
                assert my_state[my_key].shape == hf_state[hf_key].shape, f"{my_key} {my_state[my_key].shape} vs {hf_key} {hf_state[hf_key].shape}"
                my_state[my_key].copy_(hf_state[hf_key])

    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers = 32000, 2048, 32, 4, 5632, 22

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = TinyLlamaModel(vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers)
    model = model.to(device)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype = torch.float32, attn_implementation = "eager")
    hf_model = hf_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_pretrained_weights(model, hf_model, num_layers)
    # hf_state = hf_model.state_dict()
    # for layer_idx in [1, 2, 3]:
    #     for suffix in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    #                 "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    #                 "input_layernorm.weight", "post_attention_layernorm.weight"]:
    #         hf_key = f"model.layers.{layer_idx}.{suffix}"
    #         t = hf_state[hf_key]
    #         print(f"layer {layer_idx} {suffix}: mean={t.mean().item():.6f} std={t.std().item():.6f}")
    #     print()
    # comparing_model(model, hf_model)
    # token_ids = torch.randint(0, vocab_size, (5,))
    # logits = model(token_ids)
    # print(logits.shape)  # should be (5, 32000)
    # hf_state = hf_model.state_dict()
    # my_state = model.state_dict()