# TinyLlama-1.1B From Scratch

A from-scratch PyTorch reimplementation of TinyLlama-1.1B's architecture, with real
pretrained weights loaded in, verified against HuggingFace's reference implementation,
and profiled to understand where the compute actually goes.

This project was built to understand transformer internals at the implementation
level — not just use `AutoModelForCausalLM`, but build every component (RMSNorm,
SwiGLU MLP, RoPE, Grouped Query Attention) by hand, load real weights into it, and
prove numerically that it behaves the same as the original.

## What's implemented

- **RMSNorm** — pre-normalization, matching HF's `LlamaRMSNorm` computation exactly
  (`x * rsqrt(mean(x^2) + eps)`).
- **SwiGLU MLP** — `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- **RoPE (Rotary Position Embeddings)** — implemented via the half-split rotation
  trick, verified against HF's `rotate_half` convention.
- **Grouped Query Attention (GQA)** — TinyLlama's real config: 32 query heads,
  4 KV heads, 8 queries sharing each KV pair, implemented with batched reshape +
  `repeat_interleave` (no explicit loop over heads).
- **Full model stack** — embedding → 22 pre-norm transformer blocks → final norm →
  LM head, matching TinyLlama-1.1B's real dimensions (`vocab_size=32000`,
  `hidden_size=2048`, `num_hidden_layers=22`, `intermediate_size=5632`).

## Weight loading

`load_pretrained_weights()` copies TinyLlama-1.1B-Chat-v1.0's real pretrained weights
(via HuggingFace `transformers`) into the from-scratch model, using an explicit
key-name mapping and shape assertions on every tensor copy — no silent shape
mismatches.

| HF checkpoint key | This model |
|---|---|
| `model.embed_tokens.weight` | `model.embed.weight` |
| `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | `model.blocks[i].attn.W_{q,k,v,o}.weight` |
| `model.layers.{i}.mlp.{gate,up,down}_proj.weight` | `model.blocks[i].mlp.{gate,up,down}_proj.weight` |
| `model.layers.{i}.input_layernorm.weight` | `model.blocks[i].norm1.weight` |
| `model.layers.{i}.post_attention_layernorm.weight` | `model.blocks[i].norm2.weight` |
| `model.norm.weight` | `model.final_norm.weight` |
| `lm_head.weight` | `model.lm_head.weight` |

## Verification against HuggingFace

`verify_model.py` runs the same prompt through this model and the real HF model,
then compares outputs at multiple levels: raw logits, softmax probabilities, top-k
token rankings, per-layer hidden states, and cosine similarity.

**Result:** both models agree on the top predicted token, with near-identical top-5
token rankings and probabilities. Per-position cosine similarity on non-BOS tokens
is consistently ≥0.997, mostly ≥0.999.

**A specific numerical discrepancy was investigated, not just reported.** Layer-by-layer
diffing showed a large jump in absolute difference starting around block 2 that then
persisted through the remaining layers, concentrated almost entirely on the BOS
(beginning-of-sequence) token position. This traces to a known, documented phenomenon
in Llama-family models — the "attention sink," where the first token accumulates
disproportionately large activation values that models learn to rely on as a stable
attention target. This isn't a bug: at the BOS position, hidden-state magnitude is
~150–700x larger than at any other position, so a numerically-small relative
difference reads as a large absolute one. Excluding the BOS position, differences are
consistent with ordinary float32 accumulation across 22 layers of matrix
multiplication performed via different (but mathematically equivalent) kernel
implementations.

## Profiling

`profile_model.py` profiles the model at both the PyTorch-operator level and the
per-transformer-block level using `torch.profiler`.

**Finding:** ~91–96% of total GPU time is spent in `aten::mm` (plain matrix
multiplication), routed through PyTorch's tuned `cutlass`/cuBLAS GEMM kernels.
Softmax, masking, RMSNorm, and SiLU gating each account for under ~2–3% of total
time individually. At this workload size (single prompt, batch size 1, ~40 tokens),
the model is matmul-bound, not memory- or attention-bound — meaning hand-fused
attention or normalization kernels would have limited room to improve on
already well-optimized GEMM kernels here. KV caching would matter far more in a
different regime (multi-step autoregressive generation), which this project's
single-forward-pass profiling setup doesn't exercise.

Per-block timing also surfaced a real hardware effect worth noting: run-to-run
GPU timing varied by up to 3x depending on thermal/clock state (idle ramp-up vs.
sustained-load throttling on a laptop GPU), which is a larger source of variance
than any of the profiled operators individually — consistent with clock-throttling
behavior observed in earlier kernel-benchmarking work on the same hardware.

## Project structure

```
tinyllama_model.py   # architecture (RMSNorm, MLP, RoPE, GQA, TransformerBlock,
                      # TinyLlamaModel) + load_pretrained_weights()
verify_model.py       # loads real weights, compares outputs against HF's model
profile_model.py       # torch.profiler-based operator- and block-level profiling
```

## Environment

- WSL2 Ubuntu, RTX 5070 
- `conda` environment: `cuda_env`
- PyTorch (CUDA-enabled), `transformers`

## Running

```bash
conda activate cuda_env

python tinyllama_model.py   # sanity check: random-init forward pass, output shape
python verify_model.py       # loads real weights, compares against HF reference
python profile_model.py      # profiles compute breakdown
```
# test
# test2
