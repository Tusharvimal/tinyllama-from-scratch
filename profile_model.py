import torch
from torch.profiler import profile, ProfilerActivity, record_function
from tinyllama_model import TinyLlamaModel, load_pretrained_weights
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers = 32000, 2048, 32, 4, 5632, 22

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = TinyLlamaModel(vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers).to(device)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype = torch.float32, attn_implementation = "eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

load_pretrained_weights(model, hf_model, num_layers)
model.eval()

prompt = "The Eiffel Tower is located in Paris, France, and it was completed in the year 1889 during the World's Fair. It stands at a height of"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device).squeeze(0)

# more aggressive warm-up — run several times, not just once
with torch.no_grad():
    for _ in range(5):
        _ = model(input_ids)
        torch.cuda.synchronize()

with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with torch.no_grad():
        with record_function("embedding"):
            x = model.embed(input_ids)
        torch.cuda.synchronize()
        for i, block in enumerate(model.blocks):
            with record_function(f"block_{i}"):
                x = block(x)
            torch.cuda.synchronize()
        with record_function("final_norm_and_head"):
            x = model.final_norm(x)
            logits = model.lm_head(x)
            torch.cuda.synchronize()

print("\n=== Module-level profiling===")
print(prof.key_averages().table(sort_by = "cuda_time_total", row_limit = 30))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes = True) as prof2:
    with torch.no_grad():
        _ = model(input_ids)

print("Operator level profiling")
print(prof2.key_averages().table(sort_by = 'cuda_time_total', row_limit = 20))
