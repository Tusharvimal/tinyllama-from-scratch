import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinyllama_model import TinyLlamaModel, load_pretrained_weights

def comparing_model(model, hf_model):
    model.eval()
    hf_model.eval()

    # prompt = "The capital of France is"
    prompt = "The Eiffel Tower is located in Paris, France, and it was completed in the year 1889 during the World's Fair. It stands at a height of"
    inputs = tokenizer(prompt, return_tensors = "pt")
    input_ids = inputs["input_ids"].to(device)
    # print(inputs)
    # print(inputs["input_ids"].shape)
    print("num tokens:", input_ids.shape)

    with torch.no_grad():
        my_logits = model(input_ids.squeeze(0))
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits

    print("my logits shape:", my_logits.shape)
    print("hf_logits shape:", hf_logits.shape)

    hf_logits_squeezed = hf_logits.squeeze(0)
    print("hf_logits squeezed shape:", hf_logits_squeezed.shape)

    my_last = my_logits[-1]
    hf_last = hf_logits_squeezed[-1]

    print("allclose (atol = 1e-3):", torch.allclose(my_last, hf_last, atol = 1e-3))
    print("max abs diff:", (my_last - hf_last).abs().max().item())

    my_top_token = torch.argmax(my_last)
    hf_top_token = torch.argmax(hf_last)

    print("my top token:", tokenizer.decode(my_top_token))
    print("hf top token:", tokenizer.decode(hf_top_token))

    my_probs = torch.softmax(my_last, dim=-1)
    hf_probs = torch.softmax(hf_last, dim=-1)
    print("prob allclose (atol=1e-3):", torch.allclose(my_probs, hf_probs, atol=1e-3))
    print("max prob diff:", (my_probs - hf_probs).abs().max().item())

    top5_my = torch.topk(my_probs, 5)
    top5_hf = torch.topk(hf_probs, 5)
    print("my top5:", [(tokenizer.decode(i), p.item()) for i, p in zip(top5_my.indices, top5_my.values)])
    print("hf top5:", [(tokenizer.decode(i), p.item()) for i, p in zip(top5_hf.indices, top5_hf.values)])

    my_last_no_bos = my_logits[1:]
    hf_last_no_bos = hf_logits_squeezed[1:]
    print("allclose excluding position 0:", torch.allclose(my_last_no_bos, hf_last_no_bos, atol=1e-3))

    diff_no_bos = (my_logits[1:] - hf_logits_squeezed[1:]).abs()
    print("max abs diff (excl. BOS):", diff_no_bos.max().item())
    print("mean abs diff (excl. BOS):", diff_no_bos.mean().item())
    print("allclose atol=0.1:", torch.allclose(my_logits[1:], hf_logits_squeezed[1:], atol=0.1))

    cos_sim = torch.nn.functional.cosine_similarity(my_logits[1:], hf_logits_squeezed[1:], dim=-1)
    print("per-position cosine similarity:", cos_sim)

    with torch.no_grad():
        hf_output_full = hf_model(input_ids, output_hidden_states=True)
        hf_hidden_states = hf_output_full.hidden_states  # tuple: (embed_output, after_layer_0, after_layer_1, ..., after_layer_21)

        # manually replicate your forward pass, capturing intermediate outputs
        x = model.embed(input_ids.squeeze(0))
        print("embed diff:", (x - hf_hidden_states[0].squeeze(0)).abs().max().item())

        for i, block in enumerate(model.blocks):
            x = block(x)
            hf_x = hf_hidden_states[i + 1].squeeze(0)
            diff = (x - hf_x).abs().max().item()
            print(f"after block {i}: max abs diff = {diff:.6f}")

    with torch.no_grad():
        hf_output_full = hf_model(input_ids, output_hidden_states=True)
        hf_hidden_states = hf_output_full.hidden_states

        x = model.embed(input_ids.squeeze(0))
        for i, block in enumerate(model.blocks):
            x = block(x)
            if i == 2:
                hf_x = hf_hidden_states[i + 1].squeeze(0)
                diff_per_token = (x - hf_x).abs().max(dim=-1).values
                my_norm_per_token = x.abs().max(dim=-1).values
                print("diff per token position:", diff_per_token)
                print("my hidden state magnitude per token:", my_norm_per_token)
                break

if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers = 32000, 2048, 32, 4, 5632, 22

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = TinyLlamaModel(vocab_size, d_model, num_q_heads, num_kv_heads, d_ff, num_layers)
    model = model.to(device)

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, attn_implementation="eager")
    hf_model = hf_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_pretrained_weights(model, hf_model, num_layers)
    comparing_model(model, hf_model)