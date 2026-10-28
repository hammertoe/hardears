# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.1",
#   "transformers==5.20.2",
#   "peft==0.21.2",
#   "accelerate==1.15.0",
#   "huggingface-hub==2.3.0",
#   "safetensors==0.8.0",
#   "datasets==5.1.0",
# ]
#
# [tool.hf-jobs]
# image   = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
# flavor  = "a10g-small"
# secrets = ["HF_TOKEN"]
# timeout = "30m"
# ///
"""Step 0: memory + environment smoke test for text-only DAPT on LFM3-Audio-2B.

Loads the model + LoRA adapter on a packed text batch, runs one forward+backward,
and prints peak GPU memory, trainable param count, and a token throughput estimate.
Decides the flavor for all later jobs.
"""
import os, time, json, torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
REVISION = None  # pin in real runs; None = latest

def main():
    print("=== LFM3-Audio-2B text-only DAPT :: smoke test ===")
    print("torch", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("vram total GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9,2))

    t0 = time.time()
    print("loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, revision=REVISION)
    print("loading model...")
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, revision=REVISION
    )
    print("loaded in", round(time.time()-t0,1), "s")
    print("params:", model.num_parameters())

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable/total = {trainable}/{total} = {100*trainable/total:.4f}%")

    # verify adapter attached only to language_model
    bad = [n for n,_ in model.named_parameters() if _.requires_grad and "language_model" not in n]
    print("trainable params NOT under language_model:", bad[:10], "count=", len(bad))
    assert len(bad)==0, "LoRA attached outside language_model!"

    model.train()
    # build a packed text batch (no audio)
    text = "Barbados is an island nation in the Caribbean. Bridgetown is its capital. " * 64
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to("cuda")
    # pad to 2048
    if ids.shape[1] < 2048:
        pad = torch.full((1, 2048-ids.shape[1]), tok.pad_token_id, dtype=ids.dtype, device=ids.device)
        ids = torch.cat([ids, pad], dim=1)
    labels = ids.clone()
    labels[ids == tok.pad_token_id] = -100
    attn = (ids != tok.pad_token_id).long()

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    out = model(input_ids=ids, attention_mask=attn, labels=labels)
    loss = out.loss
    print("forward loss:", float(loss))
    loss.backward()
    dt = time.time()-t1
    toks = int(attn.sum().item())
    print(f"forward+backward: {dt:.2f}s for {toks} tokens => {toks/dt:.1f} tok/s")
    print("peak vram GB:", round(torch.cuda.max_memory_allocated()/1e9,3))
    print("peak reserved GB:", round(torch.cuda.max_memory_reserved()/1e9,3))
    # gradient sanity
    gnorms = []
    for n,p in model.named_parameters():
        if p.grad is not None:
            gnorms.append((n, float(p.grad.norm())))
    print("num params with grad:", len(gnorms))
    if gnorms:
        print("max grad norm:", max(g[1] for g in gnorms), "min:", min(g[1] for g in gnorms))
    # save a tiny result json
    result = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_vram_gb": round(torch.cuda.max_memory_allocated()/1e9,3),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100*trainable/total,4),
        "loss": float(loss),
        "tok_per_s": round(toks/dt,1),
        "grad_params": len(gnorms),
    }
    os.makedirs("/tmp/scratchpad", exist_ok=True)
    with open("/tmp/scratchpad/smoke_result.json","w") as f:
        json.dump(result, f, indent=2)
    print("RESULT_JSON:", json.dumps(result))
    print("SMOKE_OK")

if __name__ == "__main__":
    main()
