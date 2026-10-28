# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.1",
#   "transformers==5.20.2",
#   "peft==0.21.2",
#   "accelerate==1.15.0",
#   "huggingface-hub==2.3.0",
#   "safetensors==0.8.0",
#   "soundfile==0.12.1",
#   "numpy",
# ]
# ///
"""Audio QA SFT: teach LFM3-Audio-2B to answer Barbados knowledge questions from audio.

Generates TTS audio of each knowledge probe prompt, SFTs on audio->answer,
and evaluates audio-understanding accuracy before/after on a held-out split.
"""
import os, time, json, random, glob
import numpy as np
import torch, torch.nn.functional as F
import soundfile as sf
from transformers import AutoModelForMultimodalLM, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
PROBES_PATH = "/inputs/probes/barbados-knowledge-v1.jsonl"
OUT_DIR = "/tmp/scratchpad/audioqa_out"
ADAPTER_REPO = "hammertoe/lfm3-audio-barbados-audioqa"
MAX_STEPS = 300
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"tts"), exist_ok=True)

def load_model(adapter_path=None):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    codec = proc.audio_tokenizer.to(device="cuda").eval()
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok, proc, codec

def tts(model, proc, codec, text, out_path):
    """Generate TTS audio for text, save 16kHz wav. Returns path."""
    if os.path.exists(out_path):
        return out_path
    conversation = [
        {"role": "system", "content": "Perform TTS in American English using a female voice."},
        {"role": "user", "content": text},
        {"role": "assistant", "content": "<|audio_start|>"},
    ]
    inputs = proc.apply_chat_template(
        conversation, continue_final_message=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=256,
            audio_temperature=1.0, audio_top_k=128,
            audio_repetition_penalty=1.1, audio_repetition_window=4096,
            return_dict_in_generate=True,
        )
        wav = codec.decode(out.audio_codes[:1, :, :out.audio_lengths[0]]).audio_values[0]
    wav = wav.float().cpu().numpy()
    # resample 24k -> 16k
    import numpy as np
    src_sr=24000; dst_sr=16000
    n = int(len(wav)*dst_sr/src_sr)
    wav16 = np.interp(np.linspace(0,len(wav)-1,n), np.arange(len(wav)), wav).astype("float32")
    sf.write(out_path, wav16, dst_sr)
    return out_path

def eval_audio_qa(model, proc, items):
    """items: list of {audio_path, answer}. Generate answer from audio, check exact match (normalized)."""
    correct=0; total=0; details=[]
    with torch.inference_mode():
        for it in items:
            wav,_ = sf.read(it["audio_path"], dtype="float32")
            conversation = [
                {"role": "system", "content": "Answer the question about Barbados. Reply with only the answer, no explanation."},
                {"role": "user", "content": [{"type":"audio","audio":wav}]},
            ]
            inputs = proc.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to("cuda")
            out = model.generate(**inputs, max_new_tokens=64, suppress_mixed=True, suppress_audio_out=True, return_dict_in_generate=True)
            pred = proc.decode(out.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            gold = it["answer"].strip()
            ok = pred.lower() == gold.lower()
            if ok: correct+=1
            total+=1
            details.append({"pred":pred,"gold":gold,"ok":ok})
    return {"correct":correct,"total":total,"acc":correct/total if total else 0.0}, details

def main():
    t0=time.time()
    probes=[json.loads(l) for l in open(PROBES_PATH) if l.strip()]
    # build items: TTS each prompt
    print("=== TTS GENERATION ===")
    model, tok, proc, codec = load_model()
    items=[]
    for i,p in enumerate(probes):
        ap = os.path.join(OUT_DIR,"tts", f"q{i:03d}.wav")
        tts(model, proc, codec, p["prompt"], ap)
        items.append({"idx":i, "prompt":p["prompt"], "answer": p["choices"][p["answer_index"]],
                      "audio_path": ap, "track": p["track"]})
        if i%10==0: print("tts", i, "/", len(probes))
    # save items manifest
    with open(os.path.join(OUT_DIR,"items.json"),"w") as f: json.dump(items, f, indent=2)
    # split train/test 80/20
    random.seed(42); random.shuffle(items)
    n_train = int(0.8*len(items))
    train_items = items[:n_train]; test_items = items[n_train:]
    print("train/test:", len(train_items), len(test_items))

    # baseline eval on ALL items (audio)
    print("\n=== BASELINE (audio QA, all items) ===")
    base_scores, base_details = eval_audio_qa(model, proc, items)
    print("BASE:", json.dumps(base_scores))
    with open(os.path.join(OUT_DIR,"base_eval.json"),"w") as f:
        json.dump({"scores":base_scores,"details":base_details}, f, indent=2)

    # free base model
    del model; torch.cuda.empty_cache()

    # ---- SFT TRAINING ----
    print("\n=== TRAINING ===")
    model, tok, proc, codec = load_model()
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)",
    )
    model = get_peft_model(model, peft_config)
    model.train()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
    # build training rows manually
    # collate: processor.apply_chat_template on conversation with audio
    def make_batch(rows):
        convs = []
        for r in rows:
            wav,_ = sf.read(r["audio_path"], dtype="float32")
            convs.append([
                {"role":"system","content":"Answer the question about Barbados. Reply with only the answer, no explanation."},
                {"role":"user","content":[{"type":"audio","audio":wav}]},
                {"role":"assistant","content": r["answer"]},
            ])
        batch = proc.apply_chat_template(convs, tokenize=True, return_dict=True, return_tensors="pt", processor_kwargs={"padding":True})
        labels = torch.full_like(batch["input_ids"], -100)
        # mask everything up to assistant answer: find assistant header
        # simple: mask prompt by locating answer tokens? Use attention_mask padding only + a heuristic:
        # We'll mask all tokens where input_ids == pad, and mask the first (seq_len - answer_len) approx.
        # Better: recompute per row answer start via tokenizing answer
        for i,row in enumerate(rows):
            ans_ids = tok(row["answer"], add_special_tokens=False).input_ids
            ids = batch["input_ids"][i].tolist()
            # find last occurrence of answer token sequence
            n=len(ans_ids)
            start=None
            for j in range(len(ids)-n, -1, -1):
                if ids[j:j+n]==ans_ids:
                    start=j; break
            if start is not None:
                labels[i, start:] = batch["input_ids"][i, start:]
            else:
                # fallback: mask padding only
                pass
        labels[batch["attention_mask"]==0] = -100
        batch["labels"]=labels
        return batch

    step=0; t_train=time.time(); losses=[]; grad_norms=[]
    while step < MAX_STEPS:
        # sample a mini-batch
        batch_rows = [train_items[(step*k)%len(train_items)] for k in range(1)]  # batch 1 for memory
        # actually use 4 rows if memory allows; keep 1 for safety
        b = make_batch(batch_rows)
        b = {k:v.to("cuda") for k,v in b.items()}
        out = model(**{k:v for k,v in b.items() if k!="labels"}, labels=b["labels"])
        loss = out.loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        grad_norms.append(float(gn))
        opt.step(); opt.zero_grad()
        losses.append(float(out.loss.item()))
        step+=1
        if step%25==0 or step==1:
            print(f"step {step}/{MAX_STEPS} loss={out.loss.item():.4f} gn={gn:.3f} elapsed={time.time()-t_train:.0f}s")
        if step%100==0:
            model.save_pretrained(os.path.join(OUT_DIR, f"ckpt_step{step}"))
            print("saved ckpt", step)
    final_ckpt = os.path.join(OUT_DIR, "adapter_final")
    model.save_pretrained(final_ckpt)
    proc.save_pretrained(final_ckpt); tok.save_pretrained(final_ckpt)
    print("FINAL ADAPTER:", final_ckpt)
    with open(os.path.join(OUT_DIR,"train_stats.json"),"w") as f:
        json.dump({"steps":step,"losses":losses,"grad_norms":grad_norms,
                   "mean_first10":float(np.mean(losses[:10])),"mean_last10":float(np.mean(losses[-10:]))}, f)
    del model; torch.cuda.empty_cache()

    # ---- POST EVAL ----
    print("\n=== POST EVAL (audio QA, test items) ===")
    model, tok, proc, codec = load_model(adapter_path=final_ckpt)
    # adapter applied check
    base_model,_,_,_ = load_model()
    test_ids = tok("Barbados is an island.", return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        bo = base_model(input_ids=test_ids).logits
        ao = model(input_ids=test_ids).logits
    print("adapter logit diff:", float((bo-ao).abs().max()))
    post_scores, post_details = eval_audio_qa(model, proc, test_items)
    print("POST test:", json.dumps(post_scores))
    base_test_scores, base_test_details = eval_audio_qa(base_model, proc, test_items)
    print("BASE test:", json.dumps(base_test_scores))
    result = {
        "n_train": len(train_items), "n_test": len(test_items),
        "base_test": base_test_scores, "post_test": post_scores,
        "base_all": base_scores,
        "adapter_logit_diff": float((bo-ao).abs().max()),
        "elapsed_min": round((time.time()-t0)/60,1),
    }
    with open(os.path.join(OUT_DIR,"results.json"),"w") as f: json.dump(result, f, indent=2)
    print("RESULTS:", json.dumps(result, indent=2))
    # upload
    try:
        model.push_to_hub(ADAPTER_REPO, private=True)
        proc.push_to_hub(ADAPTER_REPO, private=True)
        tok.push_to_hub(ADAPTER_REPO, private=True)
        print("UPLOADED to", ADAPTER_REPO)
    except Exception as e:
        print("UPLOAD FAILED:", e)

if __name__ == "__main__":
    main()
