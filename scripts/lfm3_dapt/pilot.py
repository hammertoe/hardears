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
"""Pilot: text-only DAPT on LFM3-Audio-2B.

Baselines base model on (a) Barbados knowledge probes (multiple-choice) and
(b) ASR on 3 diagnostic Bajan clips, then trains a small backbone-only LoRA
on the Barbados text corpus, re-evaluates, and saves adapter + results.
"""
import os, time, json, glob, random, hashlib
import torch, torch.nn.functional as F
import numpy as np
import soundfile as sf
from transformers import AutoModelForMultimodalLM, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
MODEL_REV = None
PROBES_PATH = "/inputs/probes/barbados-knowledge-v1.jsonl"
SAMPLES_DIR = "/inputs/samples"
OUT_DIR = "/tmp/scratchpad/pilot_out"
ADAPTER_REPO = "hammertoe/lfm3-audio-barbados-dapt-pilot"
MAX_STEPS = 200
SEQ_LEN = 2048
BATCH = 1
GRAD_ACCUM = 4

os.makedirs(OUT_DIR, exist_ok=True)

def load_model(adapter_path=None):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, revision=MODEL_REV)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, revision=MODEL_REV
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok, proc

def score_knowledge(model, tok, probes):
    """Multiple-choice: for each probe, score NLL of each choice completion; pick argmin."""
    results = []
    correct = 0
    total = 0
    by_track = {}
    with torch.inference_mode():
        for p in probes:
            prompt = p["prompt"]
            choices = p["choices"]
            ans = p["answer_index"]
            # tokenize prompt once
            prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            nlls = []
            for c in choices:
                full = tok(prompt + " " + c, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
                # labels: -100 on prompt, choice tokens supervised
                labels = full.clone()
                labels[:, :prompt_ids.shape[1]] = -100
                attn = torch.ones_like(full)
                out = model(input_ids=full, attention_mask=attn, labels=labels)
                # out.loss is mean over supervised tokens (with -100 masking) for causal LM
                nlls.append(float(out.loss.item()) if out.loss is not None else float("inf"))
            pred = int(np.argmin(nlls))
            ok = (pred == ans)
            if ok: correct += 1
            total += 1
            track = p.get("track","?")
            by_track.setdefault(track, {"n":0,"correct":0})
            by_track[track]["n"] += 1
            if ok: by_track[track]["correct"] += 1
            results.append({"id":p["id"],"pred":pred,"gold":ans,"ok":ok,"nlls":nlls})
    summary = {"correct":correct,"total":total,"acc":correct/total if total else 0.0, "by_track":{k:{"n":v["n"],"acc":v["correct"]/v["n"] if v["n"] else 0.0} for k,v in by_track.items()}}
    return summary, results

def transcribe_clip(model, proc, wav_path):
    import soundfile as sf
    wav, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        import numpy as np
        # simple resample via linear interp
        dur = len(wav)/sr
        n = int(dur*16000)
        wav = np.interp(np.linspace(0,len(wav)-1,n), np.arange(len(wav)), wav).astype("float32")
    # write temp 16k wav? processor accepts path or array. Pass array via audio kwarg.
    conversation = [
        {"role": "system", "content": "Perform ASR in English."},
        {"role": "user", "content": [{"type": "audio", "audio": wav}]},
    ]
    inputs = proc.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=512, suppress_mixed=True, suppress_audio_out=True,
            return_dict_in_generate=True
        )
    text = proc.decode(out.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()

def main():
    t0 = time.time()
    print("=== PILOT START ===")
    # stage probes + samples already expected at /data
    probes = []
    if os.path.exists(PROBES_PATH):
        with open(PROBES_PATH) as f:
            for line in f:
                line=line.strip()
                if line:
                    probes.append(json.loads(line))
    print("probes:", len(probes))
    sample_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.wav")))
    print("samples:", sample_files)

    # ---- BASELINE ----
    print("\n=== BASELINE ===")
    model, tok, proc = load_model()
    base_knowledge, base_kr = score_knowledge(model, tok, probes)
    print("BASE knowledge:", json.dumps(base_knowledge))
    base_asr = {}
    for sp in sample_files:
        try:
            base_asr[os.path.basename(sp)] = transcribe_clip(model, proc, sp)
            print("BASE ASR", os.path.basename(sp), "=>", base_asr[os.path.basename(sp)][:200])
        except Exception as e:
            base_asr[os.path.basename(sp)] = f"ERROR: {e}"
            print("BASE ASR ERROR", os.path.basename(sp), e)
    # save baseline
    with open(os.path.join(OUT_DIR,"baseline.json"),"w") as f:
        json.dump({"knowledge":base_knowledge,"knowledge_results":base_kr,"asr":base_asr}, f, indent=2)

    # free
    del model; torch.cuda.empty_cache()

    # ---- DATA PREP ----
    print("\n=== DATA PREP ===")
    # gather corpus text from local bimomni? We're on HF job; data must be mounted.
    # Expect /inputs/corpus/*.jsonl (newspaper + wiki) OR a single /inputs/corpus.jsonl
    corpus_texts = []
    candidates = []
    if os.path.isdir("/inputs/corpus"):
        candidates = glob.glob("/inputs/corpus/*.jsonl")
    if os.path.exists("/inputs/corpus.jsonl"):
        candidates.append("/inputs/corpus.jsonl")
    # also try mounted bimomni? no.
    print("corpus files:", candidates)
    for cf in candidates:
        with open(cf) as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    d=json.loads(line)
                except Exception:
                    continue
                msgs = d.get("messages", [])
                # extract assistant content (article text)
                for m in msgs:
                    if m.get("role")=="assistant" and isinstance(m.get("content"),str):
                        t = m["content"].strip()
                        if t:
                            corpus_texts.append(t)
    print("raw records:", len(corpus_texts))

    # If no corpus mounted, fall back to wikipedia + a small embedded Barbados text
    if not corpus_texts:
        print("NO CORPUS MOUNTED - using embedded fallback text")
        fallback = [
            "Barbados is an island country in the Lesser Antilles of the West Indies, in the Caribbean region of the Americas. Its capital and largest city is Bridgetown.",
            "The Parliament of Barbados first convened in 1639, making it one of the oldest legislatures in the Americas.",
            "Crop Over is the major annual festival in Barbados, originating from the sugar cane harvest celebrations.",
            "Kensington Oval is the premier cricket ground in Barbados, located in Bridgetown.",
            "Oistins is a fishing town on the south coast of Barbados, famous for its Friday night fish fry.",
            "Chefette is a popular fast-food restaurant chain in Barbados.",
            "St Lawrence Gap is a nightlife and restaurant strip on the south coast of Barbados.",
            "Harrison's Cave is a crystallized limestone cave in the central uplands of Barbados.",
            "Animal Flower Cave is located at the northern tip of Barbados in St Lucy parish.",
            "The Barbados dollar is pegged to the US dollar at a rate of 2 to 1.",
        ] * 50
        corpus_texts = fallback

    # retokenize + filter
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, revision=MODEL_REV)
    filtered = []
    for t in corpus_texts:
        ids = tok(t, add_special_tokens=False).input_ids
        if 256 <= len(ids) <= 4096:
            filtered.append(t)
    print("after length filter:", len(filtered), "of", len(corpus_texts))
    random.seed(42)
    random.shuffle(filtered)

    # pack into SEQ_LEN rows
    all_ids = []
    for t in filtered:
        all_ids.extend(tok(t, add_special_tokens=False).input_ids)
        all_ids.append(tok.eos_token_id)
    rows = []
    for i in range(0, len(all_ids)-SEQ_LEN, SEQ_LEN):
        chunk = all_ids[i:i+SEQ_LEN]
        if len(chunk)==SEQ_LEN:
            rows.append(chunk)
    print("packed rows:", len(rows))
    # hold out 2%
    n_eval = max(1, len(rows)//50)
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]
    print("train rows:", len(train_rows), "eval rows:", len(eval_rows))

    # save packed data
    torch.save({"train":train_rows,"eval":eval_rows}, os.path.join(OUT_DIR,"packed.pt"))

    # ---- TRAINING ----
    print("\n=== TRAINING ===")
    model, tok2, proc = load_model()  # fresh base
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)",
    )
    model = get_peft_model(model, peft_config)
    model.train()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    sched_total = MAX_STEPS
    warmup = max(1, sched_total//20)
    def lr_lambda(step):
        if step < warmup:
            return step/warmup
        import math
        progress = (step-warmup)/(sched_total-warmup)
        return 0.5*(1+math.cos(math.pi*progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_tensor = torch.tensor(train_rows, dtype=torch.long)  # [N, SEQ]
    step = 0
    t_train = time.time()
    grad_norms = []
    losses = []
    while step < MAX_STEPS:
        idx = (step * BATCH) % (train_tensor.shape[0]-BATCH+1)
        batch = train_tensor[idx:idx+BATCH].to("cuda")
        labels = batch.clone()
        # mask padding not needed (packed, no pad) but keep eos? supervise all
        attn = torch.ones_like(batch)
        out = model(input_ids=batch, attention_mask=attn, labels=labels)
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        if (step+1) % GRAD_ACCUM == 0:
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            grad_norms.append(float(gn))
            opt.step()
            sched.step()
            opt.zero_grad()
        losses.append(float(out.loss.item()))
        step += 1
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{MAX_STEPS} loss={out.loss.item():.4f} gn={grad_norms[-1] if grad_norms else 0:.4f} elapsed={time.time()-t_train:.0f}s")
        if step % 50 == 0:
            # checkpoint to disk
            ckpt = os.path.join(OUT_DIR, f"ckpt_step{step}")
            model.save_pretrained(ckpt)
            print("saved", ckpt)
    # final save
    final_ckpt = os.path.join(OUT_DIR, "adapter_final")
    out_dir2 = "/outputs"
    os.makedirs(out_dir2, exist_ok=True)
    model.save_pretrained(final_ckpt)
    tok.save_pretrained(final_ckpt)
    proc.save_pretrained(final_ckpt)
    # also persist to mounted output volume
    model.save_pretrained(os.path.join(out_dir2, "adapter_final"))
    tok.save_pretrained(os.path.join(out_dir2, "adapter_final"))
    proc.save_pretrained(os.path.join(out_dir2, "adapter_final"))
    print("FINAL ADAPTER:", final_ckpt)
    print("grad_norms min/max:", min(grad_norms) if grad_norms else None, max(grad_norms) if grad_norms else None)
    print("mean loss first10:", np.mean(losses[:10]), "last10:", np.mean(losses[-10:]))

    # verify adapter applied assertion: check a decoder param differs from base
    # (base reloaded separately in eval)
    with open(os.path.join(OUT_DIR,"train_stats.json"),"w") as f:
        json.dump({"steps":step,"grad_norms":grad_norms,"losses":losses,"mean_loss_first10":float(np.mean(losses[:10])),"mean_loss_last10":float(np.mean(losses[-10:]))}, f)

    del model; torch.cuda.empty_cache()

    # ---- POST EVAL ----
    print("\n=== POST EVAL ===")
    model, tok, proc = load_model(adapter_path=final_ckpt)
    # audio encoder unchanged check (params exist in both base and adapted)
    base_model, _, _ = load_model()
    ae_diffs = []
    for n,p in base_model.named_parameters():
        if "audio_encoder" in n:
            ap = dict(model.named_parameters()).get(n)
            if ap is None: continue
            ae_diffs.append((n, (p.detach().float()-ap.detach().float()).abs().max().item()))
    max_ae = max(d for _,d in ae_diffs) if ae_diffs else 0.0
    print(f"audio_encoder max param diff vs base = {max_ae}")
    assert max_ae < 1e-6, "AUDIO ENCODER CHANGED"
    # adapter applied: compare effective backbone output on a fixed text input
    test_ids = tok("Barbados is an island in the Caribbean.", return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        base_out = base_model(input_ids=test_ids).logits
        adapt_out = model(input_ids=test_ids).logits
    out_diff = (base_out.detach().float() - adapt_out.detach().float()).abs().max().item()
    print(f"adapter applied check: max abs logit diff vs base = {out_diff}")
    assert out_diff > 1e-5, "ADAPTER NOT APPLIED (no output change)"
    adapter_applied_diff = out_diff

    post_knowledge, post_kr = score_knowledge(model, tok, probes)
    print("POST knowledge:", json.dumps(post_knowledge))
    post_asr = {}
    for sp in sample_files:
        try:
            post_asr[os.path.basename(sp)] = transcribe_clip(model, proc, sp)
            print("POST ASR", os.path.basename(sp), "=>", post_asr[os.path.basename(sp)][:200])
        except Exception as e:
            post_asr[os.path.basename(sp)] = f"ERROR: {e}"
            print("POST ASR ERROR", os.path.basename(sp), e)

    # knowledge probe accuracy delta by track
    def track_acc(s): return s["by_track"]
    result = {
        "base_knowledge": base_knowledge,
        "post_knowledge": post_knowledge,
        "base_asr": base_asr,
        "post_asr": post_asr,
        "adapter_applied_logit_diff": adapter_applied_diff,
        "audio_encoder_maxdiff": max_ae,
        "train_stats_file": os.path.join(OUT_DIR,"train_stats.json"),
        "elapsed_min": round((time.time()-t0)/60,1),
    }
    with open(os.path.join(OUT_DIR,"results.json"),"w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir2,"results.json"),"w") as f:
        json.dump(result, f, indent=2)
    print("RESULTS:", json.dumps(result, indent=2)[:3000])
    print("PILOT_DONE")

    # upload adapter to hub
    try:
        model.push_to_hub(ADAPTER_REPO, private=True)
        proc.push_to_hub(ADAPTER_REPO, private=True)
        tok.push_to_hub(ADAPTER_REPO, private=True)
        print("UPLOADED to", ADAPTER_REPO)
    except Exception as e:
        print("UPLOAD FAILED:", e)

if __name__ == "__main__":
    main()
