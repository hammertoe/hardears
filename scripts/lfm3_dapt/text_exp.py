# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.1",
#   "transformers==5.20.2",
#   "peft==0.21.2",
#   "accelerate==1.15.0",
#   "huggingface-hub==2.3.0",
#   "safetensors==0.8.0",
#   "numpy",
# ]
# ///
"""Decisive text-side experiment.

Hypothesis under test: the exp-1 failure was NOT "text DAPT cannot work on
this model" but (a) the corpus lacked the probe facts and (b) the run was
6% of one epoch with no task-shaped signal.

This builds a FACT-DENSE corpus for the 14 local/rare_local probe facts that
the corpus can actually support, in three flavours, and trains a LoRA on each:

  A. raw text      - fact sentences only, plain next-token (tests density)
  B. QA format     - same facts rendered as question -> answer (tests format)
  C. raw + QA      - mixture

Eval protocol fixes the fragile argmin-MCQ metric:
  - score each choice by mean token NLL (length-normalised) AND sum NLL
  - report per-item margin (gold NLL - best distractor NLL) so we can see
    movement even when argmin accuracy is noise
  - report a dev/test split so we can select on dev, not on test
"""
import os, json, re, random, time, argparse
import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
PROBES = "/inputs/probes/barbados-knowledge-v1.jsonl"
CORPUS_DIR = "/inputs/corpus"

def load_probes():
    return [json.loads(l) for l in open(PROBES) if l.strip()]

def load_corpus_text():
    import glob
    parts=[]
    for f in sorted(glob.glob(os.path.join(CORPUS_DIR,"*.jsonl"))):
        for line in open(f):
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            for m in d.get("messages",[]):
                c=m.get("content")
                if isinstance(c,str) and c.strip():
                    parts.append(c)
    return parts

def facts_from_corpus(text, probes):
    """For each non-control probe, collect corpus sentences that state its answer."""
    sents=[s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if 40 < len(s.strip()) < 600]
    facts={}
    for p in probes:
        if p["track"]=="control": continue
        ans=p["choices"][p["answer_index"]]
        keys=[w.lower() for w in re.findall(r"[A-Za-z']+", p["prompt"]) if len(w)>4]
        if not keys: continue
        ev=[s for s in sents if ans.lower() in s.lower() and sum(1 for k in keys if k in s.lower())>=2]
        if ev:
            facts[p["id"]]={"probe":p,"sentences":ev[:8]}
    return facts

def build_sft_rows(facts, tok, mode, rng):
    """mode: 'raw' | 'qa' | 'mix'. Returns list of token id lists."""
    rows=[]
    for pid, f in facts.items():
        p=f["probe"]; ans=p["choices"][p["answer_index"]]
        for s in f["sentences"]:
            if mode in ("raw","mix"):
                rows.append(tok(s, add_special_tokens=False).input_ids)
            if mode in ("qa","mix"):
                # question -> answer, chatml, answer tokens supervised later
                q=p["prompt"]
                a=ans
                text=f"<|startoftext|><|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
                rows.append(tok(text, add_special_tokens=False).input_ids)
    rng.shuffle(rows)
    return rows

def score_choices(model, tok, probes):
    """Score each choice with mean-token NLL and total NLL; return per-item detail."""
    out=[]
    with torch.inference_mode():
        for p in probes:
            prompt=p["prompt"]
            pid=prompt
            pids=tok(prompt, add_special_tokens="False" if False else False, return_tensors="pt").input_ids.to("cuda")
            nll_mean=[]; nll_sum=[]; ntok=[]
            for c in p["choices"]:
                full=tok(prompt+" "+c, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
                labels=full.clone(); labels[:, :pids.shape[1]] = -100
                if (labels!=-100).sum()==0: nll_mean.append(1e9); nll_sum.append(1e9); ntok.append(1); continue
                logits=model(input_ids=full).logits
                lp=torch.log_softmax(logits[:, :-1].float(), dim=-1)
                tgt=full[:, 1:]
                mask=(labels[:, 1:]!=-100)
                tok_lp=lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                tok_lp=tok_lp*mask
                s=-(tok_lp.sum().item()); n=int(mask.sum().item())
                nll_sum.append(s); nll_mean.append(s/max(n,1)); ntok.append(n)
            gi=p["answer_index"]
            gi_sum=nll_sum[gi]
            other=[v for i,v in enumerate(nll_sum) if i!=gi]
            margin=min(other)-gi_sum if other else 0.0
            pred_sum=int(np.argmin(nll_sum)); pred_mean=int(np.argmin(nll_mean))
            out.append({"id":p["id"],"track":p["track"],"pred_sum":pred_sum,"pred_mean":pred_mean,
                        "gold":gi,"ok_sum":pred_sum==gi,"ok_mean":pred_mean==gi,
                        "margin_sum":margin,"nll_sum":nll_sum,"nll_mean":nll_mean})
    return out

def summarize(details):
    tracks={}
    for d in details:
        t=tracks.setdefault(d["track"],{"n":0,"ok_sum":0,"ok_mean":0,"margins":[]})
        t["n"]+=1; t["ok_sum"]+=d["ok_sum"]; t["ok_mean"]+=d["ok_mean"]; t["margins"].append(d["margin_sum"])
    summ={"by_track":{k:{"n":v["n"],"acc_sum":v["ok_sum"]/v["n"],"acc_mean":v["ok_mean"]/v["n"],
                          "mean_margin":float(np.mean(v["margins"])),
                          "median_margin":float(np.median(v["margins"]))} for k,v in tracks.items()}}
    loc=[d for d in details if d["track"] in ("local","rare_local")]
    summ["local_total_acc_sum"]=np.mean([d["ok_sum"] for d in loc]) if loc else 0
    summ["local_total_acc_mean"]=np.mean([d["ok_mean"] for d in loc]) if loc else 0
    summ["local_mean_margin"]=float(np.mean([d["margin_sum"] for d in loc])) if loc else 0
    summ["overall_acc_sum"]=np.mean([d["ok_sum"] for d in details])
    summ["overall_acc_mean"]=np.mean([d["ok_mean"] for d in details])
    return summ

def train(model, tok, rows, steps, lr, outdir, rank=16):
    peft_config=LoraConfig(r=rank, lora_alpha=2*rank, lora_dropout=0.05, bias="none",
                            task_type="CAUSAL_LM",
                            target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)")
    model=get_peft_model(model, peft_config)
    model.train(); model.gradient_checkpointing_enable()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    import math
    warm=max(1, steps//20)
    sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s/warm if s<warm else 0.5*(1+math.cos(math.pi*(s-warm)/max(1,steps-warm))))
    # pack rows into fixed-length blocks
    flat=[]
    for r in rows: flat.extend(r); flat.append(tok.eos_token_id)
    BLOCK=1024
    blocks=[flat[i:i+BLOCK] for i in range(0,len(flat)-BLOCK,BLOCK)]
    print(f"  rows={len(rows)} tokens={len(flat)} blocks={len(blocks)}")
    losses=[]; gns=[]; step=0; t0=time.time()
    while step<steps:
        b=torch.tensor([blocks[step%len(blocks)]],dtype=torch.long,device="cuda")
        labels=b.clone()
        out=model(input_ids=b, labels=labels)
        out.loss.backward()
        gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        gns.append(float(gn)); losses.append(float(out.loss.item()))
        opt.step(); sched.step(); opt.zero_grad(); step+=1
        if step%25==0 or step==1:
            print(f"  step {step}/{steps} loss={out.loss.item():.4f} gn={gn:.3f} t={time.time()-t0:.0f}s")
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    with open(os.path.join(outdir,"train_stats.json"),"w") as f:
        json.dump({"steps":steps,"lr":lr,"rank":rank,"losses":losses,"grad_norms":gns,
                   "mean_first10":float(np.mean(losses[:10])),"mean_last10":float(np.mean(losses[-10:]))},f)
    print(f"  mean loss first10={np.mean(losses[:10]):.4f} last10={np.mean(losses[-10:]):.4f} gn min/max={min(gns):.4f}/{max(gns):.4f}")
    return model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", default="qa", choices=["raw","qa","mix"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default="/outputs/exp")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("=== EXP", args.mode, "steps",args.steps,"lr",args.lr,"rank",args.rank,"===")
    rng=random.Random(args.seed)
    probes=load_probes()
    corpus=load_corpus_text()
    text="\n".join(corpus)
    print("corpus parts:",len(corpus),"chars:",len(text))
    facts=facts_from_corpus(text, probes)
    print("probe facts recoverable from corpus:",len(facts),"/",len([p for p in probes if p['track']!='control']))
    with open(os.path.join(args.out,"facts.json"),"w") as f:
        json.dump({k:{"probe":v["probe"],"n_sentences":len(v["sentences"]),"examples":v["sentences"][:2]} for k,v in facts.items()},f,indent=2,default=str)
    tok=AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    rows=build_sft_rows(facts, tok, args.mode, rng)
    print("sft rows:",len(rows))
    # train/dev/test split over the FACT probes: train on half the facts, test on half
    fids=sorted(facts.keys())
    rng.shuffle(fids)
    n_test=max(1,len(fids)//4); n_dev=max(1,len(fids)//4)
    test_f=set(fids[:n_test]); dev_f=set(fids[n_test:n_test+n_dev]); train_f=set(fids[n_test+n_dev:])
    print(f"fact split: train={len(train_f)} dev={len(dev_f)} test={len(test_f)}")
    def rows_for(fset):
        sub={k:v for k,v in facts.items() if k in fset}
        return build_sft_rows(sub, tok, args.mode, random.Random(args.seed))
    train_rows=rows_for(train_f)
    # eval probes: control + the held-out fact probes (dev+test)
    eval_probes=[p for p in probes if p["id"] in (dev_f|test_f)]
    ctrl=[p for p in probes if p["track"]=="control"]
    print("eval probes (held-out facts):",len(eval_probes)," control:",len(ctrl))
    # BASE eval
    model=AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    base_details=score_choices(model, tok, eval_probes+ctrl)
    base_ctrl=score_choices(model, tok, ctrl)
    print("BASE (held-out facts):", json.dumps(summarize(base_details), indent=2))
    del model; torch.cuda.empty_cache()
    # TRAIN
    model=AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model=train(model, tok, train_rows, args.steps, args.lr, os.path.join(args.out,"adapter"), args.rank)
    model.eval()
    post_details=score_choices(model, tok, eval_probes+ctrl)
    post_ctrl=score_choices(model, tok, ctrl)
    print("POST (held-out facts):", json.dumps(summarize(post_details), indent=2))
    res={"mode":args.mode,"steps":args.steps,"lr":args.lr,"rank":args.rank,
         "n_train_facts":len(train_f),"n_dev":len(dev_f),"n_test":len(test_f),
         "base_heldout":summarize(base_details),"post_heldout":summarize(post_details),
         "base_control":summarize(base_ctrl),"post_control":summarize(post_ctrl),
         "base_details":base_details,"post_details":post_details}
    with open(os.path.join(args.out,"results.json"),"w") as f: json.dump(res,f,indent=2)
    print("RESULTS_JSON:", json.dumps({k:v for k,v in res.items() if not k.endswith("details")}, indent=2))
    print("EXP_DONE")

if __name__=="__main__":
    main()
