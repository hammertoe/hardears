# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.1",
#   "transformers==5.20.2",
#   "huggingface-hub==2.3.0",
#   "accelerate==1.15.0",
#   "soundfile==0.12.1",
#   "numpy",
# ]
# ///
"""Diagnostic: is the audio-QA failure caused by broken TTS audio, or by the model?

1. Generate TTS audio for a few probes, save wavs.
2. Transcribe each TTS wav with the model (ASR prompt) -> is the TTS intelligible?
3. Ask the model the same question as TEXT (multiple-choice) -> can it answer at all?
4. Ask the model the question as AUDIO -> does it match its own text ability?
5. Report transcript vs prompt, and text-answer vs audio-answer.
"""
import os, json, numpy as np, torch, soundfile as sf
from transformers import AutoModelForMultimodalLM, AutoTokenizer, AutoProcessor

MODEL_ID="LiquidAI/LFM3-Audio-2B"
PROBES="/inputs/probes/barbados-knowledge-v1.jsonl"
OUT="/tmp/scratchpad/diag"
os.makedirs(OUT, exist_ok=True)

def main():
    probes=[json.loads(l) for l in open(PROBES) if l.strip()]
    # pick 5 local probes
    sel=[p for p in probes if p["track"]=="local"][:5]
    tok=AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    proc=AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model=AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    codec=proc.audio_tokenizer.to("cuda").eval()

    def tts(text):
        conv=[{"role":"system","content":"Perform TTS in American English using a female voice."},
              {"role":"user","content":text},
              {"role":"assistant","content":"<|audio_start|>"}]
        inp=proc.apply_chat_template(conv, continue_final_message=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o=model.generate(**inp, max_new_tokens=256, audio_temperature=1.0, audio_top_k=128,
                              audio_repetition_penalty=1.1, audio_repetition_window=4096, return_dict_in_generate=True)
            w=codec.decode(o.audio_codes[:1,:,:o.audio_lengths[0]]).audio_values[0]
        w=w.float().cpu().numpy()
        n=int(len(w)*16000/24000)
        return np.interp(np.linspace(0,len(w)-1,n), np.arange(len(w)), w).astype("float32")

    def asr(w):
        conv=[{"role":"system","content":"Perform ASR in English."},
              {"role":"user","content":[{"type":"audio","audio":w}]}]
        inp=proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o=model.generate(**inp, max_new_tokens=128, suppress_mixed=True, suppress_audio_out=True, return_dict_in_generate=True)
        return proc.decode(o.sequences[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def free_text(prompt):
        conv=[{"role":"user","content":prompt}]
        inp=proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o=model.generate(**inp, max_new_tokens=64, suppress_mixed=True, suppress_audio_out=True, return_dict_in_generate=True)
        return proc.decode(o.sequences[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def audio_free(prompt, w):
        conv=[{"role":"user","content":[{"type":"audio","audio":w},{"type":"text","text":prompt}]}]
        inp=proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o=model.generate(**inp, max_new_tokens=64, suppress_mixed=True, suppress_audio_out=True, return_dict_in_generate=True)
        return proc.decode(o.sequences[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def mc_prompt(p):
        return p["prompt"]+" " + " ".join(f"({chr(65+i)}) {c}" for i,c in enumerate(p["choices"]))

    rows=[]
    for i,p in enumerate(sel):
        w=tts(p["prompt"])
        path=os.path.join(OUT,f"tts{i}.wav")
        sf.write(path, w, 16000)
        tr=asr(w)
        gold=p["choices"][p["answer_index"]]
        mcp=mc_prompt(p)
        txt_ans=free_text(mcp)
        aud_ans=audio_free(mcp, w)
        rows.append({"id":p["id"],"prompt":p["prompt"],"gold":gold,
                     "tts_transcript":tr,"tts_dur_s":round(len(w)/16000,2),
                     "text_mc_answer":txt_ans,"audio_mc_answer":aud_ans,
                     "text_has_gold":gold.lower() in txt_ans.lower(),
                     "audio_has_gold":gold.lower() in aud_ans.lower()})
        print(json.dumps(rows[-1], indent=2))
        print("-"*70)
    with open(os.path.join(OUT,"diag.json"),"w") as f: json.dump(rows,f,indent=2)
    print("SUMMARY: tts intelligible:", sum(1 for r in rows if r["tts_dur_s"]>0.5), "/", len(rows))
    print("text_mc correct:", sum(r["text_has_gold"] for r in rows), "/", len(rows))
    print("audio_mc correct:", sum(r["audio_has_gold"] for r in rows), "/", len(rows))
    # dump audio stats
    for r in rows:
        print(r["id"], "dur", r["tts_dur_s"], "| tts:", r["tts_transcript"][:100])
    print("DIAG_DONE")

if __name__=="__main__":
    main()
