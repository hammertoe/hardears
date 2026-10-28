# Project

## What it is

HardEars is a speech recognition model fine-tuned specifically for Bajan English and Barbadian audio.

It fine-tunes NVIDIA's `nemotron-speech-streaming-en-0.6b` on the existing Barbados Parliament audio corpus, then wraps the result in a Nemotron LLM reasoning agent to deliver useful downstream tasks.

## Why it exists

Today's leading ASR systems perform well on standard English but often struggle with Caribbean accents, local vocabulary, place names, code-switching, and fast conversational speech.

In Barbados, valuable information lives in radio, interviews, parliamentary proceedings, community media, and everyday spoken conversation. If AI systems cannot accurately understand that speech, much of it remains effectively invisible to search and other AI applications.

The premise is simple: Barbadians should not have to change how they speak to be understood by AI.

## What it's for

The wider goal is to make Bajan speech machine-readable. Practical applications:

- Better search across Barbadian audio archives
- Media monitoring of local radio and interviews
- Accessibility (transcripts, captions)
- Cultural preservation of an oral language variety
- Locally grounded AI applications and agents

## Reference projects

HardEars builds directly on:

- `../bimomni` — BimOmni multimodal model. Owns the Barbados Parliament audio corpus (`hammertoe/barbados-parliament-clips-v1`, 815+ clips) and the Gemini batch transcript-generation pipeline. We reuse the corpus, the Gemini batch script, and the chunked transcription helpers.
- `../bimgemma` — ASR baseline bake-off on held-out Bajan audio. We reuse the evaluation methodology (entity recall, WER, hallucination probes) verbatim.

See `strategy.md` for the full picture.

## Stack and approach

- **Track:** Best Apps and Agents (Nebius Global AI Hackathon). Reasoning in `strategy.md`.
- **ASR base model:** `nvidia/nemotron-speech-streaming-en-0.6b` (locked). This is the model the existing parliamentary corpus pipeline already uses, so fine-tuning closes the loop.
- **LLM reasoning layer:** Nemotron 3 Ultra / Nano / Super via Nebius Token Factory.
- **Training compute:** Nebius AI Cloud GPUs (H100 or Blackwell). Reuses the parl pipeline's memory profile.
- **Data:** `hammertoe/barbados-parliament-clips-v1` (815+ clips, transcripts filled via the bimomni Gemini batch pipeline before training).
- **Approach:** Fine-tune Nemotron-speech via NeMo ASR recipe, then wrap in an agent that does downstream reasoning via Token Factory.
- **Evaluation:** Reuses bimgemma's 12-clip / 34-entity bake-off methodology. Entity recall vs hand-written surface forms, WER against `gpt-transcribe` and `whisper-1`, hallucination probes.
- **App surface:** Gradio or lightweight web UI showing before/after transcripts and downstream task demos.

## Constraints and non-goals

- Not a general-purpose English ASR. Specifically Bajan.
- Not a replacement for human transcription. A foundation layer.
- Not a closed, opaque system. The aim is to release the model.
- Not a re-implementation of the BimOmni text-only DAPT pipeline. That recipe operates on newspaper text and the Qwen3-Omni thinker; ASR fine-tuning is a different problem with a different toolchain (NeMo).
