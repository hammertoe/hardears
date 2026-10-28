# HardEars Documentation

Documentation for HardEars — a Bajan English speech recognition agent being built for the [Nebius Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/), Best Apps and Agents track.

HardEars builds on two sibling projects that already exist:

- `../bimomni` — BimOmni multimodal model. Owns the Barbados Parliament audio corpus and the Gemini batch transcript-generation pipeline.
- `../bimgemma` — ASR baseline bake-off on Bajan audio. Provides the evaluation methodology.

## Index

- [strategy.md](./strategy.md) — Track, architecture, data plan, base model, eval methodology, reference projects
- [project.md](./project.md) — What HardEars is, why it exists, stack
- [naming.md](./naming.md) — Naming, cultural context, collision check, base-model identity
- [brand.md](./brand.md) — Taglines and brand voice
- [pitch.md](./pitch.md) — Elevator pitch iterations
- [lfm3-audio-text-dapt.md](./lfm3-audio-text-dapt.md) — Text-only DAPT experiment on LFM3-Audio-2B: plan and outcome
- [lfm3-dapt-diagnosis.md](./lfm3-dapt-diagnosis.md) — Why the DAPT pilot failed, and what to try instead
- [devpost-submission.md](./devpost-submission.md) — Hackathon submission content (Devpost form)
- [next-steps.md](./next-steps.md) — Ordered checklist of remaining work

## Status

Pre-build. Audio corpus and pipeline already exist under `../bimomni/hammertoe__barbados-parliament-clips-v1/`. Transcripts are deferred and must be filled before fine-tuning can start. Strategy decisions are recorded in `strategy.md`; everything else feeds from there.
