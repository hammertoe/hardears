#!/usr/bin/env bash
# Longform transcription pilot: 4 Barbados sittings, ~10.6 h of clip speech.
#
# Chosen from the 534 videos with an exact chamber+date order-paper match,
# balanced 2 House / 2 Senate across 2022-2024. Sitting audio is ~1.93x the
# clip speech it yields (sittings are ~34.7% speech after VAD), so transcribing
# 20.5 h of audio yields ~10.6 h of paired clip data.
#
# Resumable: a video whose _summary.json already exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEOS=(
  w9VavzudI1I   # house,   2023-05-23, 4.84 h sitting, 44th Sitting
  ItubywLJ6jA   # senate,  2022-06-01, 5.55 h sitting,  5th Sitting
  qlrZS3-6IVQ   # house,   2022-10-11, 5.94 h sitting, 25th Sitting
  ZJcYgFfu1a8   # senate,  2024-03-25, 4.15 h sitting, 43rd Sitting
)

WINDOW="${WINDOW:-3600}"
THINKING="${THINKING:-high}"
TIER="${TIER:-flex}"

for v in "${VIDEOS[@]}"; do
  if [[ -f "longform/$v/_summary.json" ]]; then
    echo "== $v: already done, skipping"
    continue
  fi
  echo "== $v: transcribing (window=${WINDOW}s thinking=${THINKING} tier=${TIER})"
  # --max-seconds 0 = whole sitting. Do not drop this: the default used to be
  # 1800, which silently capped both the download and the processing to 30 min.
  python3 scripts/transcribe_longform.py \
    --video-id "$v" \
    --max-seconds 0 \
    --window "$WINDOW" \
    --thinking "$THINKING" \
    --tier "$TIER" \
    "$@"
done

echo
echo "== pilot totals"
python3 - <<'PY'
import json, pathlib
tot = {"seconds": 0, "clips_in": 0, "clips_out": 0, "std": 0.0, "disc": 0.0}
for d in sorted(pathlib.Path("longform").glob("*/_summary.json")):
    s = json.loads(d.read_text())
    if s["video_id"] not in {"w9VavzudI1I", "ItubywLJ6jA", "qlrZS3-6IVQ", "ZJcYgFfu1a8"}:
        continue
    tot["seconds"] += s["processed_seconds"]
    tot["clips_in"] += s["clips_in_range"]
    tot["clips_out"] += s["clips_transcribed"]
    tot["std"] += s["est_cost_usd_standard"]
    tot["disc"] += s.get("est_cost_usd_discounted", 0)
    print(f"  {s['video_id']:<14}{s['clips_transcribed']:>5}/{s['clips_in_range']:<5} clips"
          f"  billed={s.get('tier_billed')}  ${s.get('est_cost_usd_discounted', 0):.3f}")
print(f"\n  {tot['seconds']/3600:.2f} h audio | {tot['clips_out']:,}/{tot['clips_in']:,} clips"
      f" | ${tot['disc']:.2f} (standard would be ${tot['std']:.2f})")
PY