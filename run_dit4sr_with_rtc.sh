#!/usr/bin/env bash
# End-to-end RTC on DiT4SR: caption each LR image with Gemma-4 (reasoning prompt),
# then super-resolve with DiT4SR conditioned on that caption.
#
# Runs entirely in the RTC env (requirements_rtc.txt). Because Gemma-4 sampling is
# stochastic, this regenerates captions; to reproduce the paper exactly, skip step 1
# and pass captioning/reason_captions.json to DiT4SR/run_dit4sr_from_caption.py.
#
# Usage:
#   GEMMA4_PATH=... SD35_PATH=... DIT4SR_Q_PATH=... \
#     bash run_dit4sr_with_rtc.sh <LR_dir> <output_dir>
set -euo pipefail

LR_DIR=${1:?"usage: run_dit4sr_with_rtc.sh <LR_dir> <output_dir>"}
OUT_DIR=${2:?"usage: run_dit4sr_with_rtc.sh <LR_dir> <output_dir>"}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAP_JSON="${OUT_DIR}/reason_captions.json"

: "${GEMMA4_PATH:?set GEMMA4_PATH to your gemma-4-31B-it directory}"
: "${SD35_PATH:?set SD35_PATH to your stable-diffusion-3.5-medium directory}"
: "${DIT4SR_Q_PATH:?set DIT4SR_Q_PATH to your dit4sr_q directory}"

mkdir -p "$OUT_DIR"

echo "[1/2] Gemma-4 captioning (LR -> reasoning caption) ..."
python "${HERE}/captioning/caption_gemma4.py" --input_dir "$LR_DIR" --output "$CAP_JSON"

echo "[2/2] DiT4SR super-resolution (caption -> SR) ..."
python "${HERE}/DiT4SR/run_dit4sr_from_caption.py" \
    --image_path "$LR_DIR" --caption_json "$CAP_JSON" --output_dir "${OUT_DIR}/images"

echo "Done. SR images in ${OUT_DIR}/images"
