"""RTC Stage 1: caption each LR image with Gemma-4 using the reasoning prompt.

For every image in --input_dir, Gemma-4 is run on the LR image with the reasoning
prompt (reason_prompt.txt) and its output (the inferred scene + text) is stored as
that image's caption. The resulting JSON is what `DiT4SR/run_dit4sr_from_caption.py`
consumes. This is RTC's reasoning module; the caption is backbone-independent, so
the same JSON can drive any text-conditioned SR model.

Generation follows Gemma-4's HF recommendation (do_sample, T=1.0, top_p=0.95,
top_k=64). Sampling is stochastic, so re-running will not reproduce our captions
verbatim; the exact captions used in the paper ship as `reason_captions.json`.

Usage:
    GEMMA4_PATH=/path/to/gemma-4-31B-it python caption_gemma4.py \
        --input_dir /path/to/LR --output reason_captions.json
"""
import argparse
import glob
import json
import os
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor

# Prefer the explicit Gemma-4 class; fall back for forward compatibility.
try:
    from transformers import Gemma4ForConditionalGeneration as _AutoModel
except ImportError:
    from transformers import AutoModelForImageTextToText as _AutoModel

DEFAULT_MODEL_ID = os.environ.get("GEMMA4_PATH", "/path/to/gemma-4-31B-it")
DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reason_prompt.txt")
DEFAULT_MODEL_TAG = "gemma4-31b"
DEFAULT_MAX_NEW_TOKENS = 1024

# Gemma-4 HF best-practice sampling (matches the paper).
GEN = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
ENABLE_THINKING = False
IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tiff"]


def collect_images(input_dir):
    if os.path.isfile(input_dir):
        return [input_dir]
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(input_dir, ext)))
    return sorted(p for p in paths if not os.path.basename(p).startswith("."))


def load_checkpoint(path):
    """Resume already-captioned (filename -> response) pairs from a JSONL checkpoint."""
    done = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item.get("filename"), str) and isinstance(item.get("response"), str):
                    done[item["filename"]] = item["response"]
    return done


def run_inference(model, processor, instruction, image_paths, max_new_tokens, checkpoint_path):
    results = load_checkpoint(checkpoint_path)
    if results:
        print(f"  Resumed {len(results)} already-captioned images from {checkpoint_path}")
    ckpt_f = open(checkpoint_path, "a", encoding="utf-8") if checkpoint_path else None
    try:
        for idx, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            if filename in results:
                continue
            print(f"  [{idx + 1}/{len(image_paths)}] {filename}")
            image = Image.open(img_path).convert("RGB")
            # Gemma-4 model card: image BEFORE text.
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **GEN)
            response = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            print(f"     -> {response[:140]}{'...' if len(response) > 140 else ''}")
            results[filename] = response
            if ckpt_f:
                ckpt_f.write(json.dumps({"filename": filename, "response": response}, ensure_ascii=False) + "\n")
                ckpt_f.flush()
    finally:
        if ckpt_f:
            ckpt_f.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="LR image file or directory of LR images.")
    parser.add_argument("--output", type=str, default="reason_captions.json", help="Output caption JSON path.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT_PATH, help="Reasoning prompt text file.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Gemma-4 weights path (or set GEMMA4_PATH).")
    parser.add_argument("--model-tag", type=str, default=DEFAULT_MODEL_TAG)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    prompt_text = Path(args.prompt).read_text(encoding="utf-8").strip()
    prompt_tag = Path(args.prompt).stem
    image_paths = collect_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {args.input_dir}")

    print(f"Model    : {args.model_id}")
    print(f"Prompt   : {args.prompt} (tag={prompt_tag})")
    print(f"Images   : {len(image_paths)}")
    print(f"Sampling : {GEN}")
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = _AutoModel.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    checkpoint_path = os.path.splitext(args.output)[0] + ".checkpoint.jsonl"
    results = run_inference(model, processor, prompt_text, image_paths, args.max_new_tokens, checkpoint_path)

    output_data = {
        "model": args.model_id,
        "model_tag": args.model_tag,
        "prompt_tag": prompt_tag,
        "prompt": prompt_text,
        "generation_config": {**GEN, "max_new_tokens": args.max_new_tokens,
                              "enable_thinking": ENABLE_THINKING, "torch_dtype": "bfloat16"},
        "runtime": {"torch_version": torch.__version__, "transformers_version": transformers.__version__},
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Saved {len(results)} captions -> {args.output}")


if __name__ == "__main__":
    main()
