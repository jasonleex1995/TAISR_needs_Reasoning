"""Run DiT4SR super-resolution using an RTC caption for each LR image.

    RTC integration (OURS). The DiT4SR backbone itself (pipelines/, model_dit4sr/,
    utils/) is from https://github.com/Adam-duan/DiT4SR and is unmodified. This
    script replaces DiT4SR's built-in captioner with an RTC caption: for each LR
    image it looks up the caption produced by `captioning/caption_gemma4.py`
    (Gemma-4 run on the LR image with the reasoning prompt) and feeds it as the
    text condition. That is the whole of RTC on DiT4SR: same backbone, reasoning
    caption swapped in. Everything below other than the caption lookup/formatting
    is standard DiT4SR inference.

Usage:
    python run_dit4sr_from_caption.py \
        --image_path  /path/to/LR \
        --caption_json ../captioning/reason_captions.json \
        --output_dir  ./results/DiT4SR_RTC

Set SD35_PATH / DIT4SR_Q_PATH (see config.py) to your weight locations first.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import re
import time
from PIL import Image

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from pipelines.pipeline_dit4sr import StableDiffusion3ControlNetPipeline
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from config import SD35_PATH, DIT4SR_Q_PATH

logger = get_logger(__name__, log_level="INFO")

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
MODEL_NAME = 'DiT4SR'


# ---------------------------------------------------------------------------
# RTC caption handling (OURS)
# ---------------------------------------------------------------------------
NO_TEXT_SENTINEL = "[NO TEXT]"

# Phrases the MLLM occasionally emits when it finds no text; strip verbatim
# (longest first so a shorter pattern does not eat the wrapper).
_HALLUCINATION_PATTERNS = (
    'The visible text reads "There is no readable text."',
    '"There is no readable text."',
    'There is no readable text.',
)
CAPTION_MODES = ("full", "text_only")


def load_captions(caption_json_path):
    """Return (results, metadata). Caption JSON shape (see caption_gemma4.py):
      {"model": ..., "prompt": ..., "results": {"<image>.png": "<caption>", ...}}
    """
    with open(caption_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", {})
    metadata = {k: v for k, v in data.items() if k != "results"}
    return results, metadata


def lookup_caption(results, image_path):
    """Look up a caption by image basename, with a stem fallback for ext changes."""
    fname = os.path.basename(image_path)
    if fname in results:
        return results[fname]
    stem = os.path.splitext(fname)[0]
    for k, v in results.items():
        if os.path.splitext(k)[0] == stem:
            return v
    raise KeyError(
        f"No caption for {fname} in caption JSON "
        f"(sample keys: {list(results.keys())[:3]})"
    )


def _scrub_hallucinations(caption):
    for pat in _HALLUCINATION_PATTERNS:
        caption = caption.replace(pat, '')
    return ' '.join(caption.split())


def _is_no_readable_text(s):
    return s.strip().rstrip('.').strip().lower() == 'there is no readable text'


def _extract_and_reformat_quotes(caption):
    """text_only mode: keep only the quoted OCR spans, reformatted as
    'The image with the text "X1", "X2", ..., "XN".'  ('' if none remain)."""
    quotes = re.findall(r'"([^"]*)"', caption)
    quotes = [q for q in quotes if q.strip() and not _is_no_readable_text(q)]
    if not quotes:
        return ""
    return 'The image with the text ' + ', '.join(f'"{q}"' for q in quotes) + '.'


def apply_caption_mode(caption, caption_mode):
    """Turn a raw RTC caption into the text the SR model sees.
      - 'full'      : scrub hallucinations, keep the caption as-is (paper default)
      - 'text_only' : keep only the quoted OCR spans
    """
    if caption == NO_TEXT_SENTINEL:
        return ""
    caption = _scrub_hallucinations(caption)
    if caption_mode == "full":
        result = caption
    elif caption_mode == "text_only":
        result = _extract_and_reformat_quotes(caption)
    else:
        raise ValueError(f"unknown caption_mode={caption_mode!r}; expected {CAPTION_MODES}")
    return result.strip()


def caption_tag_from_path(caption_json_path):
    return os.path.splitext(os.path.basename(caption_json_path))[0]


# ---------------------------------------------------------------------------
# DiT4SR pipeline setup (from DiT4SR, unchanged)
# ---------------------------------------------------------------------------
def list_images(image_path):
    if os.path.isfile(image_path):
        return [image_path]
    if not os.path.isdir(image_path):
        raise FileNotFoundError(f"image_path not found: {image_path}")
    out = []
    for name in os.listdir(image_path):
        if name.startswith('.'):
            continue
        full = os.path.join(image_path, name)
        if os.path.isfile(full) and name.lower().endswith(IMAGE_EXTS):
            out.append(full)
    return sorted(out)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision, subfolder="text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(class_one, class_two, class_three, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant)
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant)
    return text_encoder_one, text_encoder_two, text_encoder_three


def load_dit4sr_pipeline(args, accelerator):
    from model_dit4sr.transformer_sd3 import SD3Transformer2DModel

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    transformer = SD3Transformer2DModel.from_pretrained(
        args.transformer_model_name_or_path, subfolder="transformer")
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision)
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision)

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3")
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args)

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    transformer.requires_grad_(False)

    pipeline = StableDiffusion3ControlNetPipeline(
        vae=vae, text_encoder=text_encoder_one, text_encoder_2=text_encoder_two,
        text_encoder_3=text_encoder_three, tokenizer=tokenizer_one, tokenizer_2=tokenizer_two,
        tokenizer_3=tokenizer_three, transformer=transformer, scheduler=scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    for m in (text_encoder_one, text_encoder_two, text_encoder_three, vae, transformer):
        m.to(accelerator.device, dtype=weight_dtype)
    return pipeline


def main(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if args.seed is not None:
        set_seed(args.seed)

    caption_tag = caption_tag_from_path(args.caption_json)
    results, meta = load_captions(args.caption_json)

    output_dir = args.output_dir or os.path.join(
        './results', args.benchmark, f'{MODEL_NAME}__{caption_tag}__{args.caption_mode}', 'images')
    caption_dir = os.path.join(os.path.dirname(output_dir.rstrip('/')), 'captions')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(caption_dir, exist_ok=True)
    print(f'caption json:   {args.caption_json}')
    print(f'image save dir: {output_dir}')

    pipeline = load_dit4sr_pipeline(args, accelerator)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    image_names = list_images(args.image_path)
    if not image_names:
        print(f'No images found in {args.image_path}')
        return

    for image_idx, image_name in enumerate(image_names):
        print(f'============ {image_idx+1}/{len(image_names)}: {os.path.basename(image_name)} ============')
        validation_image = Image.open(image_name).convert("RGB")
        stem, _ = os.path.splitext(os.path.basename(image_name))

        # --- RTC: look up the reasoning caption and use it as the text prompt ---
        raw_caption = lookup_caption(results, image_name)
        validation_prompt = apply_caption_mode(raw_caption, args.caption_mode)
        with open(os.path.join(caption_dir, f'{stem}.json'), 'w', encoding='utf-8') as f:
            json.dump({'caption_raw': raw_caption, 'caption_mode': args.caption_mode,
                       'caption_used': validation_prompt}, f, ensure_ascii=False, indent=2)

        validation_prompt = validation_prompt + ' ' + args.added_prompt
        negative_prompt = args.negative_prompt
        print(validation_prompt)

        # --- standard DiT4SR preprocessing / inference ---
        ori_width, ori_height = validation_image.size
        rscale = args.upscale
        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            validation_image = validation_image.resize(
                (int(scale * ori_width), int(scale * ori_height)), Image.BICUBIC)
        validation_image = validation_image.resize(
            (validation_image.size[0] * rscale, validation_image.size[1] * rscale), Image.BICUBIC)
        validation_image = validation_image.resize(
            (validation_image.size[0] // 8 * 8, validation_image.size[1] // 8 * 8), Image.BICUBIC)
        width, height = validation_image.size
        print(f'input size: {height}x{width}')

        with torch.autocast("cuda"):
            start_time = time.time()
            image = pipeline(
                prompt=validation_prompt, control_image=validation_image,
                num_inference_steps=args.num_inference_steps, generator=generator,
                height=height, width=width, guidance_scale=args.guidance_scale,
                negative_prompt=negative_prompt, start_point=args.start_point,
                latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                args=args,
            ).images[0]
            print(f'inference time: {time.time() - start_time:.2f}s')

        if args.align_method == 'wavelet':
            image = wavelet_color_fix(image, validation_image)
        elif args.align_method == 'adain':
            image = adain_color_fix(image, validation_image)
        image = image.resize((ori_width * rscale, ori_height * rscale), Image.BICUBIC)
        image.save(f'{output_dir}/{stem}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=SD35_PATH)
    parser.add_argument("--transformer_model_name_or_path", type=str, default=DIT4SR_Q_PATH)
    parser.add_argument("--benchmark", type=str, default="ReasonText",
                        help="Benchmark name, used only to build the default output dir.")
    parser.add_argument("--image_path", type=str, required=True, help="Directory of LR images.")
    parser.add_argument("--caption_json", type=str, required=True,
                        help="RTC caption JSON from caption_gemma4.py (e.g. ../captioning/reason_captions.json).")
    parser.add_argument("--caption_mode", type=str, default="full", choices=list(CAPTION_MODES),
                        help="full (paper default) | text_only (quoted OCR spans only).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write SR images (default ./results/<benchmark>/DiT4SR__<tag>__<mode>/images).")
    parser.add_argument("--added_prompt", type=str,
                        default='Cinematic, hyper sharpness, highly detailed, perfect without deformations, '
                                'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                                'Grading, ultra HD, extreme meticulous detailing, skin pore detailing. ')
    parser.add_argument("--negative_prompt", type=str,
                        default='motion blur, noisy, dotted, bokeh, pointed, CG Style, 3D render, unreal engine, '
                                'blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, '
                                'jpeg artifacts, deformed, lowres, chaotic')
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=24)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RAC_SEED", 231)))
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='noise')
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    args = parser.parse_args()
    main(args)
