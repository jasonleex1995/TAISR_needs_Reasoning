# Reading the Unreadable: Text-Aware Image Super-Resolution Needs Reasoning

Official code for the NeurIPS 2026 paper. This repository provides:

- **ReasonText** — a benchmark of reasoning-required scene text for super-resolution (released on Hugging Face).
- **GTTCA** (Ground-Truth Text Crop Accuracy) — a metric that scores restored text by OCR-ing the SR output at the ground-truth text region.
- **RTC** (Reasoning Transfer via Captioning) — a training-free plug-in: run Gemma-4 on the LR image with a reasoning prompt, then feed the inferred text as a caption to any text-conditioned SR backbone. We provide a ready-to-run integration with DiT4SR.

📄 Paper: `<paper link — fill in>`

---

## Repository layout

```
TAISR_needs_Reasoning/
├── captioning/                      # RTC Stage 1: Gemma-4 LR captioning (backbone-independent)
│   ├── caption_gemma4.py            #   run Gemma-4 with the reasoning prompt on LR images
│   ├── reason_prompt.txt            #   the reasoning prompt
│   └── reason_captions.json         #   the exact captions we used (ReasonText, reason prompt)
├── DiT4SR/                          # RTC Stage 2: DiT4SR backbone + our RTC entry point
│   ├── pipelines/ model_dit4sr/ utils/   #   DiT4SR (unmodified upstream) — see Attribution
│   ├── run_dit4sr_from_caption.py   #   OURS: caption JSON -> SR
│   └── config.py                    #   weight paths (edit or set env vars)
├── gttca/                           # GTTCA metric
│   ├── evaluate_gttca.py            #   score SR outputs against GT text regions
│   ├── summarize.py                 #   aggregate per-region results into L1/L2/L3 tables
│   └── weights_config.json          #   PaddleOCR-VL path
├── run_dit4sr_with_rtc.sh           # convenience: caption -> SR in one command
├── requirements_rtc.txt             # env for captioning/ + DiT4SR/
└── requirements_gttca.txt           # env for gttca/
```

### What is ours vs. upstream

| Path | Origin |
|---|---|
| `captioning/`, `DiT4SR/run_dit4sr_from_caption.py`, `DiT4SR/config.py`, `gttca/`, `run_dit4sr_with_rtc.sh` | **Ours** |
| `DiT4SR/pipelines/`, `DiT4SR/model_dit4sr/`, `DiT4SR/utils/` | **DiT4SR** ([Adam-duan/DiT4SR](https://github.com/Adam-duan/DiT4SR)), unmodified |

RTC is training-free: `run_dit4sr_from_caption.py` simply swaps DiT4SR's built-in captioner for an RTC caption. The backbone code is untouched.

---

## Environments

Two environments are required because the components pin incompatible major versions of `transformers`:

| Env | For | Key pins |
|---|---|---|
| **RTC** (`requirements_rtc.txt`) | `captioning/` + `DiT4SR/` | Python 3.11, torch 2.6 (cu124), **transformers 5.7**, diffusers 0.39 |
| **GTTCA** (`requirements_gttca.txt`) | `gttca/` | Python 3.10, torch 2.5 (cu121), **transformers 4.55** |

Gemma-4 requires `transformers >= 5.5`, whereas PaddleOCR-VL (GTTCA's OCR engine) runs on the 4.x line, so the method and the metric live in separate envs.

```bash
# RTC (captioning + DiT4SR)
conda create -n rtc python=3.11 -y && conda activate rtc
pip install -r requirements_rtc.txt

# GTTCA (scoring)
conda create -n gttca python=3.10 -y && conda activate gttca
pip install -r requirements_gttca.txt
```

The `torch`/`torchvision` pins install CUDA wheels from the PyTorch index (`cu124` for RTC, `cu121` for GTTCA), declared by the `--extra-index-url` line at the top of each requirements file. If your CUDA version differs, edit that line (e.g. `cu118`) and the matching `torch` version.

---

## Weights (download separately)

Set the paths via environment variables (or edit `DiT4SR/config.py` and `gttca/weights_config.json`).

| Model | Used by | Source |
|---|---|---|
| Stable Diffusion 3.5 Medium | DiT4SR | https://huggingface.co/stabilityai/stable-diffusion-3.5-medium |
| DiT4SR-Q | DiT4SR | https://github.com/Adam-duan/DiT4SR |
| Gemma-4 31B-it | captioning | https://huggingface.co/google/gemma-4-31B-it |
| PaddleOCR-VL-1.5 | GTTCA | https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5 |

```bash
export SD35_PATH=/path/to/stable-diffusion-3.5-medium
export DIT4SR_Q_PATH=/path/to/dit4sr_q
export GEMMA4_PATH=/path/to/gemma-4-31B-it
# GTTCA: set paddleocr_model_path in gttca/weights_config.json, or pass --paddleocr-model-path
```

Each model's weights are governed by its own license (SD3.5 = Stability AI Community License, Gemma-4 = Gemma Terms of Use, etc.). We do not redistribute weights.

---

## Data (Hugging Face)

| Data | Link |
|---|---|
| ReasonText benchmark (LR/HR + word-level annotations) | `<HF dataset link — fill in>` |
| SR outputs (Table 1: 12 models; Table 4: 6 backbones + RTC) | `<HF dataset link — fill in>` |

All released data uses the paper's 513-image set (513 images / 3,985 annotations).

After downloading, point the commands below at these paths:
- **ReasonText benchmark** → `LR/` (pass as `--image_path`), `HR/` (`--hr-dir`), and `ReasonText-meta.json` (`--meta-json`).
- **SR outputs** → a model folder such as `sr_outputs/table4_rtc/B4_DiT4SR/` (pass as `--sr-dir`).

---

## Usage

**What do you want to do?** Reproduce our numbers → Options A / B. Run RTC on your own images → the "Run RTC on your own images" section. Measure GTTCA → Option A (it needs ground-truth text annotations).

> **Reproducibility note.** Running the pipeline yourself may differ slightly from the paper: Gemma-4 captioning is stochastic, and diffusion sampling / library versions add small run-to-run variation (typically well under ~1 point). **To reproduce the exact paper numbers, you don't need to regenerate anything — just score the SR outputs we released on Hugging Face with GTTCA** (Option A below).

### Option A — reproduce the exact paper numbers (recommended)

Download the SR outputs from Hugging Face (`<HF dataset link — fill in>`) and score any model folder with GTTCA. For example, DiT4SR + RTC (Table 4):

```bash
conda activate gttca
python gttca/evaluate_gttca.py \
    --sr-dir  /path/to/sr_outputs/table4_rtc/B4_DiT4SR \
    --hr-dir  /path/to/ReasonText/HR \
    --meta-json /path/to/ReasonText/ReasonText-meta.json \
    --weights-config gttca/weights_config.json \
    --output-dir ./results/DiT4SR_RTC/gttca \
    --model-name DiT4SR_RTC
# aggregate to Overall / Level 1 / Level 2 / Level 3:
python gttca/summarize.py --results-root ./results --output-dir ./results/summary
```

Point `--sr-dir` at any released folder to reproduce that row: `sr_outputs/table1/*` for Table 1, `sr_outputs/table4_rtc/*` for Table 4.

> **What GTTCA needs.** GTTCA scores restored text at ground-truth text regions, so it requires a `--meta-json` of word-level annotations (polygons + transcripts) in the ReasonText format. It is meant for annotated benchmarks: to score your own data, supply a meta-json in the same format as `ReasonText-meta.json`. Plain unlabeled images cannot be scored with GTTCA.

### Option B — run RTC on DiT4SR yourself (from our captions)

This regenerates the SR images from the reasoning captions we shipped (`captioning/reason_captions.json`), so numbers will be very close but not bit-identical (see the note above).

```bash
conda activate rtc
python DiT4SR/run_dit4sr_from_caption.py \
    --image_path /path/to/ReasonText/LR \
    --caption_json captioning/reason_captions.json \
    --output_dir ./results/DiT4SR_RTC/images
```

Then score `./results/DiT4SR_RTC/images` with GTTCA exactly as in Option A.

### Run RTC on your own images

Works on any folder of low-resolution images. Everything here runs in the **RTC** env; activate it and set the weight paths first (see [Weights](#weights-download-separately)).

```bash
conda activate rtc
export GEMMA4_PATH=/path/to/gemma-4-31B-it
export SD35_PATH=/path/to/stable-diffusion-3.5-medium
export DIT4SR_Q_PATH=/path/to/dit4sr_q

# caption -> SR in one command:
bash run_dit4sr_with_rtc.sh /path/to/your_LR ./results/my_run
# SR images are written to ./results/my_run/images
```

Or run the two stages separately (e.g. caption once, reuse later):

```bash
conda activate rtc
# 1) caption your images with Gemma-4 (reasoning prompt)
python captioning/caption_gemma4.py --input_dir /path/to/your_LR --output my_captions.json
# 2) super-resolve, conditioned on those captions
python DiT4SR/run_dit4sr_from_caption.py \
    --image_path /path/to/your_LR --caption_json my_captions.json \
    --output_dir ./results/my_run/images
```

### Applying RTC to another SR backbone

RTC is just "caption the LR image, then condition the SR model on that caption." `captioning/` is backbone-independent — produce a caption JSON, then feed each caption as the text prompt of any text-conditioned SR model. `DiT4SR/run_dit4sr_from_caption.py` is the reference example.

---

## License

**This project is released for non-commercial research use only.** Please do not use any part of it for commercial purposes.

- **Our contributions** — the code we wrote (`captioning/`, `DiT4SR/run_dit4sr_from_caption.py`, `DiT4SR/config.py`, `gttca/`, `run_dit4sr_with_rtc.sh`) and the ReasonText annotations (polygons, transcripts, difficulty levels, reasoning cues) — are released under the Non-Commercial Research License in [`LICENSE`](LICENSE) (annotations additionally under CC BY-NC 4.0).

- **Third-party components each retain their own licenses — please check the original sources for exact terms** before use or redistribution:
  - DiT4SR backbone under `DiT4SR/pipelines/`, `DiT4SR/model_dit4sr/`, `DiT4SR/utils/` — see [`DiT4SR/LICENSE`](DiT4SR/LICENSE) and [Adam-duan/DiT4SR](https://github.com/Adam-duan/DiT4SR).
  - ReasonText images (HR/LR) and the SR outputs are derived from the **RealSR (v3)** and **DRealSR** datasets — subject to those datasets' terms.
  - Model weights (Stable Diffusion 3.5, DiT4SR-Q, Gemma-4, PaddleOCR-VL) are **not redistributed here**; download them from their sources under their respective licenses.

ReasonText is built on RealSR-v3 [Cai et al., ICCV 2019] and DRealSR [Wei et al., ECCV 2020]; please cite both alongside our paper.

## Citation

```bibtex
<bibtex — fill in>
```

## Acknowledgements

DiT4SR ([Adam-duan/DiT4SR](https://github.com/Adam-duan/DiT4SR)), Stable Diffusion 3.5 (Stability AI), Gemma-4 (Google), and PaddleOCR-VL (PaddlePaddle).
