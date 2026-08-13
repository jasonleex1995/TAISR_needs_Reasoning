<div align="center">

# Reading the Unreadable
### Text-Aware Image Super-Resolution Needs Reasoning

**&lt;AUTHORS — fill in&gt;** · NeurIPS 2026

[📄 Paper](https://example.com) · [🌐 Project Page](https://example.com) · [🤗 ReasonText](https://huggingface.co/datasets/Jasonleex1995/ReasonText) · [🤗 SR Outputs](https://huggingface.co/datasets/Jasonleex1995/ReasonText-SR-outputs) · [⚖️ License: CC BY-NC 4.0](LICENSE)

</div>

Official code for the two tools introduced in the paper — **GTTCA** (evaluation metric) and **RTC** (training-free method). The **ReasonText** benchmark and all SR outputs live on 🤗 Hugging Face.

- 📏 **GTTCA** (Ground-Truth Text Crop Accuracy) — crops the SR output at the **ground-truth text region**, deskews it with the annotated rotation/flip, runs OCR, and checks exact-match against the ground-truth transcript.
- 🧩 **RTC** (Reasoning Transfer via Captioning) — training-free: run an **MLLM with a reasoning prompt** on the LR image (Gemma-4 in our example), then feed the inferred text as a caption to **any text-conditioned SR backbone** (DiT4SR example included).

<p align="center"><img src="assets/rtc.png" width="88%"><br><sub><i>RTC — an MLLM reasons about the LR text and the caption is injected into a text-conditioned SR backbone.</i></sub></p>

<p align="center"><img src="assets/gttca.png" width="88%"><br><sub><i>GTTCA — crop at the GT text region → deskew (GT rotation/flip) → OCR → exact-match vs GT transcript.</i></sub></p>

---

## 🔧 Installation

Two environments are needed because the components pin incompatible major versions of `transformers` (Gemma-4 needs `transformers ≥ 5.5`; PaddleOCR-VL, GTTCA's OCR engine, runs on the 4.x line):

| Env | For | Key pins |
|---|---|---|
| **RTC** (`requirements_rtc.txt`) | `captioning/` + `DiT4SR/` | Python 3.11 · torch 2.6 (cu124) · **transformers 5.7** · diffusers 0.39 |
| **GTTCA** (`requirements_gttca.txt`) | `gttca/` | Python 3.10 · torch 2.5 (cu121) · **transformers 4.55** |

```bash
# RTC (captioning + DiT4SR super-resolution)
conda create -n rtc python=3.11 -y && conda activate rtc
pip install -r requirements_rtc.txt

# GTTCA (scoring)
conda create -n gttca python=3.10 -y && conda activate gttca
pip install -r requirements_gttca.txt
```

CUDA wheels come from the PyTorch index (`cu124` / `cu121`) via the `--extra-index-url` line at the top of each requirements file — edit it (e.g. `cu118`) and the `torch` version if your CUDA differs.

## 📦 Weights & Data

**Weights** (download separately; set via env vars or edit `DiT4SR/config.py` and `gttca/weights_config.json`):

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

**Data** (Hugging Face, 513-image set = 513 images / 3,985 annotations):

| Data | Link |
|---|---|
| ReasonText benchmark (LR/HR + word-level annotations) | https://huggingface.co/datasets/Jasonleex1995/ReasonText |
| SR outputs (Table 1: 12 models · Table 4: 6 backbones + RTC) | https://huggingface.co/datasets/Jasonleex1995/ReasonText-SR-outputs |

After download, pass: `LR/` → `--image_path`, `HR/` → `--hr-dir`, `ReasonText-meta.json` → `--meta-json`, and an SR folder (e.g. `sr_outputs/table4_rtc/B4_DiT4SR/`) → `--sr-dir`.

## ⚡ Quick Start

> **Reproducibility.** Running the pipeline yourself varies slightly from the paper (MLLM sampling is stochastic; diffusion/library versions add < ~1 pt). **To reproduce the exact numbers, score the released SR outputs with GTTCA — no regeneration needed.**

**Reproduce the paper** — score released SR outputs (e.g. DiT4SR + RTC, Table 4):

```bash
conda activate gttca
python gttca/evaluate_gttca.py \
    --sr-dir  /path/to/sr_outputs/table4_rtc/B4_DiT4SR \
    --hr-dir  /path/to/ReasonText/HR \
    --meta-json /path/to/ReasonText/ReasonText-meta.json \
    --weights-config gttca/weights_config.json \
    --output-dir ./results/DiT4SR_RTC/gttca --model-name DiT4SR_RTC
python gttca/summarize.py --results-root ./results --output-dir ./results/summary
```

Point `--sr-dir` at any released folder (`sr_outputs/table1/*`, `sr_outputs/table4_rtc/*`) to reproduce that row.

**Run RTC on your own images** (RTC env; set the weight paths first):

```bash
conda activate rtc
GEMMA4_PATH=... SD35_PATH=... DIT4SR_Q_PATH=... \
  bash run_dit4sr_with_rtc.sh /path/to/your_LR ./results/my_run   # caption -> SR
```

or as two steps:

```bash
python captioning/caption_gemma4.py --input_dir /path/to/your_LR --output my_captions.json
python DiT4SR/run_dit4sr_from_caption.py --image_path /path/to/your_LR \
    --caption_json my_captions.json --output_dir ./results/my_run/images
```

To reproduce our exact SR images on ReasonText, pass the shipped `captioning/reason_captions.json` to `run_dit4sr_from_caption.py`.

## 📊 GTTCA on your own data

GTTCA scores text at **ground-truth text regions**, so it needs a `--meta-json` of word-level annotations (polygon + rotation/flip + transcript) in the ReasonText format. Supply one in the same shape as `ReasonText-meta.json`; plain unlabeled images cannot be scored. `captioning/` is backbone-independent, so RTC captions can drive any text-conditioned SR model — `DiT4SR/run_dit4sr_from_caption.py` is the reference example.

## 📁 Repository layout

```
captioning/          # RTC Stage 1: MLLM captioning with the reasoning prompt (backbone-independent)
  caption_gemma4.py  ·  reason_prompt.txt  ·  reason_captions.json   # (Gemma-4 example)
DiT4SR/              # RTC Stage 2: DiT4SR backbone + our RTC entry point
  run_dit4sr_from_caption.py  ·  config.py                           # OURS
  pipelines/ · model_dit4sr/ · utils/                                # DiT4SR (unmodified upstream)
gttca/               # GTTCA metric: evaluate_gttca.py · summarize.py · weights_config.json
run_dit4sr_with_rtc.sh  ·  requirements_rtc.txt  ·  requirements_gttca.txt
```

**Ours vs. upstream:** everything is ours except the DiT4SR backbone under `DiT4SR/pipelines/`, `DiT4SR/model_dit4sr/`, `DiT4SR/utils/` ([Adam-duan/DiT4SR](https://github.com/Adam-duan/DiT4SR), unmodified — see `DiT4SR/LICENSE`). RTC is training-free: it only swaps DiT4SR's built-in captioner for a reasoning caption.

## 📝 Citation

If you use this work, please cite our paper (and RealSR-v3 / DRealSR, which ReasonText is built on):

```bibtex
<bibtex — fill in>
```

## 🙏 Acknowledgements

[DiT4SR](https://github.com/Adam-duan/DiT4SR), Stable Diffusion 3.5 (Stability AI), Gemma-4 (Google), PaddleOCR-VL (PaddlePaddle), and the RealSR-v3 / DRealSR datasets.

## ⚖️ License

Released for **non-commercial research use only** under **CC BY-NC 4.0** (see [`LICENSE`](LICENSE)). The bundled DiT4SR backbone keeps its own license (`DiT4SR/LICENSE`); model weights and source datasets are governed by their own terms.
