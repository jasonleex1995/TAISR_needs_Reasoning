<div align="center">

# Reading the Unreadable
### Text-Aware Image Super-Resolution Needs Reasoning

**&lt;AUTHORS — fill in&gt;** · NeurIPS 2026

[📄 Paper](https://example.com) · [🌐 Project Page](https://example.com) · [🤗 ReasonText](https://huggingface.co/datasets/Jasonleex1995/ReasonText) · [🤗 SR Outputs](https://huggingface.co/datasets/Jasonleex1995/ReasonText-SR-outputs) · [⚖️ License](LICENSE)

</div>

Some degraded text **cannot be read from local pixels alone** — humans recover it by *reasoning* over context. This repo releases the two tools from our paper that make that measurable and fixable:

|  | Tool | What it does |
|:--:|---|---|
| 📏 | **GTTCA** | **Metric.** Crop the SR output at the ground-truth text region, deskew with the annotated rotation/flip, OCR, and exact-match against the GT transcript. |
| 🧩 | **RTC** | **Method** (training-free). Run an MLLM with a reasoning prompt on the LR image, then feed the inferred text as a caption to any text-conditioned SR backbone. |

The **ReasonText** benchmark and all SR outputs are on 🤗 Hugging Face (links above).

<p align="center"><img src="assets/rtc.png" width="88%"></p>
<p align="center"><img src="assets/gttca.png" width="88%"></p>

---

## 🔧 Installation

Two environments — the components pin incompatible major versions of `transformers` (Gemma-4 needs ≥ 5.5; PaddleOCR-VL, GTTCA's OCR engine, runs on the 4.x line):

| Env | Covers | Key pins |
|---|---|---|
| **RTC** &nbsp;`requirements_rtc.txt` | `captioning/` + `DiT4SR/` | py 3.11 · torch 2.6 (cu124) · transformers 5.7 · diffusers 0.39 |
| **GTTCA** &nbsp;`requirements_gttca.txt` | `gttca/` | py 3.10 · torch 2.5 (cu121) · transformers 4.55 |

```bash
conda create -n rtc   python=3.11 -y && conda activate rtc   && pip install -r requirements_rtc.txt
conda create -n gttca python=3.10 -y && conda activate gttca && pip install -r requirements_gttca.txt
```

> CUDA wheels come from the PyTorch index (`cu124` / `cu121`) via the `--extra-index-url` line at the top of each requirements file — edit it (e.g. `cu118`) and the `torch` version if your CUDA differs.

## 📦 Weights & Data

Download the weights separately and point the code at them (env vars, or edit `DiT4SR/config.py` / `gttca/weights_config.json`):

| Weight | Used by | Set via |
|---|---|---|
| [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | DiT4SR | `SD35_PATH` |
| [DiT4SR-Q](https://github.com/Adam-duan/DiT4SR) | DiT4SR | `DIT4SR_Q_PATH` |
| [Gemma-4 31B-it](https://huggingface.co/google/gemma-4-31B-it) | captioning | `GEMMA4_PATH` |
| [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | GTTCA | `gttca/weights_config.json` |

The data lives on 🤗 Hugging Face (513-image set = 513 images / 3,985 annotations):

| Data | Contents |
|---|---|
| [🤗 ReasonText](https://huggingface.co/datasets/Jasonleex1995/ReasonText) | LR / HR images + word-level annotations |
| [🤗 SR outputs](https://huggingface.co/datasets/Jasonleex1995/ReasonText-SR-outputs) | Table 1 (12 models) · Table 4 (6 backbones + RTC) |

After download, the GTTCA scorer takes: `LR/` → `--image_path`, `HR/` → `--hr-dir`, `ReasonText-meta.json` → `--meta-json`, and an SR folder → `--sr-dir`.

## ⚡ Quick Start

> **To reproduce the paper's exact numbers, score the released SR outputs** — no regeneration needed. Running the full pipeline yourself drifts slightly (MLLM sampling is stochastic; library/diffusion versions add < ~1 pt).

**▶ Reproduce a Table-4 row** (DiT4SR + RTC) — point `--sr-dir` at any released folder (`sr_outputs/table1/*`, `table4_rtc/*`) for that row:

```bash
conda activate gttca
python gttca/evaluate_gttca.py \
    --sr-dir     /path/to/sr_outputs/table4_rtc/B4_DiT4SR \
    --hr-dir     /path/to/ReasonText/HR \
    --meta-json  /path/to/ReasonText/ReasonText-meta.json \
    --weights-config gttca/weights_config.json \
    --output-dir ./results/DiT4SR_RTC/gttca --model-name DiT4SR_RTC
python gttca/summarize.py --results-root ./results --output-dir ./results/summary
```

**▶ Run RTC on your own images** (set the weight paths first):

```bash
conda activate rtc
GEMMA4_PATH=... SD35_PATH=... DIT4SR_Q_PATH=... \
  bash run_dit4sr_with_rtc.sh /path/to/your_LR ./results/my_run
```

<details><summary>… or as two explicit steps (caption → SR)</summary>

```bash
python captioning/caption_gemma4.py --input_dir /path/to/your_LR --output my_captions.json
python DiT4SR/run_dit4sr_from_caption.py --image_path /path/to/your_LR \
    --caption_json my_captions.json --output_dir ./results/my_run/images
```

To reproduce our exact SR images on ReasonText, pass the shipped `captioning/reason_captions.json` to `run_dit4sr_from_caption.py`.
</details>

## 📁 Repository layout

```
captioning/   RTC Stage 1 — MLLM captioning with the reasoning prompt (backbone-independent)
DiT4SR/       RTC Stage 2 — DiT4SR backbone + our entry point
gttca/        GTTCA metric — evaluate_gttca.py · summarize.py · weights_config.json
```

Everything is ours **except** `DiT4SR/{pipelines,model_dit4sr,utils}/` ([Adam-duan/DiT4SR](https://github.com/Adam-duan/DiT4SR), unmodified — see `DiT4SR/LICENSE`). RTC is training-free: it only swaps DiT4SR's built-in captioner for a reasoning caption, so `captioning/` can drive **any** text-conditioned SR backbone — `run_dit4sr_from_caption.py` is just the reference example. GTTCA needs a `--meta-json` of word-level annotations in the ReasonText format; plain unlabeled images cannot be scored.

## 📝 Citation

If you use this work, please cite our paper (and RealSR-v3 / DRealSR, which ReasonText is built on):

```bibtex
<bibtex — fill in>
```

## 🙏 Acknowledgements

[DiT4SR](https://github.com/Adam-duan/DiT4SR), Stable Diffusion 3.5 (Stability AI), Gemma-4 (Google), PaddleOCR-VL (PaddlePaddle), and the RealSR-v3 / DRealSR datasets.

## ⚖️ License

Released for **non-commercial research use only** under **CC BY-NC 4.0** (see [`LICENSE`](LICENSE)). The bundled DiT4SR backbone keeps its own license (`DiT4SR/LICENSE`); model weights and source datasets are governed by their own terms.
