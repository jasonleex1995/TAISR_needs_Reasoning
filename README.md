<div align="center">

### **Reading the Unreadable**
### Text-Aware Image Super-Resolution Needs Reasoning

NeurIPS 2026

[🌐 Project Page](https://example.com) · [📄 arXiv](https://example.com) · [📝 OpenReview](https://example.com) · [🤗 ReasonText Benchmark](https://huggingface.co/datasets/Jasonleex1995/ReasonText)

</div>

---

## 📏 GTTCA

<p align="center"><img src="assets/gttca.png" width="90%"></p>

GTTCA scores text restoration by OCR-matching the SR output at the ground-truth text region. Reproduce our numbers in three steps.

### 1. Installation

```bash
conda create -n gttca python=3.10 -y && conda activate gttca
pip install -r requirements_gttca.txt
```

### 2. Weights & Data

| Download | For |
|---|---|
| [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | OCR engine — set the path in `gttca/weights_config.json` |
| [🤗 ReasonText](https://huggingface.co/datasets/Jasonleex1995/ReasonText) | `HR/` → `--hr-dir`, `ReasonText-meta.json` → `--meta-json` |
| [🤗 SR outputs](https://huggingface.co/datasets/Jasonleex1995/ReasonText-SR-outputs) | the SR images to score → `--sr-dir` |

### 3. Measure GTTCA on ReasonText

Point `--sr-dir` at any released folder (`sr_outputs/table1/*`, `sr_outputs/table4_rtc/*`) to reproduce that table row:

```bash
python gttca/evaluate_gttca.py \
    --sr-dir     /path/to/sr_outputs/table4_rtc/B4_DiT4SR \
    --hr-dir     /path/to/ReasonText/HR \
    --meta-json  /path/to/ReasonText/ReasonText-meta.json \
    --weights-config gttca/weights_config.json \
    --output-dir ./results/DiT4SR_RTC/gttca --model-name DiT4SR_RTC
python gttca/summarize.py --results-root ./results --output-dir ./results/summary
```

> GTTCA scores text at **ground-truth text regions**, so it needs a `--meta-json` of word-level annotations (polygon + rotation/flip + transcript) in the ReasonText format; plain unlabeled images cannot be scored.

---

## 🧩 DiT4SR + RTC

<p align="center"><img src="assets/rtc.png" width="90%"></p>

RTC is a training-free plug-in: an MLLM reasons over the LR image, and its caption conditions a text-conditioned SR backbone (DiT4SR here). Run it in three steps.

### 1. Installation

```bash
conda create -n rtc python=3.11 -y && conda activate rtc
pip install -r requirements_rtc.txt
```

### 2. Weights & Data

| Download | Set via |
|---|---|
| [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | `SD35_PATH` |
| [DiT4SR-Q](https://github.com/Adam-duan/DiT4SR) | `DIT4SR_Q_PATH` |
| [Gemma-4 31B-it](https://huggingface.co/google/gemma-4-31B-it) | `GEMMA4_PATH` |

Run on your own LR images, or use ReasonText `LR/` to reproduce our results.

### 3. Running DiT4SR + RTC

```bash
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

To score the outputs, use **GTTCA** above. `captioning/` is backbone-independent, so the reasoning caption can drive any text-conditioned SR model — `run_dit4sr_from_caption.py` is the reference example.

---

## 📁 Repository layout

```
captioning/   RTC Stage 1 — MLLM captioning with the reasoning prompt (backbone-independent)
DiT4SR/       RTC Stage 2 — DiT4SR backbone + our entry point
gttca/        GTTCA metric — evaluate_gttca.py · summarize.py · weights_config.json
```

Everything is ours **except** `DiT4SR/{pipelines,model_dit4sr,utils}/` ([Adam-duan/DiT4SR](https://github.com/Adam-duan/DiT4SR), unmodified — see `DiT4SR/LICENSE`). RTC is training-free: it only swaps DiT4SR's built-in captioner for a reasoning caption.

## 📝 Citation

If you use this work, please cite our paper (and RealSR-v3 / DRealSR, which ReasonText is built on):

```bibtex
<bibtex — fill in>
```

## 🙏 Acknowledgements

[DiT4SR](https://github.com/Adam-duan/DiT4SR), Stable Diffusion 3.5 (Stability AI), Gemma-4 (Google), PaddleOCR-VL (PaddlePaddle), and the RealSR-v3 / DRealSR datasets.

## ⚖️ License

Released for **non-commercial research use only** under **CC BY-NC 4.0** (see [`LICENSE`](LICENSE)). The bundled DiT4SR backbone keeps its own license (`DiT4SR/LICENSE`); model weights and source datasets are governed by their own terms.
