"""Stage 3 — aggregate GTTCA per-region JSONLs into per-variant + cross-variant tables.

Mirrors the layout of exp02/eval_metrics.py:
  per-variant ─ <variant_dir>/metrics.json + metrics_details.jsonl is already what evaluate_gttca.py emits
                (text_regions.jsonl). Stage 3 adds <variant_dir>/metrics.json with the breakdown.
  long-format ─ csv/all_metrics.csv      (one row per variant, all metric columns)
  reading-friendly ─ csv/summary_table.csv  (rows = caption_variant × SR model in canonical order,
                                              columns = raw/normalized × Overall/L1/L2/L3)
  publication-ready ─ csv/summary_wide__{raw,normalized}.csv
                                             (rows = Group × Model × Caption Style; cols = Overall/L1/L2/L3.
                                              HR / LR Bicubic baselines pinned to top.)
                       csv/summary_wide_detailed__{raw,normalized}.csv
                                             (same rows; cols = Overall/L1/L2/L2-logic/L2-context/L2-prior/L3.)
  pivots ─ csv/{overall,levels,reasoning}/<metric>.csv  (rows = caption_variant, cols = SR model)

Match modes (terminology):
  raw        — direct string equality (label == prediction). The `exact_match_strict`
               field in text_regions.jsonl.
  normalized — NFKC + casefold + all-whitespace stripped, then compared. The
               `exact_match_ignore_case_whitespace` field. The numbers reported
               in the paper (e.g. HR upper-bound 93.4%) come from this mode.

Variant directory naming:
  exp01: variant = SR model only            -> sr_model="<variant>", caption_variant="baseline"
  exp03: variant = "<sr_model>__<caption_tag>__<caption_mode>"
         (e.g. B3_FaithDiff__gemma4-31b__reason_v6__full)
         -> sr_model=parts[0], caption_variant="__".join(parts[1:]) (e.g. "gemma4-31b__reason_v6__full")
  baselines: variant directories named "HR", "LR", "LR_Bicubic", "LR-Bicubic" are recognized as
             upper/lower bound baselines and rendered as standalone rows at the top of summary_wide.

Each text_regions.jsonl row is parsed for:
  text_difficulty   ∈ {Level 1, Level 2, Level 3}
  text_rotation     ∈ {0, 90, 180, 270}
  reason_types      list of strings; for Level 2, canonicalized to {logical reasoning, image context, prior knowledge}
  exact_match_strict, exact_match_ignore_case_whitespace  ∈ {0, 1}
  status            "ok" / "error" (errors counted toward total but always exact_match=0)

Metrics:
  Per slice S, per matching mode M ∈ {raw, normalized}:
      acc(S, M) = mean(exact_match_M) over rows in S, ×100.

Slices reported:
  Overall, Level 1, Level 2, Level 3, Level 1+2,
  Level 2 by reasoning: logical reasoning, image context, prior knowledge.

CSV pivot conventions (mirror exp02 single-orientation rule):
  Rows = caption_variant (exp01 → "baseline"; exp03 → "<caption_tag>__<caption_mode>")
  Cols = sr_model in canonical order: B2_SUPIR, B3_FaithDiff, B4_DiT4SR, C3_OSEDiff, E1_TeReDiff, E2_UniT
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


# ---------- canonical orderings (mirror exp02/eval_metrics.py) ----------

SR_MODEL_CANONICAL_ORDER = [
    "B2_SUPIR", "B3_FaithDiff", "B4_DiT4SR", "C3_OSEDiff", "E1_TeReDiff", "E2_UniT",
]

REASON_TYPE_MAP = {
    "context": "image context",
    "logic":   "logical reasoning",
    "prior":   "prior knowledge",
}
# Canonical display order for Level 2 reasoning types — used in print, JSON, CSV pivots,
# the long-format CSV, and the summary_wide_detailed sub-columns. Pinned by the user.
REASONING_ORDER = ["logical reasoning", "image context", "prior knowledge"]
REASONING_SHORT = {"logical reasoning": "logic", "image context": "context", "prior knowledge": "prior"}
REASONING_SHORT_ORDER = [REASONING_SHORT[rt] for rt in REASONING_ORDER]  # ["logic", "context", "prior"]

LEVEL_VALUES = ("Level 1", "Level 2", "Level 3")
LEVEL_SHORTS = {"Level 1": "L1", "Level 2": "L2", "Level 3": "L3"}

MATCH_MODES = ("raw", "normalized")
MATCH_MODE_TO_KEY = {
    # `raw`        — direct string equality (`exact_match_strict` in the JSONL)
    # `normalized` — NFKC + casefold + whitespace-stripped equality
    #                (`exact_match_ignore_case_whitespace` in the JSONL)
    "raw":        "exact_match_strict",
    "normalized": "exact_match_ignore_case_whitespace",
}


# ---------- variant naming ----------


def parse_variant_name(variant: str) -> tuple[str, str]:
    """Split exp01/exp03 variant directory name into (sr_model, caption_variant).

    Examples:
        "B3_FaithDiff"                                  -> ("B3_FaithDiff", "baseline")
        "B3_FaithDiff__gemma4-31b__reason_v6__full"     -> ("B3_FaithDiff", "gemma4-31b__reason_v6__full")
        "B3_FaithDiff__gemma4-31b__reason_v6__textonly" -> ("B3_FaithDiff", "gemma4-31b__reason_v6__textonly")
    """
    parts = variant.split("__")
    if len(parts) == 1:
        return variant, "baseline"
    return parts[0], "__".join(parts[1:])


def _has_hidden_component(path: Path, base: Path) -> bool:
    """Return True if `path` contains any component (relative to `base`) starting with '.'.

    Skips junk like `.ipynb_checkpoints/`, `.DS_Store`, `.git/`, etc. when walking
    a results tree recursively.
    """
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = path
    return any(part.startswith(".") for part in rel.parts)


def discover_jsonl_paths(roots: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.name == "text_regions.jsonl":
            paths.append(root)
            continue
        if root.is_dir():
            paths.extend(
                sorted(
                    p for p in root.rglob("text_regions.jsonl")
                    if not _has_hidden_component(p, root)
                )
            )
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(p)
    return deduped


def variant_info_from_path(jsonl_path: Path) -> tuple[str, str, str, str]:
    """Return (exp, variant, sr_model, caption_variant) inferred from path layout.

    Expected layout: <results-root>/<exp>/<variant>/text_regions.jsonl
    """
    variant_dir = jsonl_path.parent
    exp_dir = variant_dir.parent
    return exp_dir.name, variant_dir.name, *parse_variant_name(variant_dir.name)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------- aggregation ----------


def canonicalize_reason_types(raw_types: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in raw_types or ():
        key = REASON_TYPE_MAP.get(str(raw).strip().lower(), str(raw).strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def safe_pct(num: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return round(100.0 * num / denom, 4)


def acc_for(rows: list[dict[str, Any]], match_key: str) -> dict[str, Any]:
    n_total = len(rows)
    n_correct = sum(1 for r in rows if int(r.get(match_key, 0) or 0) == 1)
    return {"correct": n_correct, "total": n_total, "value": safe_pct(n_correct, n_total)}


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the full per-variant metrics breakdown."""
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_err = n_total - n_ok

    # Slice rows once.
    by_level: dict[str, list[dict[str, Any]]] = {lv: [] for lv in LEVEL_VALUES}
    level12: list[dict[str, Any]] = []
    by_reasoning: dict[str, list[dict[str, Any]]] = {rt: [] for rt in REASONING_ORDER}

    for row in rows:
        diff = row.get("text_difficulty")
        if diff in by_level:
            by_level[diff].append(row)
        if diff in ("Level 1", "Level 2"):
            level12.append(row)
        if diff == "Level 2":
            for rtype in canonicalize_reason_types(row.get("reason_types") or ()):
                if rtype in by_reasoning:
                    by_reasoning[rtype].append(row)

    def slice_metrics(slice_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {mode: acc_for(slice_rows, MATCH_MODE_TO_KEY[mode]) for mode in MATCH_MODES}

    return {
        "n_total":   n_total,
        "n_ok":      n_ok,
        "n_error":   n_err,
        "overall":   slice_metrics(rows),
        "by_level": {
            "Level 1":   slice_metrics(by_level["Level 1"]),
            "Level 2":   slice_metrics(by_level["Level 2"]),
            "Level 3":   slice_metrics(by_level["Level 3"]),
            "Level 1+2": slice_metrics(level12),
        },
        "level2_by_reasoning": {
            rt: slice_metrics(by_reasoning[rt]) for rt in REASONING_ORDER
        },
    }


# ---------- aggregate CSV outputs ----------


def variant_row_for_long(summary: dict[str, Any]) -> dict[str, Any]:
    """One CSV row per variant with every (slice × mode) accuracy column."""
    row: dict[str, Any] = {
        "exp":             summary["exp"],
        "variant":         summary["variant"],
        "sr_model":        summary["sr_model"],
        "caption_variant": summary["caption_variant"],
        "n_total":         summary["n_total"],
        "n_ok":            summary["n_ok"],
        "n_error":         summary["n_error"],
    }
    for mode in MATCH_MODES:
        row[f"overall_{mode}_acc"] = summary["overall"][mode]["value"]
        row[f"overall_{mode}_correct"] = summary["overall"][mode]["correct"]
        row[f"overall_{mode}_total"] = summary["overall"][mode]["total"]
    for level_full in ("Level 1", "Level 2", "Level 3", "Level 1+2"):
        short = "L12" if level_full == "Level 1+2" else LEVEL_SHORTS[level_full]
        for mode in MATCH_MODES:
            cell = summary["by_level"][level_full][mode]
            row[f"{short}_{mode}_acc"] = cell["value"]
            row[f"{short}_{mode}_correct"] = cell["correct"]
            row[f"{short}_{mode}_total"] = cell["total"]
    for rtype in REASONING_ORDER:
        short = REASONING_SHORT[rtype]
        for mode in MATCH_MODES:
            cell = summary["level2_by_reasoning"][rtype][mode]
            row[f"L2_{short}_{mode}_acc"] = cell["value"]
            row[f"L2_{short}_{mode}_correct"] = cell["correct"]
            row[f"L2_{short}_{mode}_total"] = cell["total"]
    return row


LONG_CSV_COLUMNS = (
    ["exp", "variant", "sr_model", "caption_variant", "n_total", "n_ok", "n_error"]
    + [f"overall_{m}_acc" for m in MATCH_MODES]
    + [f"overall_{m}_correct" for m in MATCH_MODES]
    + [f"overall_{m}_total" for m in MATCH_MODES]
    + [f"{lv}_{m}_acc" for lv in ("L1", "L2", "L3", "L12") for m in MATCH_MODES]
    + [f"{lv}_{m}_correct" for lv in ("L1", "L2", "L3", "L12") for m in MATCH_MODES]
    + [f"{lv}_{m}_total" for lv in ("L1", "L2", "L3", "L12") for m in MATCH_MODES]
    + [f"L2_{rt}_{m}_acc" for rt in ("logic", "context", "prior") for m in MATCH_MODES]
    + [f"L2_{rt}_{m}_correct" for rt in ("logic", "context", "prior") for m in MATCH_MODES]
    + [f"L2_{rt}_{m}_total" for rt in ("logic", "context", "prior") for m in MATCH_MODES]
)


def write_long_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = [variant_row_for_long(s) for s in summaries]
    rows.sort(key=lambda r: (r["exp"], r["caption_variant"], r["sr_model"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LONG_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# Reading-friendly summary table:
#   rows = (caption_variant × sr_model in canonical order)
#   columns = raw/normalized × Overall/L1/L2/L3
SUMMARY_TABLE_COLUMNS = (
    ["caption_variant", "SR_model"]
    + [f"{m}_{slice_}" for m in MATCH_MODES for slice_ in ("Overall", "L1", "L2", "L3", "L12")]
)


def write_summary_table(path: Path, summaries: list[dict[str, Any]]) -> None:
    by_combo = {(s["sr_model"], s["caption_variant"]): s for s in summaries}
    caption_variants_sorted = sorted({s["caption_variant"] for s in summaries},
                                     key=lambda v: (v != "baseline", v))
    sr_models_in_data = {s["sr_model"] for s in summaries}
    extra_models = sorted(sr_models_in_data - set(SR_MODEL_CANONICAL_ORDER))
    sr_model_order = [m for m in SR_MODEL_CANONICAL_ORDER if m in sr_models_in_data] + extra_models

    def cell(value: Any) -> str:
        return "" if value is None else str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_TABLE_COLUMNS)
        for caption_variant in caption_variants_sorted:
            for sr_model in sr_model_order:
                summary = by_combo.get((sr_model, caption_variant))
                row = [caption_variant, sr_model]
                if summary is None:
                    row.extend([""] * (len(SUMMARY_TABLE_COLUMNS) - 2))
                    writer.writerow(row)
                    continue
                for mode in MATCH_MODES:
                    row.append(cell(summary["overall"][mode]["value"]))
                    row.append(cell(summary["by_level"]["Level 1"][mode]["value"]))
                    row.append(cell(summary["by_level"]["Level 2"][mode]["value"]))
                    row.append(cell(summary["by_level"]["Level 3"][mode]["value"]))
                    row.append(cell(summary["by_level"]["Level 1+2"][mode]["value"]))
                writer.writerow(row)


# Pivots: rows = caption_variant, cols = sr_model. One file per (category × metric).


def write_pivot(path: Path, summaries: list[dict[str, Any]],
                getter) -> None:
    by_combo = {(s["sr_model"], s["caption_variant"]): s for s in summaries}
    caption_variants_sorted = sorted({s["caption_variant"] for s in summaries},
                                     key=lambda v: (v != "baseline", v))
    sr_models_in_data = {s["sr_model"] for s in summaries}
    extra_models = sorted(sr_models_in_data - set(SR_MODEL_CANONICAL_ORDER))
    sr_model_order = [m for m in SR_MODEL_CANONICAL_ORDER if m in sr_models_in_data] + extra_models

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["caption_variant"] + sr_model_order)
        for cv in caption_variants_sorted:
            row = [cv]
            for m in sr_model_order:
                s = by_combo.get((m, cv))
                if s is None:
                    row.append("")
                else:
                    v = getter(s)
                    row.append("" if v is None else v)
            writer.writerow(row)


# ---------- publication-ready wide CSV (Group × Model × Caption Style x slices) ----------

# SR-model dir names use a "<group_letter><digit>_<friendly_name>" convention
# (e.g. B2_SUPIR -> Group B, friendly "SUPIR"). The trailing fallback covers
# anything that doesn't fit the pattern.
SR_MODEL_PREFIX_RE = re.compile(r"^([A-Z])\d+_(.+)$")

# Variant-name patterns used when the SR output isn't a model run but a baseline.
# Matched case-insensitively, separators normalised.
BASELINE_VARIANTS = {
    "HR":         "HR",
    "LR":         "LR Bicubic",
    "LR_BICUBIC": "LR Bicubic",
    "LRBICUBIC":  "LR Bicubic",
}


def parse_sr_model_for_display(sr_model: str) -> tuple[str, str]:
    """Return (group_letter, friendly_model_name).

    Examples:
        "B2_SUPIR"      -> ("B", "SUPIR")
        "B3_FaithDiff"  -> ("B", "FaithDiff")
        "C3_OSEDiff"    -> ("C", "OSEDiff")
        "E1_TeReDiff"       -> ("E", "TeReDiff")
        "WeirdModel"    -> ("",  "WeirdModel")  # passthrough fallback
    """
    m = SR_MODEL_PREFIX_RE.match(sr_model)
    if m:
        return m.group(1), m.group(2)
    return "", sr_model


def baseline_display_name(variant: str) -> str | None:
    """If `variant` looks like an HR / LR baseline, return its display name; else None."""
    norm = variant.upper().replace("-", "_")
    return BASELINE_VARIANTS.get(norm)


_TEXT_MODE_ALIASES = {"text", "textonly", "texton1y", "txt"}  # any spelling that means text-only mode


def _normalize_mode(mode_raw: str) -> str:
    """Collapse mode spellings so 'text_only', 'textonly', 'TEXT-ONLY' all become 'text'.
    Other modes (full, etc.) are passed through with underscores turned into spaces.
    """
    canonical = mode_raw.replace("_", "").replace("-", "").lower()
    if canonical in _TEXT_MODE_ALIASES:
        return "text"
    return mode_raw.replace("_", " ")


def humanize_caption_variant(caption_variant: str) -> str:
    """Render a caption_variant as the 'Caption Style' cell.

    Mapping:
        "baseline"                                 -> "Original"   (exp01 — model's own caption)
        "<caption_model>__<prompt_tag>__<mode>"    -> "<prompt_tag w/ underscores spaced> - <mode>"
                                                       where any spelling of text-only is collapsed to "text"
        "<caption_model>__<prompt_tag>"            -> "<prompt_tag w/ underscores spaced>"
        anything else                              -> caption_variant verbatim with __ -> ' - '

    Examples:
        "baseline"                              -> "Original"
        "gemma4-31b__reason_v1__full"           -> "reason v1 - full"
        "gemma4-31b__reason_v1__text_only"      -> "reason v1 - text"
        "gemma4-31b__reason_v1__textonly"       -> "reason v1 - text"
        "gemma4-31b__noreason_v2__full"         -> "noreason v2 - full"
    """
    if caption_variant == "baseline":
        return "Original"
    parts = caption_variant.split("__")
    if len(parts) == 3:
        prompt_tag = parts[1].replace("_", " ")
        return f"{prompt_tag} - {_normalize_mode(parts[2])}"
    if len(parts) == 2:
        return parts[1].replace("_", " ")
    return caption_variant.replace("__", " - ").replace("_", " ")


def caption_style_sort_key(style: str) -> tuple[int, str, int]:
    """Sort caption styles consistently:
        Original first, then prompt-style rows (text before full), then plain ones, GT last.
    """
    if style == "Original":
        return (0, "", 0)
    if style.lower().startswith("gt"):
        return (3, style.lower(), 0)
    if " - " in style:
        tag, _, mode = style.rpartition(" - ")
        mode_key = 0 if mode.lower().startswith("text") else 1
        return (1, tag.lower(), mode_key)
    return (2, style.lower(), 0)


# A "slice spec" is (display_label, getter). getter(summary_dict) -> (value, total).
# Two flavours below: basic (Overall/L1/L2/L3) and detailed (with L2 reasoning sub-cols).


def _basic_slice_specs(mode: str) -> list[tuple[str, Any]]:
    def overall(s):
        return s["overall"][mode]["value"], int(s["overall"][mode]["total"])

    def level(name):
        return lambda s: (
            s["by_level"][name][mode]["value"],
            int(s["by_level"][name][mode]["total"]),
        )

    return [
        ("Overall", overall),
        ("Level 1", level("Level 1")),
        ("Level 2", level("Level 2")),
        ("Level 3", level("Level 3")),
    ]


def _detailed_slice_specs(mode: str) -> list[tuple[str, Any]]:
    """Same as _basic_slice_specs but with the three Level-2 reasoning sub-slices
    inserted right after Level 2 (logical reasoning -> image context -> prior knowledge)."""
    def overall(s):
        return s["overall"][mode]["value"], int(s["overall"][mode]["total"])

    def level(name):
        return lambda s: (
            s["by_level"][name][mode]["value"],
            int(s["by_level"][name][mode]["total"]),
        )

    def reasoning(rt):
        return lambda s: (
            s["level2_by_reasoning"][rt][mode]["value"],
            int(s["level2_by_reasoning"][rt][mode]["total"]),
        )

    return [
        ("Overall", overall),
        ("Level 1", level("Level 1")),
        ("Level 2", level("Level 2")),
        *[
            (f"Level 2 {rt}", reasoning(rt))
            for rt in REASONING_ORDER  # logical reasoning -> image context -> prior knowledge
        ],
        ("Level 3", level("Level 3")),
    ]


def _write_wide_csv(path: Path, summaries: list[dict[str, Any]],
                    slice_specs: list[tuple[str, Any]]) -> None:
    """Shared body for the wide summary CSVs.

    Layout:
        Header: Group | Model | Caption Style | <slice_label> (n) | ...
        Rows:   HR baseline (if any) | LR Bicubic baseline (if any) | per (Group, Model,
                Caption Style) sorted by group letter -> SR model canonical order -> caption style key.
    """
    # Per-slice totals (max across variants so partial runs don't shrink the header).
    slice_totals: list[int] = []
    for _, getter in slice_specs:
        slice_totals.append(max((getter(s)[1] for s in summaries), default=0))

    header = ["Group", "Model", "Caption Style"] + [
        f"{label} ({total})" for (label, _), total in zip(slice_specs, slice_totals)
    ]

    # Bucket: baseline rows vs (group, model, caption_style) rows.
    baseline_rows: dict[str, dict[str, Any]] = {}
    main_rows: list[tuple[str, str, str, str, dict[str, Any]]] = []
    for s in summaries:
        bname = baseline_display_name(s["variant"])
        if bname is not None:
            baseline_rows.setdefault(bname, s)
            continue
        group, friendly_model = parse_sr_model_for_display(s["sr_model"])
        caption_style = humanize_caption_variant(s["caption_variant"])
        main_rows.append((group, s["sr_model"], friendly_model, caption_style, s))

    canonical_index = {m: i for i, m in enumerate(SR_MODEL_CANONICAL_ORDER)}

    def main_row_key(item):
        group, sr_model, _friendly, caption_style, _s = item
        model_idx = canonical_index.get(sr_model, len(canonical_index) + ord(sr_model[0]))
        return (group, model_idx, caption_style_sort_key(caption_style))

    main_rows.sort(key=main_row_key)

    def fmt(value: Any) -> str:
        return "" if value is None else str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for bname in ("HR", "LR Bicubic"):
            if bname not in baseline_rows:
                continue
            s = baseline_rows[bname]
            row = ["", "", bname] + [fmt(getter(s)[0]) for _, getter in slice_specs]
            writer.writerow(row)
        for group, _sr, friendly_model, caption_style, s in main_rows:
            row = [group, friendly_model, caption_style] + [
                fmt(getter(s)[0]) for _, getter in slice_specs
            ]
            writer.writerow(row)


def write_wide_summary_csv(path: Path, summaries: list[dict[str, Any]], mode: str) -> None:
    """Basic publication table: Overall | Level 1 | Level 2 | Level 3."""
    _write_wide_csv(path, summaries, _basic_slice_specs(mode))


def write_wide_detailed_summary_csv(path: Path, summaries: list[dict[str, Any]], mode: str) -> None:
    """Detailed publication table: Overall | Level 1 | Level 2 | L2 logical reasoning |
    L2 image context | L2 prior knowledge | Level 3."""
    _write_wide_csv(path, summaries, _detailed_slice_specs(mode))


def build_pivot_specs() -> dict[str, list[tuple[str, Any]]]:
    """Return {category -> [(metric_filename_stem, getter), ...]}."""
    specs: dict[str, list[tuple[str, Any]]] = {"overall": [], "levels": [], "reasoning": []}

    def make_overall(mode: str):
        return lambda s: s["overall"][mode]["value"]

    def make_level(level_full: str, mode: str):
        return lambda s: s["by_level"][level_full][mode]["value"]

    def make_reasoning(rtype_full: str, mode: str):
        return lambda s: s["level2_by_reasoning"][rtype_full][mode]["value"]

    for mode in MATCH_MODES:
        specs["overall"].append((mode, make_overall(mode)))
    for level_full in ("Level 1", "Level 2", "Level 3", "Level 1+2"):
        short = "L12" if level_full == "Level 1+2" else LEVEL_SHORTS[level_full]
        for mode in MATCH_MODES:
            specs["levels"].append((f"{short}__{mode}", make_level(level_full, mode)))
    for rtype in REASONING_ORDER:
        short = REASONING_SHORT[rtype]
        for mode in MATCH_MODES:
            specs["reasoning"].append((f"{short}__{mode}", make_reasoning(rtype, mode)))
    return specs


# ---------- pretty printing ----------


def fmt_pct(v: Any) -> str:
    if v is None:
        return "  N/A "
    return f"{float(v):6.2f}%"


def print_variant(summary: dict[str, Any]) -> None:
    head = f"{summary['exp']}/{summary['variant']}"
    n = summary["n_total"]
    err = summary["n_error"]
    print(f"\n=== {head}  (n={n}, errors={err}) ===")
    for mode in MATCH_MODES:
        print(f"  [{mode}]")
        ov = summary["overall"][mode]["value"]
        print(f"    overall          : acc={fmt_pct(ov)}")
        for lv in ("Level 1", "Level 2", "Level 3", "Level 1+2"):
            v = summary["by_level"][lv][mode]["value"]
            print(f"    {lv:16s}: acc={fmt_pct(v)}")
        for rt in REASONING_ORDER:
            v = summary["level2_by_reasoning"][rt][mode]["value"]
            print(f"    L2/{rt:18s}: acc={fmt_pct(v)}")


# ---------- main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, action="append", default=None,
                        help="Directory under which to discover text_regions.jsonl recursively. "
                             "Pass multiple times to combine. Typical: "
                             "/path/to/gttca_results")
    parser.add_argument("--jsonl-paths", type=Path, nargs="*", default=None,
                        help="Explicit list of text_regions.jsonl paths (in addition to --results-root).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write csv/{all_metrics.csv, summary_table.csv, "
                             "<category>/<metric>.csv} and the printed log.")
    parser.add_argument("--quiet", action="store_true",
                        help="Skip per-variant pretty print to stdout.")
    args = parser.parse_args()

    if not args.results_root and not args.jsonl_paths:
        parser.error("Provide --results-root and/or --jsonl-paths.")

    paths: list[Path] = []
    if args.results_root:
        paths.extend(discover_jsonl_paths(args.results_root))
    if args.jsonl_paths:
        paths.extend(args.jsonl_paths)
    paths = discover_jsonl_paths(paths)
    if not paths:
        raise SystemExit("No text_regions.jsonl files found.")

    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Stage 3: discovered {len(paths)} text_regions.jsonl file(s) ===")
    if args.results_root:
        for r in args.results_root:
            print(f"    --results-root {r}")
    print()

    summaries: list[dict[str, Any]] = []
    skipped: list[Path] = []
    for path in paths:
        rows = load_jsonl(path)
        if not rows:
            skipped.append(path)
            print(f"[skip] empty jsonl: {path}")
            continue
        exp, variant, sr_model, caption_variant = variant_info_from_path(path)
        metrics = compute_metrics(rows)
        summary = {
            "jsonl_path":      str(path),
            "exp":             exp,
            "variant":         variant,
            "sr_model":        sr_model,
            "caption_variant": caption_variant,
            **metrics,
        }
        summaries.append(summary)
        # Per-variant metrics.json next to text_regions.jsonl
        out_metrics = path.parent / "metrics.json"
        out_metrics.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.quiet:
            print_variant(summary)

    # Quick (exp, variant) audit so the user can spot anything missing.
    print()
    by_exp: dict[str, list[str]] = {}
    for s in summaries:
        by_exp.setdefault(s["exp"], []).append(s["variant"])
    for exp in sorted(by_exp):
        variants = sorted(by_exp[exp])
        print(f"  [{exp}] {len(variants)} variant(s):")
        for v in variants:
            print(f"      {v}")
    if skipped:
        print(f"  [skipped — empty jsonl] {len(skipped)} file(s):")
        for p in skipped:
            print(f"      {p}")

    # Long-format CSV.
    long_path = csv_dir / "all_metrics.csv"
    write_long_csv(long_path, summaries)

    # Reading-friendly summary table.
    summary_table_path = csv_dir / "summary_table.csv"
    write_summary_table(summary_table_path, summaries)

    # Publication-ready wide CSVs (Group × Model × Caption Style x slices).
    # One pair per matching mode (raw + normalized).
    #   summary_wide__<mode>.csv          : Overall | L1 | L2 | L3
    #   summary_wide_detailed__<mode>.csv : Overall | L1 | L2 | L2 reasoning sub-cols | L3
    for mode in MATCH_MODES:
        write_wide_summary_csv(csv_dir / f"summary_wide__{mode}.csv", summaries, mode)
        write_wide_detailed_summary_csv(csv_dir / f"summary_wide_detailed__{mode}.csv", summaries, mode)

    # Pivots.
    specs = build_pivot_specs()
    pivot_count = 0
    for category, items in specs.items():
        cat_dir = csv_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for metric_name, getter in items:
            write_pivot(cat_dir / f"{metric_name}.csv", summaries, getter)
            pivot_count += 1

    print(f"\n=== Stage 3 done. {len(summaries)} variants aggregated -> {csv_dir}/ ===")
    print(f"  all_metrics.csv         (long format, every metric column)")
    print(f"  summary_table.csv       (caption_variant × SR_model × raw/normalized × Overall/L1/L2/L3/L12)")
    for mode in MATCH_MODES:
        print(f"  summary_wide__{mode}.csv           (Group × Model × Caption Style × Overall/L1/L2/L3 — paper-ready)")
        print(f"  summary_wide_detailed__{mode}.csv  (same + L2 reasoning sub-cols: logical reasoning, image context, prior knowledge)")
    for category in specs:
        print(f"  {category}/")
        for p in sorted((csv_dir / category).glob("*.csv")):
            print(f"      {p.name}")


if __name__ == "__main__":
    main()
