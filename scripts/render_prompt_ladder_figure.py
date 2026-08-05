#!/usr/bin/env python3
"""Render color-coded figures for incremental prompt ladders."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from llm_eval_prompts import PROMPT_VARIANTS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

# Internal prompt IDs (llm_eval_prompts.py / run dirs like llm-suite__p16) → paper figure labels.
# Loading and diffs still use internal IDs; only titles, subtitles, and legends use these labels.
#
# Internal  Paper  Role in the Gemma deep-dive ladder
# ------    -----  -----------------------------------
# 16        0      minimal generic checker (base)
# 0         1      radiologist + oncology FDG PET/CT
# 2         2      error examples on the false JSON branch
# 3         3      parenthetical examples on the false branch
# 8         4      precision / false-alarm instruction
# 12        5      review-protocol block
# 13        6      refined protocol (final prompt text shown)
PROMPT_DISPLAY_LABELS: dict[str, str] = {
    "16": "0",
    "0": "1",
    "2": "2",
    "3": "3",
    "8": "4",
    "12": "5",
    "13": "6",
}

# Full ladder used as the canonical color reference for any figure ending at prompt 13.
CANONICAL_COLOR_LADDER_IDS: tuple[str, ...] = ("16", "0", "2", "3", "8", "12", "13")

FONT_SIZE = 8.6
FONT_FAMILY = "DejaVu Sans Mono"
LABEL_SIZE = 9.0
TITLE_SIZE = 14.0

BOX_X = 0.06
BOX_W = 0.88
PAD_X = 0.018
PAD_Y = 0.014
LINE_HEIGHT = 0.0175

LAYER_PALETTE = [
    {"face": "#FFFFFF", "edge": "#CCCCCC", "text": "#1A1A1A"},
    {"face": "#7CB9FF", "edge": "#1D6FD4", "text": "#0A2E5C"},
    {"face": "#FFB366", "edge": "#D97706", "text": "#5C2D00"},
    {"face": "#B5E8B5", "edge": "#2E8B57", "text": "#1A3D1A"},
    {"face": "#DDA0DD", "edge": "#8B008B", "text": "#4A004A"},
    {"face": "#FFD700", "edge": "#B8860B", "text": "#5C4A00"},
    {"face": "#FFB6C1", "edge": "#DC143C", "text": "#5C1A24"},
    {"face": "#87CEEB", "edge": "#4682B4", "text": "#1A3A4A"},
]


@dataclass(frozen=True)
class PromptLadder:
    name: str
    prompt_ids: tuple[str, ...]
    output_filename: str
    display_labels: tuple[str, ...] | None = None
    title: str | None = None
    subtitle: str | None = None
    # When False, diff only this ladder's prompt_ids but reuse deep-dive palette slots.
    full_ladder_colors: bool = False


def resolve_display_labels(
    prompt_ids: tuple[str, ...],
    display_labels: tuple[str, ...] | None = None,
    *,
    use_internal_ids: bool = False,
) -> tuple[str, ...]:
    if use_internal_ids:
        return prompt_ids
    if display_labels is not None:
        if len(display_labels) != len(prompt_ids):
            raise ValueError(
                f"display_labels length {len(display_labels)} "
                f"!= prompt_ids length {len(prompt_ids)}"
            )
        return display_labels
    return tuple(PROMPT_DISPLAY_LABELS.get(pid, pid) for pid in prompt_ids)


def format_prompt_chain(labels: tuple[str, ...]) -> str:
    return " → ".join(labels)


def default_title(ladder: PromptLadder, display_labels: tuple[str, ...]) -> str:
    chain = format_prompt_chain(display_labels)
    if ladder.name == "gemma_deep_dive":
        return f"Gemma deep-dive prompt ladder (prompts {chain})"
    return f"Incremental LLM prompt composition (prompts {chain})"


def default_subtitle(display_labels: tuple[str, ...]) -> str:
    final = display_labels[-1]
    if len(display_labels) == 3:
        return (
            f"Prompt {final} text; green = added in prompt {display_labels[1]}, "
            f"pink = added in prompt {display_labels[2]}."
        )
    return (
        f"Full prompt {final} composition; each color marks text "
        "introduced at that ladder step."
    )


LADDERS: dict[str, PromptLadder] = {
    "paper": PromptLadder(
        name="paper",
        prompt_ids=("0", "3", "13"),
        # Short 3-step paper figure: relabel as 0 → 1 → 2 (not the global 1 → 3 → 6 map).
        display_labels=("0", "1", "2"),
        output_filename="prompt_ladder_0_1_2.png",
    ),
    "gemma_deep_dive": PromptLadder(
        name="gemma_deep_dive",
        prompt_ids=("16", "0", "2", "3", "8", "12", "13"),
        full_ladder_colors=True,
        output_filename="prompt_ladder_gemma_deep_dive.png",
    ),
    "legacy_2_3_13": PromptLadder(
        name="legacy_2_3_13",
        prompt_ids=("2", "3", "13"),
        display_labels=("2", "3", "13"),
        output_filename="prompt_ladder_2_3_13.png",
    ),
}


@dataclass(frozen=True)
class Segment:
    text: str
    layer: str


def display_template(template: str) -> str:
    return template.replace("{{", "{").replace("}}", "}")


def layer_for_prompt_id(prompt_id: str) -> str:
    """Map an internal prompt ID to a stable layer key (shared across all figures)."""
    paper_label = PROMPT_DISPLAY_LABELS.get(prompt_id, prompt_id)
    return "base" if paper_label == "0" else f"layer_{paper_label}"


def layer_style(layer: str) -> dict[str, str]:
    if layer == "base":
        return LAYER_PALETTE[0]
    step = int(layer.split("_", 1)[1])
    return LAYER_PALETTE[min(step, len(LAYER_PALETTE) - 1)]


def segments_to_text(segments: list[Segment]) -> str:
    return "".join(segment.text for segment in segments)


def slice_segments_by_char_range(
    segments: list[Segment], start: int, end: int
) -> list[Segment]:
    result: list[Segment] = []
    pos = 0
    for segment in segments:
        seg_end = pos + len(segment.text)
        if seg_end <= start:
            pos = seg_end
            continue
        if pos >= end:
            break
        cut_start = max(0, start - pos)
        cut_end = min(len(segment.text), end - pos)
        result.append(Segment(segment.text[cut_start:cut_end], segment.layer))
        pos = seg_end
    return result


def diff_replace_preserving_layers(
    old_slice: list[Segment],
    old_sub: str,
    new_sub: str,
    insert_layer: str,
) -> list[Segment]:
    """Diff a replace region while keeping prior layer colors on unchanged text."""
    if not old_sub:
        return [Segment(new_sub, insert_layer)] if new_sub else []
    if not new_sub:
        return []

    sub_matcher = SequenceMatcher(None, old_sub, new_sub)
    replaced: list[Segment] = []
    for sub_op, si1, si2, sj1, sj2 in sub_matcher.get_opcodes():
        if sub_op == "equal":
            replaced.extend(slice_segments_by_char_range(old_slice, si1, si2))
        elif sub_op in {"insert", "replace"}:
            replaced.append(Segment(new_sub[sj1:sj2], insert_layer))
    return replaced


def evolve_segments(
    segments: list[Segment], new_text: str, insert_layer: str
) -> list[Segment]:
    old_text = segments_to_text(segments)
    if old_text == new_text:
        return segments

    matcher = SequenceMatcher(None, old_text, new_text)
    result: list[Segment] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            result.extend(slice_segments_by_char_range(segments, i1, i2))
        elif op == "insert":
            result.append(Segment(new_text[j1:j2], insert_layer))
        elif op == "replace":
            old_slice = slice_segments_by_char_range(segments, i1, i2)
            result.extend(
                diff_replace_preserving_layers(
                    old_slice,
                    old_text[i1:i2],
                    new_text[j1:j2],
                    insert_layer,
                )
            )
    return result


def build_layered_segments(
    prompt_ids: tuple[str, ...],
    field: str,
    *,
    full_ladder_colors: bool = False,
) -> list[Segment]:
    """Build layered segments.

    full_ladder_colors=True walks the full deep-dive ladder for coloring.
    Otherwise only diffs this ladder's prompt_ids, but layer keys still map to
  the shared palette (e.g. internal p3 → green, p13 → pink).
    """
    color_ids = (
        CANONICAL_COLOR_LADDER_IDS if full_ladder_colors else prompt_ids
    )

    first = PROMPT_VARIANTS[color_ids[0]]
    if field == "system":
        initial = first.system_prompt
    else:
        initial = display_template(first.user_prompt_template)

    segments = [Segment(initial, "base")]
    for prompt_id in color_ids[1:]:
        variant = PROMPT_VARIANTS[prompt_id]
        new_text = (
            variant.system_prompt
            if field == "system"
            else display_template(variant.user_prompt_template)
        )
        segments = evolve_segments(segments, new_text, layer_for_prompt_id(prompt_id))
    return segments


def build_legend(
    display_labels: tuple[str, ...],
    prompt_ids: tuple[str, ...],
) -> list[tuple[str, str]]:
    entries = [("base", f"Base prompt ({display_labels[0]})")]
    for prompt_id, label in zip(prompt_ids[1:], display_labels[1:]):
        entries.append(
            (layer_for_prompt_id(prompt_id), f"+ Added in prompt {label}")
        )
    return entries


def tokenize_segments(segments: list[Segment]) -> list[Segment]:
    tokens: list[Segment] = []
    for segment in segments:
        text = segment.text
        idx = 0
        while idx < len(text):
            char = text[idx]
            if char == "\n":
                tokens.append(Segment("\n", segment.layer))
                idx += 1
                continue
            if char == " ":
                tokens.append(Segment(" ", segment.layer))
                idx += 1
                continue
            start = idx
            while idx < len(text) and text[idx] not in " \n":
                idx += 1
            tokens.append(Segment(text[start:idx], segment.layer))
    return tokens


def measure_char_width(ax, fontsize: float) -> float:
    fig = ax.figure
    probe = ax.text(
        0,
        0,
        "0" * 20,
        fontsize=fontsize,
        family=FONT_FAMILY,
        transform=ax.transAxes,
    )
    fig.canvas.draw()
    bbox = probe.get_window_extent(fig.canvas.get_renderer())
    ax_bbox = ax.get_window_extent()
    probe.remove()
    return (bbox.width / ax_bbox.width) / 20


def wrap_tokens(tokens: list[Segment], max_width: float, char_width: float) -> list[list[Segment]]:
    max_chars = max(20, int(max_width / char_width))
    lines: list[list[Segment]] = [[]]
    line_len = 0

    def append_token(token: Segment) -> None:
        nonlocal line_len
        if token.text == "\n":
            lines.append([])
            line_len = 0
            return
        if token.text == " " and line_len == 0:
            return
        token_len = len(token.text)
        if line_len + token_len > max_chars and line_len > 0:
            lines.append([])
            line_len = 0
            if token.text == " ":
                return
        if lines[-1] and lines[-1][-1].layer == token.layer:
            lines[-1][-1] = Segment(lines[-1][-1].text + token.text, token.layer)
        else:
            lines[-1].append(token)
        line_len += token_len

    for token in tokens:
        append_token(token)

    return [line for line in lines if line]


def draw_text_block(
    ax,
    y_top: float,
    label: str,
    segments: list[Segment],
    char_width: float,
) -> float:
    inner_w = BOX_W - 2 * PAD_X
    wrapped = wrap_tokens(tokenize_segments(segments), inner_w, char_width)
    label_lines = 1
    content_lines = len(wrapped)
    box_h = PAD_Y + label_lines * LINE_HEIGHT + content_lines * LINE_HEIGHT + PAD_Y
    y_bottom = y_top - box_h

    box = FancyBboxPatch(
        (BOX_X, y_bottom),
        BOX_W,
        box_h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0,
        edgecolor="#BBBBBB",
        facecolor="#FAFAFA",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)

    cursor_y = y_top - PAD_Y
    ax.text(
        BOX_X + PAD_X,
        cursor_y,
        label,
        transform=ax.transAxes,
        fontsize=LABEL_SIZE,
        fontweight="bold",
        color="#555555",
        family="sans-serif",
        va="top",
    )
    cursor_y -= LINE_HEIGHT * 1.05

    for parts in wrapped:
        cursor_x = BOX_X + PAD_X
        for part in parts:
            style = layer_style(part.layer)
            token_w = len(part.text) * char_width
            if part.text.strip() and part.layer != "base":
                highlight = FancyBboxPatch(
                    (cursor_x, cursor_y - LINE_HEIGHT * 0.78),
                    token_w,
                    LINE_HEIGHT * 0.88,
                    boxstyle="square,pad=0",
                    linewidth=0.8,
                    edgecolor=style["edge"],
                    facecolor=style["face"],
                    transform=ax.transAxes,
                    clip_on=False,
                    zorder=1,
                )
                ax.add_patch(highlight)
            ax.text(
                cursor_x,
                cursor_y,
                part.text,
                transform=ax.transAxes,
                fontsize=FONT_SIZE,
                color=style["text"],
                family=FONT_FAMILY,
                va="top",
                zorder=2,
            )
            cursor_x += token_w
        cursor_y -= LINE_HEIGHT

    return box_h + 0.012


def legend_layout(entry_count: int) -> tuple[int, float, float]:
    if entry_count <= 3:
        return 1, 0.31, 0.028
    if entry_count <= 4:
        return 1, 0.23, 0.028
    columns = 4
    rows = (entry_count + columns - 1) // columns
    return rows, 0.22, 0.028


def draw_legend(ax, y: float, legend: list[tuple[str, str]]) -> float:
    rows, col_width, row_height = legend_layout(len(legend))
    columns = (len(legend) + rows - 1) // rows
    legend_x = 0.06
    legend_y = y

    for idx, (layer, label) in enumerate(legend):
        row = idx // columns
        col = idx % columns
        lx = legend_x + col * col_width
        ly = legend_y - row * row_height
        style = layer_style(layer)
        patch = FancyBboxPatch(
            (lx, ly - 0.01),
            0.03,
            0.018,
            boxstyle="square,pad=0",
            linewidth=0.8,
            edgecolor=style["edge"] if layer != "base" else "#CCCCCC",
            facecolor=style["face"],
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(patch)
        ax.text(
            lx + 0.038,
            ly,
            label,
            transform=ax.transAxes,
            fontsize=8.5,
            va="center",
            color="#333333",
        )

    return rows * row_height + 0.02


def render_ladder(ladder: PromptLadder, *, use_internal_ids: bool = False) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for prompt_id in ladder.prompt_ids:
        if prompt_id not in PROMPT_VARIANTS:
            raise KeyError(f"Unknown prompt id: {prompt_id}")

    display_labels = resolve_display_labels(
        ladder.prompt_ids,
        ladder.display_labels,
        use_internal_ids=use_internal_ids,
    )
    title = ladder.title or default_title(ladder, display_labels)
    subtitle = ladder.subtitle or default_subtitle(display_labels)

    system_segments = build_layered_segments(
        ladder.prompt_ids, "system", full_ladder_colors=ladder.full_ladder_colors
    )
    user_segments = build_layered_segments(
        ladder.prompt_ids, "user", full_ladder_colors=ladder.full_ladder_colors
    )
    legend = build_legend(display_labels, ladder.prompt_ids)

    fig_probe, ax_probe = plt.subplots(figsize=(11, 4))
    ax_probe.set_xlim(0, 1)
    ax_probe.set_ylim(0, 1)
    char_width = measure_char_width(ax_probe, FONT_SIZE)
    inner_w = BOX_W - 2 * PAD_X
    system_lines = len(
        wrap_tokens(tokenize_segments(system_segments), inner_w, char_width)
    )
    user_lines = len(wrap_tokens(tokenize_segments(user_segments), inner_w, char_width))
    legend_rows, _, legend_row_h = legend_layout(len(legend))
    plt.close(fig_probe)

    header_h = 0.09
    legend_h = legend_rows * legend_row_h + 0.03
    system_block_h = PAD_Y + LINE_HEIGHT + system_lines * LINE_HEIGHT + PAD_Y + 0.012
    user_block_h = PAD_Y + LINE_HEIGHT + user_lines * LINE_HEIGHT + PAD_Y + 0.012
    fig_h = max(12.0, (header_h + system_block_h + user_block_h + legend_h) * 16)

    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    char_width = measure_char_width(ax, FONT_SIZE)

    ax.text(
        0.5,
        0.99,
        title,
        ha="center",
        va="top",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        0.5,
        0.967,
        subtitle,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#555555",
    )

    y = 0.935
    y -= draw_text_block(ax, y, "System", system_segments, char_width)
    y -= draw_text_block(ax, y, "User", user_segments, char_width)
    draw_legend(ax, y - 0.014, legend)

    png_path = OUT_DIR / ladder.output_filename
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def ladder_from_prompt_ids(
    prompt_ids: tuple[str, ...],
    display_labels: tuple[str, ...] | None = None,
) -> PromptLadder:
    labels = resolve_display_labels(prompt_ids, display_labels)
    return PromptLadder(
        name="custom",
        prompt_ids=prompt_ids,
        output_filename=f"prompt_ladder_{'_'.join(labels)}.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ladder",
        choices=[*LADDERS.keys(), "all"],
        default="all",
        help="Which predefined ladder to render (default: all).",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        metavar="ID",
        help="Custom prompt ID sequence (overrides --ladder).",
    )
    parser.add_argument(
        "--internal-labels",
        action="store_true",
        help="Use internal prompt IDs in titles/legends instead of paper labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_internal_ids = args.internal_labels

    if args.prompts:
        ladder = ladder_from_prompt_ids(tuple(args.prompts))
        paths = [render_ladder(ladder, use_internal_ids=use_internal_ids)]
    elif args.ladder == "all":
        paths = [
            render_ladder(LADDERS[name], use_internal_ids=use_internal_ids)
            for name in ("paper", "gemma_deep_dive")
        ]
    else:
        paths = [render_ladder(LADDERS[args.ladder], use_internal_ids=use_internal_ids)]

    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
