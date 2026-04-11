from __future__ import annotations

import hashlib
import textwrap
from typing import Sequence


def _format_alpha(alpha: float) -> str:
    return f"{alpha:.3f}".rstrip("0").rstrip(".")


def stable_color(key: str, alpha: float | None = None) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") % 360
    saturation = 62 + digest[2] % 15
    lightness = 48 + digest[3] % 14
    if alpha is None:
        return f"hsl({hue}, {saturation}%, {lightness}%)"
    return f"hsla({hue}, {saturation}%, {lightness}%, {_format_alpha(alpha)})"


def compact_k(value: int | None) -> str | None:
    if value is None:
        return None
    value = int(value)
    if value < 1000:
        return str(value)
    compact = f"{value / 1000:.1f}k"
    return compact.replace(".0k", "k")


def wrap_text_label(text: str, width: int = 34) -> list[str]:
    cleaned = " ".join(str(text).split())
    lines = textwrap.wrap(
        cleaned,
        width=width,
        break_long_words=False,
        break_on_hyphens=True,
    )
    return lines or [cleaned]


def wrap_label_parts(parts: Sequence[str], width: int = 34) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw_part in parts:
        part = str(raw_part).strip()
        if not part:
            continue
        for subpart in wrap_text_label(part, width=width):
            if not current:
                current = subpart
            elif len(current) + 3 + len(subpart) <= width:
                current += f" · {subpart}"
            else:
                lines.append(current)
                current = subpart
    if current:
        lines.append(current)
    return lines or [""]


def bar_chart_height(
    wrapped_labels: Sequence[Sequence[str] | str],
    *,
    min_height: int = 360,
    base_padding: int = 84,
    base_per_label: int = 28,
    extra_per_line: int = 14,
) -> int:
    height = base_padding
    for label in wrapped_labels:
        line_count = len(label) if not isinstance(label, str) else 1
        height += base_per_label + extra_per_line * max(0, line_count - 1)
    return max(min_height, height)
