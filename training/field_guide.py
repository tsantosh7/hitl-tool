# training/field_guide.py
from __future__ import annotations

from typing import Dict, Any, List

from fields import FIELDS, SENTINEL


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def build_field_guide_text(
    *,
    max_fields: int | None = None,
    examples_per_field: int = 3,
    max_example_chars: int = 140,
) -> str:
    """
    Builds a compact "field guide" text for the prompt.
    Uses the definitions/examples in fields.py.
    """
    lines: List[str] = []
    lines.append("FIELD GUIDE (for output_1):")
    lines.append(f"- If no explicit evidence, return ['{SENTINEL}'].")
    lines.append("- Otherwise, EACH string must be copied verbatim from DOCUMENT.")
    lines.append("")

    items = list(FIELDS.items())
    if max_fields is not None:
        items = items[:max_fields]

    for name, spec in items:
        desc = spec.get("description", "")
        ex = spec.get("example", [])
        if not isinstance(ex, list):
            ex = []

        ex = ex[: max(0, int(examples_per_field))]
        ex_clean = [_clip(str(x), max_example_chars) for x in ex]

        lines.append(f"{name}: {_clip(desc, 220)}")
        if ex_clean:
            lines.append(f"Examples: {ex_clean}")
        lines.append("")

    return "\n".join(lines).strip()

