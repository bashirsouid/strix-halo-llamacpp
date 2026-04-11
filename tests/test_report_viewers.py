from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_viewer  # type: ignore
import parallel_viewer  # type: ignore
from report_helpers import stable_color  # type: ignore

AIDER_RESULTS = REPO_ROOT / "results" / "aider" / "aider_results.jsonl"
PARALLEL_RESULTS = REPO_ROOT / "results" / "benchmark" / "bench_parallel_results.jsonl"


def _extract_json_const(html: str, name: str):
    match = re.search(rf"const {name} = (.*?);\n", html, flags=re.DOTALL)
    assert match, f"missing JS const: {name}"
    return json.loads(match.group(1))



def test_stable_color_is_deterministic() -> None:
    key = "Qwen3 Coder Next (Q6_K) (rocm7, python-quick, max=24576, tries=2)"
    assert stable_color(key) == stable_color(key)
    assert stable_color(key, alpha=0.2) == stable_color(key, alpha=0.2)
    assert stable_color(key).startswith("hsl(")
    assert stable_color(key, alpha=0.2).startswith("hsla(")



def test_eval_chart_labels_wrap_long_series_names() -> None:
    records = eval_viewer.load_records(AIDER_RESULTS)
    latest = eval_viewer._latest_per_series(records)
    longest = max(latest, key=lambda record: len(eval_viewer._series_key(record)))

    wrapped = eval_viewer._chart_label(longest, width=34)

    assert len(wrapped) >= 2
    assert all(len(line) <= 34 for line in wrapped)



def test_eval_series_colors_stay_consistent_across_charts() -> None:
    html = eval_viewer.generate_html(eval_viewer.load_records(AIDER_RESULTS))

    pass_labels = _extract_json_const(html, "passFullLabels")
    pass_colors = _extract_json_const(html, "passColors")
    speed_labels = _extract_json_const(html, "speedFullLabels")
    speed_colors = _extract_json_const(html, "speedColors")
    time_labels = _extract_json_const(html, "timeFullLabels")
    time_colors = _extract_json_const(html, "timeColors")

    pass_map = dict(zip(pass_labels, pass_colors, strict=True))
    speed_map = dict(zip(speed_labels, speed_colors, strict=True))
    time_map = dict(zip(time_labels, time_colors, strict=True))

    assert pass_map == speed_map == time_map
    for key, color in pass_map.items():
        assert color == stable_color(key)



def test_parallel_report_uses_stable_model_colors() -> None:
    html = parallel_viewer.generate_html(parallel_viewer.load_records(PARALLEL_RESULTS))
    model_key = "glm-4.7-flash-q8 (radv)"
    color = stable_color(model_key)
    translucent = stable_color(model_key, alpha=0.20)

    assert color in html
    assert translucent in html
    assert "const npLabels" in html
