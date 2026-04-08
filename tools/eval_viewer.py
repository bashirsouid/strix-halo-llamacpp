#!/usr/bin/env python3
"""
Aider evaluation viewer.

Reads the Aider benchmark JSONL at results/aider/aider_results.jsonl and opens an
interactive report in your browser. No dependencies beyond the Python stdlib.

Usage:
    python tools/eval_viewer.py
    python tools/eval_viewer.py path/to/aider_results.jsonl
"""

from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FILE = PROJECT_DIR / "results" / "aider" / "aider_results.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "eval_report.html"


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def load_records(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _display_model(record: dict) -> str:
    name = str(record.get("model_display_name") or record.get("model") or "unknown").strip()
    quant = str(record.get("quant") or "").strip()
    if quant and quant not in name:
        return f"{name} [{quant}]"
    return name


def _variant_parts(record: dict) -> list[str]:
    parts: list[str] = []
    backend = str(record.get("backend") or "").strip()
    profile = str(record.get("profile") or record.get("eval_profile") or "").strip()
    label = str(record.get("run_label") or "").strip()
    edit_format = str(record.get("edit_format") or "").strip()
    max_tokens = _safe_int(record.get("max_tokens"))
    tries = _safe_int(record.get("tries"))

    if backend:
        parts.append(backend)
    if profile:
        parts.append(profile)
    if max_tokens:
        parts.append(f"max={max_tokens}")
    if tries:
        parts.append(f"tries={tries}")
    if edit_format and edit_format != "whole":
        parts.append(edit_format)
    if label:
        parts.append(f"label={label}")
    return parts


def _series_key(record: dict) -> str:
    base = _display_model(record)
    parts = _variant_parts(record)
    return f"{base} ({', '.join(parts)})" if parts else base


def _latest_per_series(records: list[dict]) -> list[dict]:
    latest: dict[str, tuple[str, int, dict]] = {}
    for index, record in enumerate(records):
        key = _series_key(record)
        stamp = str(record.get("timestamp") or "")
        current = latest.get(key)
        if current is None or (stamp, index) >= (current[0], current[1]):
            latest[key] = (stamp, index, record)
    return [value[2] for value in latest.values()]


def _metric(record: dict, name: str, default: float = 0.0) -> float:
    value = _safe_float(record.get(name))
    return value if value is not None else default


def _int_metric(record: dict, name: str, default: int = 0) -> int:
    value = _safe_int(record.get(name))
    return value if value is not None else default


def generate_html(records: list[dict]) -> str:
    palette = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
        "#264653", "#a8dadc", "#6d6875", "#b5838d", "#ffb703",
        "#023047", "#8ecae6",
    ]

    latest = _latest_per_series(records)
    latest = sorted(
        latest,
        key=lambda record: (
            _metric(record, "pass_rate_2", -1.0),
            _metric(record, "pass_rate_1", -1.0),
            str(record.get("timestamp") or ""),
        ),
        reverse=True,
    )

    if not latest:
        return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Strix Halo Aider Results</title>
  <style>
    body { font-family: Inter, system-ui, sans-serif; margin: 40px; background: #0d1117; color: #e6edf3; }
    .box { max-width: 760px; margin: 80px auto; padding: 24px; border: 1px solid #30363d; border-radius: 16px; background: #161b22; }
    code { background: #0d1117; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class=\"box\">
    <h1>No Aider benchmark results found</h1>
    <p>Run <code>python server.py aider-bench MODEL --profile python-quick</code> first.</p>
  </div>
</body>
</html>"""

    pass_labels = [_series_key(record) for record in latest]
    pass_try1 = [round(_metric(record, "pass_rate_1"), 1) for record in latest]
    pass_try2 = [round(_metric(record, "pass_rate_2"), 1) for record in latest]
    pass_colors = [palette[index % len(palette)] for index in range(len(latest))]

    speed_sorted = sorted(latest, key=lambda record: _metric(record, "completion_tok_s_wall"), reverse=True)
    speed_labels = [_series_key(record) for record in speed_sorted]
    speed_values = [round(_metric(record, "completion_tok_s_wall"), 2) for record in speed_sorted]
    speed_colors = [palette[index % len(palette)] for index in range(len(speed_sorted))]

    time_sorted = sorted(
        latest,
        key=lambda record: _metric(record, "seconds_per_case_wall", 10**9),
    )
    time_labels = [_series_key(record) for record in time_sorted]
    time_values = [round(_metric(record, "seconds_per_case_wall"), 2) for record in time_sorted]
    time_colors = [palette[index % len(palette)] for index in range(len(time_sorted))]

    warning_labels = [_series_key(record) for record in latest]
    exhausted_values = [_int_metric(record, "exhausted_context_windows") for record in latest]
    malformed_values = [_int_metric(record, "num_malformed_responses") for record in latest]
    timeout_values = [_int_metric(record, "test_timeouts") for record in latest]

    scatter_points = []
    for index, record in enumerate(latest):
        seconds = _safe_float(record.get("seconds_per_case_wall"))
        score = _safe_float(record.get("pass_rate_2"))
        if seconds is None or score is None:
            continue
        scatter_points.append(
            {
                "x": round(seconds, 2),
                "y": round(score, 1),
                "label": _series_key(record),
                "color": palette[index % len(palette)],
                "wellFormed": round(_metric(record, "percent_cases_well_formed"), 1),
            }
        )

    timestamps = sorted({str(record.get("timestamp") or "") for record in records if record.get("timestamp")})
    history_map: dict[str, list[dict]] = {}
    for record in records:
        if record.get("pass_rate_2") is None:
            continue
        history_map.setdefault(_series_key(record), []).append(record)

    history_datasets = []
    for index, (key, series_records) in enumerate(sorted(history_map.items())):
        if len(series_records) < 2:
            continue
        color = palette[index % len(palette)]
        lookup = {
            str(record.get("timestamp") or ""): round(_metric(record, "pass_rate_2"), 1)
            for record in series_records
        }
        values = [lookup.get(timestamp, None) for timestamp in timestamps]
        history_datasets.append(
            {
                "label": key,
                "data": values,
                "borderColor": color,
                "backgroundColor": f"{color}33",
            }
        )

    best_try2 = max(latest, key=lambda record: _metric(record, "pass_rate_2", -1.0), default=None)
    fastest = min(
        [record for record in latest if record.get("seconds_per_case_wall") is not None],
        key=lambda record: _metric(record, "seconds_per_case_wall", 10**9),
        default=None,
    )
    fastest_tps = max(
        [record for record in latest if record.get("completion_tok_s_wall") is not None],
        key=lambda record: _metric(record, "completion_tok_s_wall", -1.0),
        default=None,
    )
    cleanest = max(
        latest,
        key=lambda record: _metric(record, "percent_cases_well_formed", -1.0),
        default=None,
    )

    def card(metric: str, record: dict | None, value: str) -> str:
        if not record:
            return ""
        return f"""
      <div class=\"stat-card\">
        <div class=\"stat-value\">{value}</div>
        <div class=\"stat-label\">{metric}</div>
        <div class=\"stat-model\">{_series_key(record)}</div>
      </div>
"""

    cards_html = "".join(
        [
            card(
                "Best try 2",
                best_try2,
                f"{_metric(best_try2, 'pass_rate_2'):.1f}%" if best_try2 else "—",
            ),
            card(
                "Fastest case",
                fastest,
                f"{_metric(fastest, 'seconds_per_case_wall'):.2f}s" if fastest else "—",
            ),
            card(
                "Best tok/s",
                fastest_tps,
                f"{_metric(fastest_tps, 'completion_tok_s_wall'):.2f}" if fastest_tps else "—",
            ),
            card(
                "Cleanest output",
                cleanest,
                f"{_metric(cleanest, 'percent_cases_well_formed'):.1f}%" if cleanest else "—",
            ),
        ]
    )

    table_rows = []
    for record in latest:
        cases = f"{_int_metric(record, 'completed_tests')}/{_int_metric(record, 'total_tests')}"
        ok_flag = "✓" if record.get("ok") else "✗"
        table_rows.append(
            f"""
        <tr>
          <td>{_display_model(record)}</td>
          <td class=\"mono\">{record.get('backend') or '—'}</td>
          <td class=\"mono\">{record.get('profile') or '—'}</td>
          <td class=\"mono\">{record.get('run_label') or '—'}</td>
          <td class=\"mono\">{_int_metric(record, 'max_tokens') or '—'}</td>
          <td class=\"score\">{_metric(record, 'pass_rate_1'):.1f}%</td>
          <td class=\"score\">{_metric(record, 'pass_rate_2'):.1f}%</td>
          <td class=\"mono\">{cases}</td>
          <td class=\"mono\">{_metric(record, 'seconds_per_case_wall'):.2f}</td>
          <td class=\"mono\">{_metric(record, 'completion_tok_s_wall'):.2f}</td>
          <td class=\"mono\">{_metric(record, 'percent_cases_well_formed'):.1f}%</td>
          <td class=\"mono\">{_int_metric(record, 'exhausted_context_windows')}</td>
          <td class=\"mono\">{_int_metric(record, 'num_malformed_responses')}</td>
          <td class=\"mono\">{_int_metric(record, 'syntax_errors')}</td>
          <td class=\"mono\">{_int_metric(record, 'test_timeouts')}</td>
          <td class=\"status {'ok' if record.get('ok') else 'fail'}\">{ok_flag}</td>
          <td class=\"mono dim\">{record.get('timestamp') or '—'}</td>
        </tr>
"""
        )
    table_rows_html = "".join(table_rows)

    timestamps_js = json.dumps(timestamps)
    pass_labels_js = json.dumps(pass_labels)
    pass_try1_js = json.dumps(pass_try1)
    pass_try2_js = json.dumps(pass_try2)
    pass_colors_js = json.dumps(pass_colors)
    speed_labels_js = json.dumps(speed_labels)
    speed_values_js = json.dumps(speed_values)
    speed_colors_js = json.dumps(speed_colors)
    time_labels_js = json.dumps(time_labels)
    time_values_js = json.dumps(time_values)
    time_colors_js = json.dumps(time_colors)
    warning_labels_js = json.dumps(warning_labels)
    exhausted_js = json.dumps(exhausted_values)
    malformed_js = json.dumps(malformed_values)
    timeout_js = json.dumps(timeout_values)
    scatter_js = json.dumps(scatter_points)
    history_datasets_js = json.dumps(history_datasets)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Strix Halo Aider Results</title>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0d1117;
      --panel: #161b22;
      --panel-2: #11161d;
      --text: #e6edf3;
      --muted: #8b949e;
      --border: #30363d;
      --ok: #3fb950;
      --fail: #f85149;
      --accent: #58a6ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, system-ui, sans-serif; }}
    main {{ max-width: 1600px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    p {{ margin: 0; }}
    .lead {{ color: var(--muted); margin-bottom: 22px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 18px 0 24px; }}
    .stat-card, .chart-box, .table-box {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; box-shadow: 0 12px 32px rgba(0,0,0,0.20); }}
    .stat-card {{ padding: 18px 18px 16px; }}
    .stat-value {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
    .stat-label {{ font-size: 13px; color: var(--muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .stat-model {{ font-size: 13px; line-height: 1.45; color: var(--text); }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; }}
    .chart-box {{ padding: 18px; }}
    .chart-sub {{ color: var(--muted); font-size: 13px; margin-bottom: 12px; }}
    .chart-box canvas {{ width: 100%; max-width: 100%; height: 360px; }}
    .table-box {{ margin-top: 22px; overflow: hidden; }}
    .table-scroll {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }}
    th {{ background: var(--panel-2); position: sticky; top: 0; z-index: 1; }}
    tr:hover td {{ background: #1d2530; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .dim {{ color: var(--muted); }}
    .score {{ font-weight: 600; }}
    .status.ok {{ color: var(--ok); font-weight: 700; }}
    .status.fail {{ color: var(--fail); font-weight: 700; }}
    @media (max-width: 900px) {{
      main {{ padding: 18px; }}
      .chart-grid {{ grid-template-columns: 1fr; }}
      .chart-box canvas {{ height: 320px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Strix Halo — Aider benchmark results</h1>
    <p class=\"lead\">{len(records)} total records · {len(latest)} latest comparison rows · input file: {DEFAULT_FILE if DEFAULT_FILE.exists() else 'custom path'}</p>

    <section class=\"stats-grid\">{cards_html}</section>

    <section class=\"chart-grid\">
      <div class=\"chart-box\">
        <h2>Pass rate by model</h2>
        <p class=\"chart-sub\">Try 2 is the most important comparison; try 1 shows first-shot quality.</p>
        <canvas id=\"passChart\"></canvas>
      </div>
      <div class=\"chart-box\">
        <h2>Completion tok/s</h2>
        <p class=\"chart-sub\">Higher is better. This is wall-clock throughput across the full run.</p>
        <canvas id=\"speedChart\"></canvas>
      </div>
      <div class=\"chart-box\">
        <h2>Seconds per case</h2>
        <p class=\"chart-sub\">Lower is better. Useful for judging whether a profile still fits your 30–60 minute budget.</p>
        <canvas id=\"timeChart\"></canvas>
      </div>
      <div class=\"chart-box\">
        <h2>Quality vs speed</h2>
        <p class=\"chart-sub\">Up and left is better: higher try 2 pass rate, lower seconds per case.</p>
        <canvas id=\"scatterChart\"></canvas>
      </div>
      <div class=\"chart-box\">
        <h2>Warning counters</h2>
        <p class=\"chart-sub\">Shows context exhaustion, malformed responses, and timed-out test runs.</p>
        <canvas id=\"warningChart\"></canvas>
      </div>
      <div class=\"chart-box\">
        <h2>Try 2 history</h2>
        <p class=\"chart-sub\">Use labels or max_tokens changes to compare repeated runs of the same model.</p>
        <canvas id=\"historyChart\"></canvas>
      </div>
    </section>

    <section class=\"table-box\">
      <div class=\"table-scroll\">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Backend</th>
              <th>Profile</th>
              <th>Label</th>
              <th>Max tokens</th>
              <th>Try 1</th>
              <th>Try 2</th>
              <th>Cases</th>
              <th>s/case</th>
              <th>tok/s</th>
              <th>Well formed</th>
              <th>Ctx exhaust</th>
              <th>Malformed</th>
              <th>Syntax</th>
              <th>Timeouts</th>
              <th>Status</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>{table_rows_html}</tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const passLabels = {pass_labels_js};
    const passTry1 = {pass_try1_js};
    const passTry2 = {pass_try2_js};
    const passColors = {pass_colors_js};
    const speedLabels = {speed_labels_js};
    const speedValues = {speed_values_js};
    const speedColors = {speed_colors_js};
    const timeLabels = {time_labels_js};
    const timeValues = {time_values_js};
    const timeColors = {time_colors_js};
    const warningLabels = {warning_labels_js};
    const exhaustedValues = {exhausted_js};
    const malformedValues = {malformed_js};
    const timeoutValues = {timeout_js};
    const scatterPoints = {scatter_js};
    const historyTimestamps = {timestamps_js};
    const historyDatasets = {history_datasets_js};

    const axisCommon = {{
      ticks: {{ color: '#c9d1d9' }},
      grid: {{ color: '#30363d' }},
      border: {{ color: '#30363d' }},
    }};
    const pluginCommon = {{
      legend: {{ labels: {{ color: '#c9d1d9', usePointStyle: true }} }},
      tooltip: {{
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        borderWidth: 1,
        titleColor: '#e6edf3',
        bodyColor: '#c9d1d9',
      }},
    }};

    new Chart(document.getElementById('passChart'), {{
      type: 'bar',
      data: {{
        labels: passLabels,
        datasets: [
          {{
            label: 'Try 1 (%)',
            data: passTry1,
            backgroundColor: passColors.map(color => color + '99'),
            borderColor: passColors,
            borderWidth: 1,
            borderRadius: 5,
          }},
          {{
            label: 'Try 2 (%)',
            data: passTry2,
            backgroundColor: passColors,
            borderColor: passColors,
            borderWidth: 1,
            borderRadius: 5,
          }},
        ],
      }},
      options: {{
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: pluginCommon,
        scales: {{
          x: {{ ...axisCommon, beginAtZero: true, max: 100, title: {{ display: true, text: 'Pass rate %', color: '#c9d1d9' }} }},
          y: axisCommon,
        }},
      }},
    }});

    new Chart(document.getElementById('speedChart'), {{
      type: 'bar',
      data: {{
        labels: speedLabels,
        datasets: [{{
          label: 'Completion tok/s',
          data: speedValues,
          backgroundColor: speedColors,
          borderColor: speedColors,
          borderWidth: 1,
          borderRadius: 5,
        }}],
      }},
      options: {{
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: pluginCommon,
        scales: {{
          x: {{ ...axisCommon, beginAtZero: true, title: {{ display: true, text: 'tok/s', color: '#c9d1d9' }} }},
          y: axisCommon,
        }},
      }},
    }});

    new Chart(document.getElementById('timeChart'), {{
      type: 'bar',
      data: {{
        labels: timeLabels,
        datasets: [{{
          label: 'Seconds per case',
          data: timeValues,
          backgroundColor: timeColors,
          borderColor: timeColors,
          borderWidth: 1,
          borderRadius: 5,
        }}],
      }},
      options: {{
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: pluginCommon,
        scales: {{
          x: {{ ...axisCommon, beginAtZero: true, title: {{ display: true, text: 'seconds', color: '#c9d1d9' }} }},
          y: axisCommon,
        }},
      }},
    }});

    new Chart(document.getElementById('scatterChart'), {{
      type: 'scatter',
      data: {{
        datasets: scatterPoints.map(point => ({{
          label: point.label,
          data: [{{ x: point.x, y: point.y }}],
          pointRadius: 6,
          pointHoverRadius: 8,
          pointBackgroundColor: point.color,
          pointBorderColor: point.color,
        }})),
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          ...pluginCommon,
          tooltip: {{
            ...pluginCommon.tooltip,
            callbacks: {{
              label(context) {{
                const dataset = scatterPoints[context.datasetIndex];
                return `${{dataset.label}} | ${{dataset.x}}s/case | ${{dataset.y}}% try2 | ${{dataset.wellFormed}}% well formed`;
              }},
            }},
          }},
          legend: {{ display: false }},
        }},
        scales: {{
          x: {{ ...axisCommon, beginAtZero: true, title: {{ display: true, text: 'Seconds per case', color: '#c9d1d9' }} }},
          y: {{ ...axisCommon, beginAtZero: true, max: 100, title: {{ display: true, text: 'Try 2 pass rate %', color: '#c9d1d9' }} }},
        }},
      }},
    }});

    new Chart(document.getElementById('warningChart'), {{
      type: 'bar',
      data: {{
        labels: warningLabels,
        datasets: [
          {{ label: 'Context exhaustion', data: exhaustedValues, backgroundColor: '#e63946', borderRadius: 5 }},
          {{ label: 'Malformed responses', data: malformedValues, backgroundColor: '#f4a261', borderRadius: 5 }},
          {{ label: 'Timed-out tests', data: timeoutValues, backgroundColor: '#457b9d', borderRadius: 5 }},
        ],
      }},
      options: {{
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: pluginCommon,
        scales: {{
          x: {{ ...axisCommon, beginAtZero: true, stacked: true, title: {{ display: true, text: 'count', color: '#c9d1d9' }} }},
          y: {{ ...axisCommon, stacked: true }},
        }},
      }},
    }});

    new Chart(document.getElementById('historyChart'), {{
      type: 'line',
      data: {{
        labels: historyTimestamps,
        datasets: historyDatasets.map(dataset => ({{
          ...dataset,
          pointRadius: 4,
          pointHoverRadius: 6,
          borderWidth: 2,
          tension: 0.2,
          spanGaps: true,
        }})),
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: pluginCommon,
        scales: {{
          x: axisCommon,
          y: {{ ...axisCommon, beginAtZero: true, max: 100, title: {{ display: true, text: 'Try 2 pass rate %', color: '#c9d1d9' }} }},
        }},
      }},
    }});
  </script>
</body>
</html>
"""


def output_path_for(input_path: Path) -> Path:
    try:
        if input_path.resolve() == DEFAULT_FILE.resolve():
            return DEFAULT_OUTPUT
    except FileNotFoundError:
        pass
    return input_path.with_suffix(".html")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    input_path = Path(args[0]).expanduser().resolve() if args else DEFAULT_FILE
    output_path = output_path_for(input_path)

    records = load_records(input_path)
    html = generate_html(records)
    output_path.write_text(html, encoding="utf-8")
    print(output_path)
    webbrowser.open(output_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
