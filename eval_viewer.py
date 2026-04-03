#!/usr/bin/env python3
"""
Eval viewer — reads eval_results.jsonl and opens an interactive report
in your browser.  No dependencies beyond Python stdlib.

Usage:
    python eval_viewer.py                    # uses ./eval_results.jsonl
    python eval_viewer.py path/to/results.jsonl
"""

import json
import sys
import webbrowser
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent / "eval_results.jsonl"


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _model_key(r: dict) -> str:
    """Short label: model [quant] (backend, suite)"""
    quant = r.get("quant") or ""
    suite = r.get("suite", "")
    model = r["model"]
    backend = r["backend"]
    base = f"{model} ({backend}, {suite})"
    return f"{base} [{quant}]" if quant else base


def _latest_per_model(records: list[dict]) -> list[dict]:
    """For each unique model/backend/suite/quant combo, keep only the most recent run."""
    latest: dict[str, dict] = {}
    for r in records:
        key = _model_key(r)
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r
    return list(latest.values())


def generate_html(records: list[dict]) -> str:
    palette = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
        "#264653", "#a8dadc", "#6d6875", "#b5838d", "#ffb703",
        "#023047", "#8ecae6",
    ]

    latest = _latest_per_model(records)
    timestamps = sorted(set(r["timestamp"] for r in records))

    # ── Pass@1 bar data (ranked by plus, or base if no plus) ─────────────
    scored = [r for r in latest if r.get("pass_at_1_base") is not None]
    scored_sorted = sorted(scored, key=lambda r: r.get("pass_at_1_plus") or r.get("pass_at_1_base") or 0, reverse=True)
    bar_labels    = [_model_key(r) for r in scored_sorted]
    bar_base      = [round((r.get("pass_at_1_base") or 0) * 100, 1) for r in scored_sorted]
    bar_plus      = [round((r.get("pass_at_1_plus") or 0) * 100, 1) for r in scored_sorted]
    bar_colors    = [palette[i % len(palette)] for i in range(len(scored_sorted))]
    has_scores    = len(scored_sorted) > 0

    # ── Wall time bar data (ranked fastest first) ─────────────────────────
    timed_sorted = sorted(latest, key=lambda r: r.get("wall_time_sec") or 0)
    time_labels  = [_model_key(r) for r in timed_sorted]
    time_values  = [r.get("wall_time_sec") or 0 for r in timed_sorted]
    time_colors  = [palette[i % len(palette)] for i in range(len(timed_sorted))]

    # ── Scatter: pass@1 plus vs wall time ─────────────────────────────────
    scatter_points = []
    for i, r in enumerate(scored):
        if r.get("wall_time_sec"):
            scatter_points.append({
                "x": r["wall_time_sec"],
                "y": round((r.get("pass_at_1_plus") or 0) * 100, 1),
                "label": _model_key(r),
                "color": palette[i % len(palette)],
            })
    has_scatter = len(scatter_points) >= 2

    # ── History lines: pass@1 plus over time per model ────────────────────
    history_series: dict[str, list[dict]] = {}
    for r in records:
        if r.get("pass_at_1_plus") is not None:
            key = _model_key(r)
            history_series.setdefault(key, []).append(r)
    has_history = any(len(v) > 1 for v in history_series.values())

    history_datasets_js = []
    for i, (key, recs) in enumerate(sorted(history_series.items())):
        color = palette[i % len(palette)]
        lookup = {r["timestamp"]: round((r["pass_at_1_plus"] or 0) * 100, 1) for r in recs}
        pts = [lookup.get(ts, "null") for ts in timestamps]
        history_datasets_js.append(f"""{{
            label: {json.dumps(key)},
            data: [{', '.join(str(p) for p in pts)}],
            borderColor: '{color}',
            backgroundColor: '{color}33',
            borderWidth: 2, pointRadius: 4, pointHoverRadius: 6,
            tension: 0.2, spanGaps: true,
        }}""")

    # ── Wall time history lines ───────────────────────────────────────────
    wall_series: dict[str, list[dict]] = {}
    for r in records:
        if r.get("wall_time_sec") is not None:
            key = _model_key(r)
            wall_series.setdefault(key, []).append(r)

    wall_datasets_js = []
    for i, (key, recs) in enumerate(sorted(wall_series.items())):
        color = palette[i % len(palette)]
        lookup = {r["timestamp"]: r["wall_time_sec"] for r in recs}
        pts = [lookup.get(ts, "null") for ts in timestamps]
        wall_datasets_js.append(f"""{{
            label: {json.dumps(key)},
            data: [{', '.join(str(p) for p in pts)}],
            borderColor: '{color}',
            backgroundColor: '{color}33',
            borderWidth: 2, pointRadius: 4, pointHoverRadius: 6,
            tension: 0.2, spanGaps: true,
        }}""")

    # ── Summary table rows ────────────────────────────────────────────────
    table_rows_sorted = sorted(latest,
        key=lambda r: r.get("pass_at_1_plus") or r.get("pass_at_1_base") or 0,
        reverse=True)
    table_rows_html = ""
    for r in table_rows_sorted:
        base = r.get("pass_at_1_base")
        plus = r.get("pass_at_1_plus")
        wall = r.get("wall_time_sec")
        ok_flag = "✓" if r.get("ok") else "✗"
        base_str = f"{base*100:.1f}%" if base is not None else "—"
        plus_str = f"{plus*100:.1f}%" if plus is not None else "—"
        wall_str = f"{wall:.0f}s ({wall/60:.1f}m)" if wall else "—"
        suite = r.get("suite", "—")
        quant = r.get("quant") or "—"
        backend = r.get("backend", "—")
        model = r["model"]
        ts = r["timestamp"]
        table_rows_html += f"""
        <tr>
          <td>{model}</td>
          <td class="mono">{quant}</td>
          <td class="mono">{backend}</td>
          <td class="mono">{suite}</td>
          <td class="score {'good' if base and base >= 0.7 else 'mid' if base and base >= 0.5 else 'low' if base else ''}">{base_str}</td>
          <td class="score {'good' if plus and plus >= 0.7 else 'mid' if plus and plus >= 0.5 else 'low' if plus else ''}">{plus_str}</td>
          <td class="mono">{wall_str}</td>
          <td class="status {'ok' if r.get('ok') else 'fail'}">{ok_flag}</td>
          <td class="mono dim">{ts}</td>
        </tr>"""

    # ── JSON blobs for JS ─────────────────────────────────────────────────
    bar_labels_js    = json.dumps(bar_labels)
    bar_base_js      = json.dumps(bar_base)
    bar_plus_js      = json.dumps(bar_plus)
    bar_colors_js    = json.dumps(bar_colors)
    time_labels_js   = json.dumps(time_labels)
    time_values_js   = json.dumps(time_values)
    time_colors_js   = json.dumps(time_colors)
    scatter_js       = json.dumps(scatter_points)
    timestamps_js    = json.dumps(timestamps)

    score_section = f"""
  <div class="chart-box">
    <h2>pass@1 — base vs plus (latest run per model)</h2>
    <p class="chart-sub">Base = original HumanEval/MBPP tests &nbsp;|&nbsp; Plus = expanded test suite (harder, less contamination risk)</p>
    <canvas id="passChart"></canvas>
  </div>
""" if has_scores else """
  <div class="chart-box empty-box">
    <h2>pass@1 scores</h2>
    <p class="dim">No pass@1 data yet — run evaluate.sh to generate scores.</p>
  </div>
"""

    scatter_section = f"""
  <div class="chart-box">
    <h2>Quality vs speed tradeoff (plus pass@1 vs wall time)</h2>
    <p class="chart-sub">Up and left is better — higher score, faster eval</p>
    <canvas id="scatterChart"></canvas>
  </div>
""" if has_scatter else ""

    history_section = f"""
  <div class="chart-box">
    <h2>pass@1 plus — history over time</h2>
    <canvas id="historyChart"></canvas>
  </div>
""" if has_history else ""

    score_chart_js = f"""
new Chart(document.getElementById('passChart'), {{
  type: 'bar',
  data: {{
    labels: {bar_labels_js},
    datasets: [
      {{
        label: 'Base pass@1 (%)',
        data: {bar_base_js},
        backgroundColor: {bar_colors_js}.map(c => c + 'aa'),
        borderColor: {bar_colors_js},
        borderWidth: 1,
        borderRadius: 3,
      }},
      {{
        label: 'Plus pass@1 (%)',
        data: {bar_plus_js},
        backgroundColor: {bar_colors_js},
        borderColor: {bar_colors_js},
        borderWidth: 1,
        borderRadius: 3,
      }},
    ]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ font: {{ size: 11 }}, usePointStyle: true }} }},
      tooltip: {{
        backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
        callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.x + '%' }}
      }}
    }},
    scales: {{
      x: {{
        min: 0, max: 100,
        title: {{ display: true, text: 'pass@1 (%)', font: {{ size: 11 }} }},
        ticks: {{ font: {{ size: 11 }}, callback: v => v + '%' }}
      }},
      y: {{ ticks: {{ font: {{ size: 10 }} }} }}
    }}
  }}
}});
""" if has_scores else ""

    scatter_chart_js = f"""
new Chart(document.getElementById('scatterChart'), {{
  type: 'scatter',
  data: {{
    datasets: {scatter_js}.map((pt, i) => ({{
      label: pt.label,
      data: [{{ x: pt.x, y: pt.y }}],
      backgroundColor: pt.color,
      borderColor: pt.color,
      pointRadius: 8,
      pointHoverRadius: 10,
    }}))
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ font: {{ size: 10 }} }} }},
      tooltip: {{
        backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
        callbacks: {{
          label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y}}% pass@1, ${{ctx.parsed.x}}s wall`
        }}
      }}
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Wall time (s)', font: {{ size: 11 }} }},
        ticks: {{ font: {{ size: 11 }} }}
      }},
      y: {{
        min: 0, max: 100,
        title: {{ display: true, text: 'Plus pass@1 (%)', font: {{ size: 11 }} }},
        ticks: {{ font: {{ size: 11 }}, callback: v => v + '%' }}
      }}
    }}
  }}
}});
""" if has_scatter else ""

    history_chart_js = f"""
new Chart(document.getElementById('historyChart'), {{
  type: 'line',
  data: {{
    labels: {timestamps_js},
    datasets: [{', '.join(history_datasets_js)}]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ font: {{ size: 11 }}, usePointStyle: true, pointStyle: 'circle' }} }},
      tooltip: {{
        backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
        callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y + '%' }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ font: {{ size: 10 }}, maxRotation: 45 }} }},
      y: {{
        min: 0, max: 100,
        title: {{ display: true, text: 'Plus pass@1 (%)', font: {{ size: 11 }} }},
        ticks: {{ font: {{ size: 11 }}, callback: v => v + '%' }}
      }}
    }}
  }}
}});
""" if has_history else ""

    n_models = len(set(r["model"] for r in records))
    n_backends = len(set(r["backend"] for r in records))
    best = max(scored_sorted, key=lambda r: r.get("pass_at_1_plus") or 0) if scored_sorted else None
    best_str = f"{_model_key(best)} @ {best['pass_at_1_plus']*100:.1f}%" if best else "—"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strix Halo Eval Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0e17; color: #c9d1d9;
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
    padding: 24px;
  }}
  .container {{ max-width: 1060px; margin: 0 auto; }}
  h1 {{ font-size: 22px; color: #58a6ff; margin-bottom: 4px; }}
  .subtitle {{ font-size: 13px; color: #6e7681; margin-bottom: 24px; }}
  .stats {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat {{
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 10px 16px; flex: 1 1 100px;
  }}
  .stat-value {{ font-size: 20px; font-weight: 700; color: #f0f6fc; }}
  .stat-label {{ font-size: 11px; color: #6e7681; text-transform: uppercase; letter-spacing: 1px; }}
  .chart-box {{
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; padding: 20px; margin-bottom: 24px;
  }}
  .empty-box {{ opacity: 0.5; }}
  .chart-box h2 {{ font-size: 14px; font-weight: 600; color: #f0f6fc; margin-bottom: 6px; }}
  .chart-sub {{ font-size: 11px; color: #6e7681; margin-bottom: 14px; }}
  canvas {{ max-height: 380px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{
    text-align: left; padding: 8px 10px;
    background: #161b22; color: #8b949e;
    border-bottom: 1px solid #21262d;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 10px;
  }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #161b22; }}
  tr:hover td {{ background: #161b22; }}
  .mono {{ font-family: 'JetBrains Mono', monospace; }}
  .dim {{ color: #6e7681; }}
  .score {{ font-weight: 700; }}
  .good {{ color: #3fb950; }}
  .mid  {{ color: #e3b341; }}
  .low  {{ color: #f85149; }}
  .status.ok   {{ color: #3fb950; }}
  .status.fail {{ color: #f85149; }}
</style>
</head>
<body>
<div class="container">
  <h1>🧪 Strix Halo Eval Results</h1>
  <p class="subtitle">
    {len(records)} runs &middot;
    {n_models} models &middot;
    {n_backends} backends &middot;
    {len(timestamps)} sessions
  </p>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">{len(records)}</div>
      <div class="stat-label">Total Runs</div>
    </div>
    <div class="stat">
      <div class="stat-value">{n_models}</div>
      <div class="stat-label">Models</div>
    </div>
    <div class="stat">
      <div class="stat-value">{len(scored_sorted)}</div>
      <div class="stat-label">With Scores</div>
    </div>
    <div class="stat" style="flex: 2 1 200px;">
      <div class="stat-value" style="font-size:14px; padding-top:3px;">{best_str}</div>
      <div class="stat-label">Best Plus pass@1</div>
    </div>
  </div>

  {score_section}

  {scatter_section}

  <div class="chart-box">
    <h2>Wall time — latest run per model (seconds, lower is faster)</h2>
    <canvas id="wallChart"></canvas>
  </div>

  {history_section}

  <div class="chart-box">
    <h2>Wall time — history over time</h2>
    <canvas id="wallHistoryChart"></canvas>
  </div>

  <div class="chart-box">
    <h2>All runs</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th><th>Quant</th><th>Backend</th><th>Suite</th>
          <th>Base pass@1</th><th>Plus pass@1</th>
          <th>Wall time</th><th>OK</th><th>Timestamp</th>
        </tr>
      </thead>
      <tbody>{table_rows_html}</tbody>
    </table>
  </div>

</div>
<script>
Chart.defaults.color = '#6e7681';
Chart.defaults.borderColor = '#21262d';

{score_chart_js}

{scatter_chart_js}

new Chart(document.getElementById('wallChart'), {{
  type: 'bar',
  data: {{
    labels: {time_labels_js},
    datasets: [{{
      data: {time_values_js},
      backgroundColor: {time_colors_js},
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
        callbacks: {{ label: ctx => (ctx.parsed.x / 60).toFixed(1) + ' min (' + ctx.parsed.x + 's)' }}
      }}
    }},
    scales: {{
      x: {{
        beginAtZero: true,
        title: {{ display: true, text: 'Seconds', font: {{ size: 11 }} }},
        ticks: {{ font: {{ size: 11 }} }}
      }},
      y: {{ ticks: {{ font: {{ size: 10 }} }} }}
    }}
  }}
}});

{history_chart_js}

new Chart(document.getElementById('wallHistoryChart'), {{
  type: 'line',
  data: {{
    labels: {timestamps_js},
    datasets: [{', '.join(wall_datasets_js)}]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ font: {{ size: 11 }}, usePointStyle: true, pointStyle: 'circle' }} }},
      tooltip: {{
        backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
        callbacks: {{ label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y/60).toFixed(1) + ' min' }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ font: {{ size: 10 }}, maxRotation: 45 }} }},
      y: {{
        beginAtZero: true,
        title: {{ display: true, text: 'Wall time (s)', font: {{ size: 11 }} }},
        ticks: {{ font: {{ size: 11 }} }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE

    if not path.exists():
        print(f"No eval file found at: {path}")
        print("Run './evaluate.sh MODEL' or './evaluate-all.sh' first.")
        sys.exit(1)

    records = load_records(path)
    if not records:
        print(f"No valid records in {path}")
        sys.exit(1)

    html = generate_html(records)
    out = Path(path).parent / "eval_report.html"
    out.write_text(html)
    print(f"Generated: {out}")
    print("Opening in browser ...")
    webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    main()
