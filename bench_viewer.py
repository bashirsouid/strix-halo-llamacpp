#!/usr/bin/env python3
"""
Bench viewer — reads bench_results.jsonl and opens an interactive chart
in your browser.  No dependencies beyond Python stdlib.

Usage:
    python bench_viewer.py                    # uses ./bench_results.jsonl
    python bench_viewer.py path/to/results.jsonl
"""

import json
import sys
import tempfile
import webbrowser
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent / "bench_results.jsonl"


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


def generate_html(records: list[dict]) -> str:
    # Build datasets grouped by model+backend
    series: dict[str, list[dict]] = {}
    for r in records:
        key = f"{r['model']} ({r['backend']})"
        series.setdefault(key, []).append(r)

    # Get sorted unique timestamps
    timestamps = sorted(set(r["timestamp"] for r in records))

    # Colors
    palette = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
        "#264653", "#a8dadc", "#6d6875", "#b5838d", "#ffb703",
        "#023047", "#8ecae6",
    ]

    # Build Chart.js datasets
    datasets_js = []
    for i, (key, recs) in enumerate(sorted(series.items())):
        color = palette[i % len(palette)]
        is_amdvlk = "amdvlk" in key
        lookup = {r["timestamp"]: r["avg_tok_s"] for r in recs}
        data_points = [lookup.get(ts, "null") for ts in timestamps]
        datasets_js.append(f"""{{
            label: {json.dumps(key)},
            data: [{', '.join(str(d) for d in data_points)}],
            borderColor: '{color}',
            backgroundColor: '{color}33',
            borderWidth: 2,
            borderDash: {[6, 3] if is_amdvlk else []},
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0.2,
            spanGaps: true,
        }}""")

    # Latest snapshot for bar chart
    latest: dict[str, dict] = {}
    for r in records:
        key = f"{r['model']} ({r['backend']})"
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r
    bar_items = sorted(latest.values(), key=lambda x: -x["avg_tok_s"])
    bar_labels = [f"{r['model']} ({r['backend']})" for r in bar_items]
    bar_values = [r["avg_tok_s"] for r in bar_items]
    bar_colors = [palette[i % len(palette)] for i in range(len(bar_items))]

    labels_js = json.dumps(timestamps)
    bar_labels_js = json.dumps(bar_labels)
    bar_values_js = json.dumps(bar_values)
    bar_colors_js = json.dumps(bar_colors)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strix Halo Bench Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0e17; color: #c9d1d9;
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
    padding: 24px;
  }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 22px; color: #58a6ff; margin-bottom: 4px; }}
  .subtitle {{ font-size: 13px; color: #6e7681; margin-bottom: 24px; }}
  .stats {{
    display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap;
  }}
  .stat {{
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 10px 16px; flex: 1 1 100px;
  }}
  .stat-value {{ font-size: 20px; font-weight: 700; color: #f0f6fc; }}
  .stat-label {{
    font-size: 11px; color: #6e7681;
    text-transform: uppercase; letter-spacing: 1px;
  }}
  .chart-box {{
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; padding: 20px; margin-bottom: 24px;
  }}
  .chart-box h2 {{
    font-size: 14px; font-weight: 600; color: #f0f6fc;
    margin-bottom: 12px;
  }}
  canvas {{ max-height: 350px; }}
</style>
</head>
<body>
<div class="container">
  <h1>⚡ Strix Halo Bench Results</h1>
  <p class="subtitle">
    {len(records)} records &middot;
    {len(series)} model+backend combos &middot;
    {len(timestamps)} sessions
  </p>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">{len(records)}</div>
      <div class="stat-label">Records</div>
    </div>
    <div class="stat">
      <div class="stat-value">{len(set(r['model'] for r in records))}</div>
      <div class="stat-label">Models</div>
    </div>
    <div class="stat">
      <div class="stat-value">{len(timestamps)}</div>
      <div class="stat-label">Sessions</div>
    </div>
    <div class="stat">
      <div class="stat-value">{len(set(r['backend'] for r in records))}</div>
      <div class="stat-label">Backends</div>
    </div>
  </div>

  <div class="chart-box">
    <h2>tok/s over time</h2>
    <canvas id="lineChart"></canvas>
  </div>

  <div class="chart-box">
    <h2>Latest results (ranked)</h2>
    <canvas id="barChart"></canvas>
  </div>
</div>

<script>
Chart.defaults.color = '#6e7681';
Chart.defaults.borderColor = '#21262d';

new Chart(document.getElementById('lineChart'), {{
  type: 'line',
  data: {{
    labels: {labels_js},
    datasets: [{', '.join(datasets_js)}]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ font: {{ size: 11 }}, usePointStyle: true, pointStyle: 'circle' }} }},
      tooltip: {{
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        borderWidth: 1,
        titleFont: {{ size: 12 }},
        bodyFont: {{ size: 12 }},
        callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y + ' tok/s' }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ font: {{ size: 10 }}, maxRotation: 45 }} }},
      y: {{
        beginAtZero: true,
        title: {{ display: true, text: 'tok/s', font: {{ size: 11 }} }},
        ticks: {{ font: {{ size: 11 }} }}
      }}
    }}
  }}
}});

new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: {bar_labels_js},
    datasets: [{{
      data: {bar_values_js},
      backgroundColor: {bar_colors_js},
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        borderWidth: 1,
        callbacks: {{ label: ctx => ctx.parsed.x + ' tok/s' }}
      }}
    }},
    scales: {{
      x: {{
        beginAtZero: true,
        title: {{ display: true, text: 'tok/s', font: {{ size: 11 }} }},
        ticks: {{ font: {{ size: 11 }} }}
      }},
      y: {{ ticks: {{ font: {{ size: 11 }} }} }}
    }}
  }}
}});
</script>
</body>
</html>"""


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE

    if not path.exists():
        print(f"No benchmark file found at: {path}")
        print(f"Run 'python server.py bench MODEL' first to generate results.")
        sys.exit(1)

    records = load_records(path)
    if not records:
        print(f"No valid records in {path}")
        sys.exit(1)

    html = generate_html(records)

    # Write to a temp file and open in browser
    out = Path(path).parent / "bench_report.html"
    out.write_text(html)
    print(f"Generated: {out}")
    print(f"Opening in browser ...")
    webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    main()
