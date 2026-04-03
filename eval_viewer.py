#!/usr/bin/env python3
"""
Eval viewer — reads eval_results.jsonl and opens a small chart in your browser.

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


def _series_key(r: dict) -> str:
    quant = r.get("quant") or ""
    suite = r.get("suite", "")
    model = r["model"]
    backend = r["backend"]
    base = f"{model} ({backend}, {suite})"
    return f"{base} [{quant}]" if quant else base


def generate_html(records: list[dict]) -> str:
    series: dict[str, list[dict]] = {}
    for r in records:
        key = _series_key(r)
        series.setdefault(key, []).append(r)

    timestamps = sorted(set(r["timestamp"] for r in records))

    palette = [
        "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
        "#264653", "#a8dadc", "#6d6875", "#b5838d", "#ffb703",
        "#023047", "#8ecae6",
    ]

    datasets_js = []
    for i, (key, recs) in enumerate(sorted(series.items())):
        color = palette[i % len(palette)]
        lookup = {r["timestamp"]: r["wall_time_sec"] for r in recs}
        data_points = [lookup.get(ts, "null") for ts in timestamps]
        datasets_js.append(f"""{{
            label: {json.dumps(key)},
            data: [{', '.join(str(d) for d in data_points)}],
            borderColor: '{color}',
            backgroundColor: '{color}33',
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0.2,
            spanGaps: true,
        }}""")

    labels_js = json.dumps(timestamps)

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
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 22px; color: #58a6ff; margin-bottom: 4px; }}
  .subtitle {{ font-size: 13px; color: #6e7681; margin-bottom: 24px; }}
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
  <h1>🧪 Strix Halo Eval Results</h1>
  <p class="subtitle">
    {len(records)} records &middot;
    {len(series)} model/backend/suite combos &middot;
    {len(timestamps)} sessions
  </p>

  <div class="chart-box">
    <h2>Eval wall time over time (seconds)</h2>
    <canvas id="lineChart"></canvas>
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
        callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y + ' s' }}
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
        print(f"Run './evaluate.sh MODEL' or './evaluate-all.sh' first.")
        sys.exit(1)

    records = load_records(path)
    if not records:
        print(f"No valid records in {path}")
        sys.exit(1)

    html = generate_html(records)
    out = Path(path).parent / "eval_report.html"
    out.write_text(html)
    print(f"Generated: {out}")
    print(f"Opening in browser ...")
    webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    main()