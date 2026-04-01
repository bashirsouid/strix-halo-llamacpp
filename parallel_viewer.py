#!/usr/bin/env python3
"""
Parallel benchmark viewer — reads bench_parallel_results.jsonl and opens
interactive charts in your browser.  No dependencies beyond Python stdlib.

For each model, shows the most recent measurement at each --parallel value.

Usage:
    python parallel_viewer.py                              # default file
    python parallel_viewer.py path/to/results.jsonl
"""

import json
import sys
import webbrowser
from pathlib import Path

DEFAULT_FILE = Path(__file__).resolve().parent / "bench_parallel_results.jsonl"


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
    """Display label for a model series."""
    return f"{r['model']} ({r['backend']})"


def _dedupe_latest(records: list[dict]) -> dict[str, dict[int, dict]]:
    """For each model, keep only the most recent record per np value.

    Returns: { model_key: { np: record, ... }, ... }
    """
    grouped: dict[str, dict[int, dict]] = {}
    for r in records:
        key = _model_key(r)
        np = r["np"]
        existing = grouped.setdefault(key, {}).get(np)
        if existing is None or r["timestamp"] > existing["timestamp"]:
            grouped[key][np] = r
    return grouped


def generate_html(records: list[dict]) -> str:
    grouped = _dedupe_latest(records)

    # Palette — enough for many models
    palette = [
        "#2a9d8f", "#7f77dd", "#457b9d", "#e63946", "#e9c46a",
        "#f4a261", "#264653", "#b5838d", "#ffb703", "#8ecae6",
        "#6d6875", "#023047",
    ]

    all_np = sorted(set(r["np"] for r in records))
    max_np = max(all_np) if all_np else 12
    np_labels = list(range(1, max_np + 1))
    np_labels_js = json.dumps(np_labels)

    # ── Build Chart.js datasets for each chart type ──────────────────────
    def _make_datasets(field: str) -> str:
        datasets = []
        for i, (key, np_map) in enumerate(sorted(grouped.items())):
            color = palette[i % len(palette)]
            data = [np_map[n][field] if n in np_map else "null"
                    for n in np_labels]
            datasets.append(f"""{{
                label: {json.dumps(key)},
                data: [{', '.join(str(d) for d in data)}],
                borderColor: '{color}',
                backgroundColor: '{color}33',
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: '{color}',
                pointHoverRadius: 6,
                tension: 0.25,
                spanGaps: false,
            }}""")
        return ', '.join(datasets)

    agg_datasets = _make_datasets("concurrent_agg_tok_s")
    single_datasets = _make_datasets("single_tok_s")
    perreq_datasets = _make_datasets("concurrent_per_req_tok_s")

    # ── Summary stats ────────────────────────────────────────────────────
    n_records = len(records)
    n_models = len(grouped)

    # Best aggregate per model
    best_per_model = []
    for key, np_map in sorted(grouped.items()):
        best_rec = max(np_map.values(), key=lambda r: r["concurrent_agg_tok_s"])
        best_per_model.append((key, best_rec))

    overall_best = max(best_per_model, key=lambda x: x[1]["concurrent_agg_tok_s"])

    # ── Recommendation cards HTML ────────────────────────────────────────
    reco_cards = []
    for key, rec in best_per_model:
        np_val = rec["np"]
        agg = rec["concurrent_agg_tok_s"]
        single = rec["single_tok_s"]
        is_best = (key == overall_best[0])
        border = "border: 1px solid #58a6ff;" if is_best else "border: 1px solid #21262d;"
        star = " ⭐" if is_best else ""
        reco_cards.append(f"""
        <div class="reco-card" style="{border}">
          <div class="reco-model">{key}{star}</div>
          <div class="reco-row">
            <span class="reco-label">Best --parallel</span>
            <span class="reco-value">{np_val}</span>
          </div>
          <div class="reco-row">
            <span class="reco-label">Aggregate tok/s</span>
            <span class="reco-value">{agg}</span>
          </div>
          <div class="reco-row">
            <span class="reco-label">Single tok/s</span>
            <span class="reco-value">{single}</span>
          </div>
        </div>""")

    reco_html = "\n".join(reco_cards)

    # ── Ranked bar chart data (best aggregate per model) ─────────────────
    bar_items = sorted(best_per_model, key=lambda x: -x[1]["concurrent_agg_tok_s"])
    bar_labels_js = json.dumps([f"{k}  np={r['np']}" for k, r in bar_items])
    bar_agg_js = json.dumps([r["concurrent_agg_tok_s"] for _, r in bar_items])
    bar_single_js = json.dumps([r["single_tok_s"] for _, r in bar_items])
    bar_colors_js = json.dumps([palette[i % len(palette)]
                                for i in range(len(bar_items))])

    # ── Wall time dataset ────────────────────────────────────────────────
    wall_datasets = _make_datasets("wall_time")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parallel Sweep Results — Strix Halo</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0e17; color: #c9d1d9;
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
    padding: 24px;
  }}
  .container {{ max-width: 1020px; margin: 0 auto; }}
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
    margin-bottom: 4px;
  }}
  .chart-box .chart-desc {{
    font-size: 11px; color: #6e7681; margin-bottom: 12px;
  }}
  canvas {{ max-height: 340px; }}
  .reco-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }}
  .reco-card {{
    background: #161b22; border-radius: 8px; padding: 14px 16px;
  }}
  .reco-model {{
    font-size: 13px; font-weight: 600; color: #f0f6fc;
    margin-bottom: 10px; line-height: 1.3;
  }}
  .reco-row {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 12px; padding: 3px 0;
  }}
  .reco-label {{ color: #6e7681; }}
  .reco-value {{ color: #c9d1d9; font-weight: 600; }}
  .section-label {{
    font-size: 16px; font-weight: 700; color: #f0f6fc;
    margin: 28px 0 14px; padding-top: 12px;
    border-top: 1px solid #21262d;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>⚡ Parallel sweep results</h1>
  <p class="subtitle">
    {n_records} measurements &middot;
    {n_models} model(s) &middot;
    --parallel 1&ndash;{max_np}
  </p>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">{overall_best[1]["concurrent_agg_tok_s"]}</div>
      <div class="stat-label">Best aggregate tok/s</div>
    </div>
    <div class="stat">
      <div class="stat-value">np={overall_best[1]["np"]}</div>
      <div class="stat-label">Best --parallel</div>
    </div>
    <div class="stat">
      <div class="stat-value">{overall_best[0].split(" (")[0]}</div>
      <div class="stat-label">Fastest model</div>
    </div>
    <div class="stat">
      <div class="stat-value">{n_models}</div>
      <div class="stat-label">Models tested</div>
    </div>
  </div>

  <div class="section-label">Recommendations per model</div>
  <div class="reco-grid">
    {reco_html}
  </div>

  <div class="section-label">Charts</div>

  <div class="chart-box">
    <h2>Aggregate throughput vs --parallel</h2>
    <p class="chart-desc">Total tok/s when all slots are generating concurrently.  Higher is better for multi-request workloads.</p>
    <canvas id="aggChart"></canvas>
  </div>

  <div class="chart-box">
    <h2>Single-request latency vs --parallel</h2>
    <p class="chart-desc">tok/s for a single request at each --parallel setting.  Should stay flat — if it drops, the slot count is hurting interactive use.</p>
    <canvas id="singleChart"></canvas>
  </div>

  <div class="chart-box">
    <h2>Per-request throughput under full load</h2>
    <p class="chart-desc">tok/s each individual request gets when all slots are busy.  Drops as slots increase — that's expected.</p>
    <canvas id="perreqChart"></canvas>
  </div>

  <div class="chart-box">
    <h2>Wall time for full concurrent batch</h2>
    <p class="chart-desc">Seconds to complete all N concurrent requests.  Lower is better.</p>
    <canvas id="wallChart"></canvas>
  </div>

  <div class="chart-box">
    <h2>Best aggregate by model (ranked)</h2>
    <canvas id="barChart"></canvas>
  </div>
</div>

<script>
Chart.defaults.color = '#6e7681';
Chart.defaults.borderColor = '#21262d';

const npLabels = {np_labels_js};

const commonOpts = {{
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
    }}
  }},
  scales: {{
    x: {{
      title: {{ display: true, text: '--parallel (np)', font: {{ size: 11 }} }},
      ticks: {{ font: {{ size: 11 }} }},
    }},
    y: {{
      beginAtZero: true,
      ticks: {{ font: {{ size: 11 }} }},
    }}
  }}
}};

function makeLineChart(id, datasets, yLabel, unit, yMax) {{
  const opts = JSON.parse(JSON.stringify(commonOpts));
  opts.scales.y.title = {{ display: true, text: yLabel, font: {{ size: 11 }} }};
  if (yMax) opts.scales.y.max = yMax;
  opts.plugins.tooltip.callbacks = {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y + ' ' + unit }};
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{ labels: npLabels, datasets: datasets }},
    options: opts,
  }});
}}

makeLineChart('aggChart', [{agg_datasets}], 'Aggregate tok/s', 'tok/s', null);
makeLineChart('singleChart', [{single_datasets}], 'Single-request tok/s', 'tok/s', null);
makeLineChart('perreqChart', [{perreq_datasets}], 'Per-request tok/s', 'tok/s', null);
makeLineChart('wallChart', [{wall_datasets}], 'Wall time (s)', 's', null);

new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: {bar_labels_js},
    datasets: [
      {{
        label: 'Aggregate tok/s',
        data: {bar_agg_js},
        backgroundColor: {bar_colors_js},
        borderRadius: 4,
      }},
      {{
        label: 'Single tok/s',
        data: {bar_single_js},
        backgroundColor: {bar_colors_js}.map(c => c + '66'),
        borderRadius: 4,
      }}
    ]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ font: {{ size: 11 }}, usePointStyle: true, pointStyle: 'rect' }} }},
      tooltip: {{
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        borderWidth: 1,
        callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.x + ' tok/s' }}
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
        print(f"No parallel benchmark file found at: {path}")
        print(f"Run 'python server.py bench-parallel MODEL' first to generate results.")
        sys.exit(1)

    records = load_records(path)
    if not records:
        print(f"No valid records in {path}")
        sys.exit(1)

    html = generate_html(records)

    out = Path(path).parent / "parallel_report.html"
    out.write_text(html)
    print(f"Generated: {out}")
    print(f"Opening in browser ...")
    webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    main()
