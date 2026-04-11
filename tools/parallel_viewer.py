#!/usr/bin/env python3
"""
Parallel benchmark viewer — reads bench_parallel_results.jsonl and opens
interactive charts in your browser.  No dependencies beyond Python stdlib.

Shows performance at small, medium, and large payloads so you can find the
right parallelization level for your workload.  Backward-compatible with
older records that don't have a payload field (treated as "small").

Usage:
    python parallel_viewer.py                              # default file
    python parallel_viewer.py path/to/results.jsonl
"""

import json
import sys
import webbrowser
from pathlib import Path

try:
    from .report_helpers import stable_color, wrap_text_label
except ImportError:
    from report_helpers import stable_color, wrap_text_label

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FILE = PROJECT_DIR / "results" / "benchmark" / "bench_parallel_results.jsonl"

PAYLOAD_ORDER = ["small", "medium", "large"]
PAYLOAD_DESC = {
    "small":  "~50 input / 256 output (generation-dominant)",
    "medium": "~1K input / 512 output (balanced)",
    "large":  "~8K input / 2K output (prefill-heavy)",
}


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
            # Backward compat: old records have no "payload" field
            if "payload" not in r:
                r["payload"] = "small"
            records.append(r)
        except json.JSONDecodeError:
            pass
    return records


def _model_key(r: dict) -> str:
    return f"{r['model']} ({r['backend']})"


def _dedupe_latest(records: list[dict]) -> dict[str, dict[str, dict[int, dict]]]:
    """For each model+backend, payload, np → keep latest record.

    Returns: { model_key: { payload: { np: record } } }
    """
    grouped: dict[str, dict[str, dict[int, dict]]] = {}
    for r in records:
        key = _model_key(r)
        payload = r["payload"]
        np_val = r["np"]
        bucket = grouped.setdefault(key, {}).setdefault(payload, {})
        existing = bucket.get(np_val)
        if existing is None or r["timestamp"] > existing["timestamp"]:
            bucket[np_val] = r
    return grouped


def generate_html(records: list[dict]) -> str:
    grouped = _dedupe_latest(records)

    # Determine all np values and payloads present
    all_np = sorted(set(r["np"] for r in records))
    max_np = max(all_np) if all_np else 12
    np_labels = list(range(1, max_np + 1))
    np_labels_js = json.dumps(np_labels)

    payloads_present = sorted(
        set(r["payload"] for r in records),
        key=lambda p: PAYLOAD_ORDER.index(p) if p in PAYLOAD_ORDER else 99
    )

    model_keys = sorted(grouped.keys())
    model_colors = {mk: stable_color(mk) for mk in model_keys}
    summary_full_labels = model_keys
    summary_labels = [wrap_text_label(label, width=24) for label in summary_full_labels]
    summary_labels_js = json.dumps(summary_labels)
    summary_full_labels_js = json.dumps(summary_full_labels)

    # ── Per-payload aggregate tok/s line charts ──────────────────────────
    payload_charts_js = {}
    for payload in payloads_present:
        datasets = []
        for mk in model_keys:
            color = model_colors[mk]
            np_map = grouped[mk].get(payload, {})
            data = [np_map[n]["concurrent_agg_tok_s"] if n in np_map else "null"
                    for n in np_labels]
            datasets.append(f"""{{
                label: {json.dumps(mk)},
                data: [{', '.join(str(d) for d in data)}],
                borderColor: '{color}',
                backgroundColor: '{stable_color(mk, alpha=0.20)}',
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: '{color}',
                pointHoverRadius: 6,
                tension: 0.25,
                spanGaps: false,
            }}""")
        payload_charts_js[payload] = ', '.join(datasets)

    # ── Per-payload single-request line charts ───────────────────────────
    single_charts_js = {}
    for payload in payloads_present:
        datasets = []
        for mk in model_keys:
            color = model_colors[mk]
            np_map = grouped[mk].get(payload, {})
            data = [np_map[n]["single_tok_s"] if n in np_map else "null"
                    for n in np_labels]
            datasets.append(f"""{{
                label: {json.dumps(mk)},
                data: [{', '.join(str(d) for d in data)}],
                borderColor: '{color}',
                borderWidth: 2,
                pointRadius: 3,
                pointBackgroundColor: '{color}',
                tension: 0.25,
                spanGaps: false,
            }}""")
        single_charts_js[payload] = ', '.join(datasets)

    # ── Recommendation per model per payload ─────────────────────────────
    reco: dict[str, dict[str, dict]] = {}  # model_key → payload → best record
    for mk in model_keys:
        reco[mk] = {}
        for payload in payloads_present:
            np_map = grouped[mk].get(payload, {})
            if np_map:
                best = max(np_map.values(), key=lambda r: r["concurrent_agg_tok_s"])
                reco[mk][payload] = best

    # ── Summary: best np per model per payload (grouped bar) ─────────────
    summary_datasets_js = []
    payload_colors = {"small": "#2a9d8f", "medium": "#457b9d", "large": "#e63946"}
    for payload in payloads_present:
        color = payload_colors.get(payload, "#6e7681")
        data = []
        for mk in model_keys:
            best = reco[mk].get(payload)
            data.append(best["concurrent_agg_tok_s"] if best else 0)
        summary_datasets_js.append(f"""{{
            label: '{payload}',
            data: {json.dumps(data)},
            backgroundColor: '{color}',
            borderColor: '{color}',
            borderWidth: 1,
            borderRadius: 4,
        }}""")

    # ── Recommendation cards HTML ────────────────────────────────────────
    reco_cards = []
    for mk in model_keys:
        rows = []
        for payload in payloads_present:
            best = reco[mk].get(payload)
            if best:
                color = payload_colors.get(payload, "#6e7681")
                rows.append(
                    f'<div class="reco-row">'
                    f'<span class="reco-label" style="color:{color}">{payload}</span>'
                    f'<span class="reco-value">np={best["np"]}</span>'
                    f'<span class="reco-sub">{best["concurrent_agg_tok_s"]} agg tok/s</span>'
                    f'</div>'
                )
        reco_cards.append(
            f'<div class="reco-card">'
            f'<div class="reco-model">{mk}</div>'
            + "".join(rows) +
            f'</div>'
        )
    reco_html = "\n    ".join(reco_cards)

    # ── Stats ────────────────────────────────────────────────────────────
    n_records = len(records)
    n_models = len(model_keys)

    # Overall best
    all_latest = [r for mk_payloads in grouped.values()
                  for np_map in mk_payloads.values()
                  for r in np_map.values()]
    overall_best = max(all_latest, key=lambda r: r["concurrent_agg_tok_s"]) if all_latest else None

    # ── Generate payload chart sections ──────────────────────────────────
    payload_sections = []
    for idx, payload in enumerate(payloads_present):
        desc = PAYLOAD_DESC.get(payload, payload)
        agg_id = f"aggChart_{payload}"
        single_id = f"singleChart_{payload}"
        color = payload_colors.get(payload, "#6e7681")

        bar_height = max(250, n_models * 50)

        payload_sections.append(f"""
  <div class="section-label" style="border-color:{color};">{payload.upper()} payload — {desc}</div>

  <div class="chart-box">
    <h2>Aggregate throughput vs --parallel ({payload})</h2>
    <div class="desc">Total tok/s when all slots generate concurrently.  Find the peak — that's your optimal np for this payload.</div>
    <div style="position:relative;height:320px;">
      <canvas id="{agg_id}"></canvas>
    </div>
  </div>

  <div class="chart-box">
    <h2>Single-request latency vs --parallel ({payload})</h2>
    <div class="desc">Should stay flat.  If it drops, the slot count is hurting interactive use at this payload size.</div>
    <div style="position:relative;height:280px;">
      <canvas id="{single_id}"></canvas>
    </div>
  </div>
""")

    payload_sections_html = "\n".join(payload_sections)

    # ── Chart.js init for all payload charts ─────────────────────────────
    chart_inits = []
    for payload in payloads_present:
        agg_id = f"aggChart_{payload}"
        single_id = f"singleChart_{payload}"
        chart_inits.append(f"""
makeLineChart('{agg_id}', [{payload_charts_js[payload]}], 'Aggregate tok/s');
makeLineChart('{single_id}', [{single_charts_js[payload]}], 'Single-request tok/s');
""")
    chart_inits_js = "\n".join(chart_inits)

    summary_bar_height = max(250, n_models * 50 * len(payloads_present))

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
  .stats {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat {{
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 10px 16px; flex: 1 1 140px;
  }}
  .stat-value {{ font-size: 20px; font-weight: 700; color: #f0f6fc; }}
  .stat-sub {{ font-size: 11px; color: #6e7681; }}
  .stat-label {{ font-size: 11px; color: #6e7681; text-transform: uppercase; letter-spacing: 1px; }}
  .chart-box {{
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; padding: 20px; margin-bottom: 24px;
  }}
  .chart-box h2 {{ font-size: 14px; font-weight: 600; color: #f0f6fc; margin-bottom: 4px; }}
  .chart-box .desc {{ font-size: 11px; color: #6e7681; margin-bottom: 12px; }}
  .section-label {{
    font-size: 16px; font-weight: 700; color: #f0f6fc;
    margin: 28px 0 14px; padding-top: 12px;
    border-top: 3px solid #21262d;
  }}
  .reco-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }}
  .reco-card {{
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 14px 16px;
  }}
  .reco-model {{ font-size: 13px; font-weight: 600; color: #f0f6fc; margin-bottom: 10px; overflow-wrap: anywhere; }}
  .reco-row {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 12px; padding: 3px 0;
  }}
  .reco-label {{ width: 60px; font-weight: 600; }}
  .reco-value {{ color: #f0f6fc; font-weight: 600; width: 50px; }}
  .reco-sub {{ color: #6e7681; font-size: 11px; }}
  .legend {{
    font-size: 11px; color: #8b949e; margin-bottom: 16px;
    display: flex; flex-wrap: wrap; gap: 4px 16px;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>⚡ Parallel sweep results</h1>
  <p class="subtitle">
    {n_records} measurements &middot;
    {n_models} model(s) &middot;
    {len(payloads_present)} payload tier(s) &middot;
    --parallel 1&ndash;{max_np}
  </p>

  <div class="stats">
    {"" if not overall_best else f'''
    <div class="stat">
      <div class="stat-value">{overall_best["concurrent_agg_tok_s"]}</div>
      <div class="stat-sub">{_model_key(overall_best)} · np={overall_best["np"]} · {overall_best["payload"]}</div>
      <div class="stat-label">Best aggregate tok/s</div>
    </div>
    '''}
    <div class="stat">
      <div class="stat-value">{n_models}</div>
      <div class="stat-label">Models tested</div>
    </div>
    <div class="stat">
      <div class="stat-value">{len(payloads_present)}</div>
      <div class="stat-label">Payload tiers</div>
    </div>
  </div>

  <div class="section-label">Recommendations per model</div>
  <div class="legend">
    {"".join(f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:2px;background:{payload_colors.get(p, "#6e7681")};"></span>{p}: {PAYLOAD_DESC.get(p, p)}</span>' for p in payloads_present)}
  </div>
  <div class="reco-grid">
    {reco_html}
  </div>

  <div class="section-label">Best aggregate throughput by payload size</div>
  <div class="chart-box">
    <h2>Best aggregate tok/s per model (at optimal np for each payload)</h2>
    <div class="desc">Shows how each model's peak throughput changes with payload size.  Large payloads stress prefill + context.</div>
    <div style="position:relative;height:{summary_bar_height}px;">
      <canvas id="summaryChart"></canvas>
    </div>
  </div>

  {payload_sections_html}

</div>

<script>
Chart.defaults.color = '#6e7681';
Chart.defaults.borderColor = '#21262d';

const npLabels = {np_labels_js};

function makeLineChart(id, datasets, yLabel) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{ labels: npLabels, datasets: datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ labels: {{ font: {{ size: 10 }}, usePointStyle: true, pointStyle: 'circle' }} }},
        tooltip: {{
          backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
          bodyFont: {{ size: 11 }},
          callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y + ' tok/s' }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: '--parallel (np)', font: {{ size: 11 }} }},
          ticks: {{ font: {{ size: 11 }} }},
        }},
        y: {{
          beginAtZero: true,
          title: {{ display: true, text: yLabel, font: {{ size: 11 }} }},
          ticks: {{ font: {{ size: 11 }} }},
        }}
      }}
    }}
  }});
}}

new Chart(document.getElementById('summaryChart'), {{
  type: 'bar',
  data: {{
    labels: {summary_labels_js},
    datasets: [{', '.join(summary_datasets_js)}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ font: {{ size: 11 }}, usePointStyle: true, pointStyle: 'rect' }} }},
      tooltip: {{
        backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
        bodyFont: {{ size: 11 }},
        callbacks: {{
          title: items => {summary_full_labels_js}[items[0].dataIndex],
          label: ctx => ctx.dataset.label + ': ' + ctx.parsed.x + ' tok/s'
        }}
      }}
    }},
    scales: {{
      x: {{ beginAtZero: true, title: {{ display: true, text: 'Aggregate tok/s', font: {{ size: 11 }} }}, ticks: {{ font: {{ size: 11 }} }} }},
      y: {{ ticks: {{ font: {{ size: 11 }}, autoSkip: false }} }}
    }}
  }}
}});

{chart_inits_js}
</script>
</body>
</html>"""


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE

    if not path.exists():
        print(f"No parallel benchmark file found at: {path}")
        print("Run './benchmark-run.sh --parallel MODEL' first to generate results.")
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
