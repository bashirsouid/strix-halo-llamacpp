#!/usr/bin/env python3
"""
Bench viewer — reads bench_results.jsonl and opens interactive charts
in your browser.  Shows per-payload-tier (small/medium/large) comparisons
and context degradation analysis.

Backward-compatible with older records that have no payload field (treated
as "small") or no gen_tok_s/pp_tok_s (uses avg_tok_s).

Usage:
    python bench_viewer.py                    # uses ./bench_results.jsonl
    python bench_viewer.py path/to/results.jsonl
"""

import json
import sys
import webbrowser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FILE = PROJECT_DIR / "results" / "benchmark" / "bench_results.jsonl"

PAYLOAD_ORDER = ["small", "medium", "large"]
PAYLOAD_DESC = {
    "small":  "~50 in / 256 out",
    "medium": "~1K in / 512 out",
    "large":  "~8K in / 2K out",
}
PAYLOAD_COLORS = {"small": "#2a9d8f", "medium": "#457b9d", "large": "#e63946"}
BACKEND_COLORS = {
    "radv":   {"gen": "#457b9d", "pp": "#a8dadc"},
    "rocm":   {"gen": "#e63946", "pp": "#f4a261"},
    "amdvlk": {"gen": "#2a9d8f", "pp": "#b5e48c"},
}


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
            if "payload" not in r:
                r["payload"] = "small"
            if "gen_tok_s" not in r:
                r["gen_tok_s"] = r.get("avg_tok_s", 0)
            if "pp_tok_s" not in r:
                r["pp_tok_s"] = 0
            if "combined_tok_s" not in r:
                r["combined_tok_s"] = r.get("avg_tok_s", r.get("gen_tok_s", 0))
            records.append(r)
        except json.JSONDecodeError:
            pass
    return records


def _model_label(r: dict) -> str:
    model = r["model"]
    quant = r.get("quant", "")
    return f"{model} [{quant}]" if quant else model


def _series_key(r: dict) -> str:
    return f"{r['model']} ({r['backend']})"


def generate_html(records: list[dict]) -> str:
    # ── Dedupe: latest record per (model, backend, payload) ──────────────
    latest: dict[tuple, dict] = {}
    for r in records:
        key = (r["model"], r["backend"], r["payload"])
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r

    recs = list(latest.values())
    models = sorted(set(r["model"] for r in recs))
    backends = sorted(set(r["backend"] for r in recs))
    payloads = sorted(
        set(r["payload"] for r in recs),
        key=lambda p: PAYLOAD_ORDER.index(p) if p in PAYLOAD_ORDER else 99,
    )
    has_pp = any(r["pp_tok_s"] > 0 for r in recs)

    def _get(model, backend, payload, field):
        r = latest.get((model, backend, payload))
        return r[field] if r else 0

    def _label(model):
        for r in recs:
            if r["model"] == model and r.get("quant"):
                return f"{model} [{r['quant']}]"
        return model

    model_labels = [_label(m) for m in models]
    model_labels_js = json.dumps(model_labels)

    # ── Chart 1: Generation tok/s at each tier, grouped by model ─────────
    # One dataset per payload tier, models on Y axis, side-by-side bars
    # Separate chart per backend
    gen_charts = {}
    for b in backends:
        datasets = []
        for p in payloads:
            color = PAYLOAD_COLORS.get(p, "#6e7681")
            data = [round(_get(m, b, p, "gen_tok_s"), 1) for m in models]
            datasets.append(f"""{{
                label: '{p} ({PAYLOAD_DESC.get(p, p)})',
                data: {json.dumps(data)},
                backgroundColor: '{color}',
                borderColor: '{color}',
                borderWidth: 1,
                borderRadius: 4,
            }}""")
        gen_charts[b] = ', '.join(datasets)

    # ── Chart 2: Prefill tok/s at each tier ──────────────────────────────
    pp_charts = {}
    if has_pp:
        for b in backends:
            datasets = []
            for p in payloads:
                color = PAYLOAD_COLORS.get(p, "#6e7681")
                data = [round(_get(m, b, p, "pp_tok_s"), 1) for m in models]
                datasets.append(f"""{{
                    label: '{p}',
                    data: {json.dumps(data)},
                    backgroundColor: '{color}88',
                    borderColor: '{color}',
                    borderWidth: 1,
                    borderRadius: 4,
                }}""")
            pp_charts[b] = ', '.join(datasets)

    # ── Chart 3: Degradation — gen tok/s retained from small → large ─────
    degrade_datasets = []
    for b in backends:
        bc = BACKEND_COLORS.get(b, BACKEND_COLORS["radv"])
        data = []
        for m in models:
            small_gen = _get(m, b, "small", "gen_tok_s")
            large_gen = _get(m, b, "large", "gen_tok_s")
            if small_gen > 0 and large_gen > 0:
                pct = round(large_gen / small_gen * 100, 1)
            else:
                pct = 0
            data.append(pct)
        degrade_datasets.append(f"""{{
            label: '{b.upper()}',
            data: {json.dumps(data)},
            backgroundColor: '{bc["gen"]}',
            borderColor: '{bc["gen"]}',
            borderWidth: 1,
            borderRadius: 4,
        }}""")
    degrade_datasets_js = ', '.join(degrade_datasets)

    # ── Chart 4: Backend comparison — stacked gen+pp per tier ────────────
    # For each backend+payload combo, stacked bar showing gen (solid) + pp (light)
    backend_compare_datasets = []
    for b in backends:
        bc = BACKEND_COLORS.get(b, BACKEND_COLORS["radv"])
        for p in payloads:
            gen_data = [round(_get(m, b, p, "gen_tok_s"), 1) for m in models]
            pp_data = [round(_get(m, b, p, "pp_tok_s"), 1) for m in models]
            stack_id = f"{b}_{p}"
            backend_compare_datasets.append(f"""{{
                label: '{b.upper()} {p} gen',
                data: {json.dumps(gen_data)},
                backgroundColor: '{bc["gen"]}',
                borderWidth: 0,
                stack: '{stack_id}',
                borderRadius: {{ bottomLeft: 4, bottomRight: 4, topLeft: 0, topRight: 0 }},
            }}""")
            if has_pp:
                backend_compare_datasets.append(f"""{{
                    label: '{b.upper()} {p} pp',
                    data: {json.dumps(pp_data)},
                    backgroundColor: '{bc["pp"]}',
                    borderWidth: 0,
                    stack: '{stack_id}',
                    borderRadius: {{ topLeft: 4, topRight: 4, bottomLeft: 0, bottomRight: 0 }},
                }}""")
    backend_compare_js = ', '.join(backend_compare_datasets)

    # ── Stats ────────────────────────────────────────────────────────────
    best_small = max((r for r in recs if r["payload"] == "small"),
                     key=lambda r: r["gen_tok_s"], default=None)
    best_large = max((r for r in recs if r["payload"] == "large"),
                     key=lambda r: r["gen_tok_s"], default=None)
    most_resilient = None
    if "small" in payloads and "large" in payloads:
        resilience = []
        for m in models:
            for b in backends:
                sg = _get(m, b, "small", "gen_tok_s")
                lg = _get(m, b, "large", "gen_tok_s")
                if sg > 0 and lg > 0:
                    resilience.append((m, b, lg / sg * 100))
        if resilience:
            most_resilient = max(resilience, key=lambda x: x[2])

    # ── Detailed table ───────────────────────────────────────────────────
    table_rows = []
    for m in models:
        label = _label(m)
        for b in backends:
            cells = [f'<td style="font-weight:600;white-space:nowrap;">{label}</td>',
                     f'<td>{b}</td>']
            for p in payloads:
                gen = _get(m, b, p, "gen_tok_s")
                pp = _get(m, b, p, "pp_tok_s")
                comb = _get(m, b, p, "combined_tok_s")
                if gen > 0:
                    cells.append(f'<td style="text-align:right;">{gen:.1f}</td>'
                                 f'<td style="text-align:right;color:#6e7681;">{pp:.0f}</td>'
                                 f'<td style="text-align:right;font-weight:600;">{comb:.1f}</td>')
                else:
                    cells.append('<td colspan="3" style="text-align:center;color:#6e7681;">—</td>')
            table_rows.append("<tr>" + "".join(cells) + "</tr>")
    table_html = "\n        ".join(table_rows)

    th_payload = ""
    th_sub = ""
    for p in payloads:
        pc = PAYLOAD_COLORS.get(p, "#6e7681")
        th_payload += f'<th colspan="3" style="text-align:center;border-bottom:2px solid {pc};">{p}</th>'
        th_sub += '<th style="text-align:right;font-size:10px;">Gen</th><th style="text-align:right;font-size:10px;">PP</th><th style="text-align:right;font-size:10px;">Comb</th>'

    # ── Build per-backend gen chart sections ─────────────────────────────
    bar_height = max(250, len(models) * 50 * len(payloads))
    gen_sections = []
    gen_inits = []
    for b in backends:
        chart_id = f"genChart_{b}"
        gen_sections.append(f"""
  <div class="chart-box">
    <h2>Generation tok/s by payload — {b.upper()}</h2>
    <div class="desc">How decode speed changes with context size.  Shorter bars at larger payloads = context degradation.</div>
    <div style="position:relative;height:{bar_height}px;">
      <canvas id="{chart_id}"></canvas>
    </div>
  </div>""")
        gen_inits.append(f"makeBar('{chart_id}', [{gen_charts[b]}], 'Generation tok/s');")

    pp_sections = []
    pp_inits = []
    if has_pp:
        for b in backends:
            chart_id = f"ppChart_{b}"
            pp_sections.append(f"""
  <div class="chart-box">
    <h2>Prefill tok/s by payload — {b.upper()}</h2>
    <div class="desc">Prompt processing speed at each context size.</div>
    <div style="position:relative;height:{bar_height}px;">
      <canvas id="{chart_id}"></canvas>
    </div>
  </div>""")
            pp_inits.append(f"makeBar('{chart_id}', [{pp_charts[b]}], 'Prefill tok/s');")

    gen_sections_html = "\n".join(gen_sections)
    pp_sections_html = "\n".join(pp_sections)
    gen_inits_js = "\n".join(gen_inits)
    pp_inits_js = "\n".join(pp_inits)

    stacked_height = max(350, len(models) * 50 * len(payloads) * len(backends))
    degrade_height = max(250, len(models) * 50 * len(backends))

    # ── Legend ────────────────────────────────────────────────────────────
    legend = " ".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;">'
        f'<span style="width:10px;height:10px;border-radius:2px;background:{PAYLOAD_COLORS.get(p, "#6e7681")};"></span>'
        f'{p}: {PAYLOAD_DESC.get(p, p)}</span>'
        for p in payloads
    )

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
  .container {{ max-width: 1060px; margin: 0 auto; }}
  h1 {{ font-size: 22px; color: #58a6ff; margin-bottom: 4px; }}
  .subtitle {{ font-size: 13px; color: #6e7681; margin-bottom: 24px; }}
  .stats {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat {{
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 10px 16px; flex: 1 1 160px;
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
    margin: 28px 0 14px; padding-top: 12px; border-top: 1px solid #21262d;
  }}
  .legend {{ font-size: 11px; color: #8b949e; margin-bottom: 16px; display: flex; flex-wrap: wrap; gap: 4px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid #21262d; }}
  th {{ color: #8b949e; font-weight: 500; }}
</style>
</head>
<body>
<div class="container">
  <h1>⚡ Strix Halo — benchmark results</h1>
  <p class="subtitle">
    {len(recs)} latest records &middot;
    {len(models)} models &middot;
    {len(backends)} backend(s) &middot;
    {len(payloads)} payload tier(s) &middot;
    {len(records)} total records
  </p>

  <div class="stats">
    {"" if not best_small else f'''<div class="stat">
      <div class="stat-value">{best_small["gen_tok_s"]:.0f}</div>
      <div class="stat-sub">{best_small["model"]} ({best_small["backend"]})</div>
      <div class="stat-label">Best gen tok/s (small)</div>
    </div>'''}
    {"" if not best_large else f'''<div class="stat">
      <div class="stat-value">{best_large["gen_tok_s"]:.0f}</div>
      <div class="stat-sub">{best_large["model"]} ({best_large["backend"]})</div>
      <div class="stat-label">Best gen tok/s (large)</div>
    </div>'''}
    {"" if not most_resilient else f'''<div class="stat">
      <div class="stat-value">{most_resilient[2]:.0f}%</div>
      <div class="stat-sub">{most_resilient[0]} ({most_resilient[1]})</div>
      <div class="stat-label">Most resilient (gen retained)</div>
    </div>'''}
  </div>

  <div class="legend">{legend}</div>

  <div class="section-label">Generation throughput by context size</div>
  {gen_sections_html}

  {"" if not has_pp else f'''
  <div class="section-label">Prefill throughput by context size</div>
  {pp_sections_html}
  '''}

  {"" if "large" not in payloads else f'''
  <div class="section-label">Context degradation</div>
  <div class="chart-box">
    <h2>Generation retained: small → large payload (%)</h2>
    <div class="desc">100% = no degradation.  Lower = model slows down more at long context.  Models that hold up best here are ideal for long documents and RAG.</div>
    <div style="position:relative;height:{degrade_height}px;">
      <canvas id="degradeChart"></canvas>
    </div>
  </div>
  '''}

  {"" if len(backends) < 2 else f'''
  <div class="section-label">Backend comparison — all tiers stacked</div>
  <div class="chart-box">
    <h2>Gen + prefill stacked by backend and payload</h2>
    <div class="desc">Side-by-side comparison across backends and payload sizes.  Each stack = gen (solid) + pp (light).</div>
    <div style="position:relative;height:{stacked_height}px;">
      <canvas id="backendChart"></canvas>
    </div>
  </div>
  '''}

  <div class="section-label">Detailed results</div>
  <div class="chart-box">
    <div style="overflow-x:auto;">
    <table>
      <thead>
        <tr><th>Model</th><th>Backend</th>{th_payload}</tr>
        <tr><th></th><th></th>{th_sub}</tr>
      </thead>
      <tbody>
        {table_html}
      </tbody>
    </table>
    </div>
  </div>
</div>

<script>
Chart.defaults.color = '#6e7681';
Chart.defaults.borderColor = '#21262d';

const modelLabels = {model_labels_js};

function makeBar(id, datasets, xLabel) {{
  new Chart(document.getElementById(id), {{
    type: 'bar',
    data: {{ labels: modelLabels, datasets: datasets }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ labels: {{ font: {{ size: 10 }}, usePointStyle: true, pointStyle: 'rect' }} }},
        tooltip: {{
          backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
          bodyFont: {{ size: 11 }},
          callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.x + ' tok/s' }}
        }}
      }},
      scales: {{
        x: {{ beginAtZero: true, title: {{ display: true, text: xLabel, font: {{ size: 11 }} }}, ticks: {{ font: {{ size: 11 }} }} }},
        y: {{ ticks: {{ font: {{ size: 11 }}, autoSkip: false }} }}
      }}
    }}
  }});
}}

{gen_inits_js}
{pp_inits_js}

{"" if "large" not in payloads else f"""
new Chart(document.getElementById('degradeChart'), {{
  type: 'bar',
  data: {{ labels: modelLabels, datasets: [{degrade_datasets_js}] }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ font: {{ size: 10 }}, usePointStyle: true, pointStyle: 'rect' }} }},
      tooltip: {{
        backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
        bodyFont: {{ size: 11 }},
        callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.x + '% retained' }}
      }}
    }},
    scales: {{
      x: {{ beginAtZero: true, max: 110, title: {{ display: true, text: '% of small-payload gen speed retained', font: {{ size: 11 }} }}, ticks: {{ font: {{ size: 11 }}, callback: v => v + '%' }} }},
      y: {{ ticks: {{ font: {{ size: 11 }}, autoSkip: false }} }}
    }}
  }}
}});
"""}

{"" if len(backends) < 2 else f"""
new Chart(document.getElementById('backendChart'), {{
  type: 'bar',
  data: {{ labels: modelLabels, datasets: [{backend_compare_js}] }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
        bodyFont: {{ size: 11 }},
        callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.x + ' tok/s' }}
      }}
    }},
    scales: {{
      x: {{ stacked: true, beginAtZero: true, title: {{ display: true, text: 'tok/s (gen + pp stacked)', font: {{ size: 11 }} }}, ticks: {{ font: {{ size: 11 }} }} }},
      y: {{ stacked: true, ticks: {{ font: {{ size: 11 }}, autoSkip: false }} }}
    }}
  }}
}});
"""}
</script>
</body>
</html>"""


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE

    if not path.exists():
        print(f"No benchmark file found at: {path}")
        print("Run './benchmark-run.sh MODEL' or './benchmark-run.sh --all' first.")
        sys.exit(1)

    records = load_records(path)
    if not records:
        print(f"No valid records in {path}")
        sys.exit(1)

    html = generate_html(records)

    out = Path(path).parent / "bench_report.html"
    out.write_text(html)
    print(f"Generated: {out}")
    print(f"Opening in browser ...")
    webbrowser.open(f"file://{out.resolve()}")


if __name__ == "__main__":
    main()
