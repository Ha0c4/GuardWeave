#!/usr/bin/env python3
"""Render a Chinese HTML/PDF report for a multi-run GuardWeave experiment suite."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return escape(str(value))


def render_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head_html = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{fmt(cell)}</td>" for cell in row) + "</tr>")
    return "<table><thead><tr>" + head_html + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def render_report(suite_dir: Path, title: str) -> str:
    suite_summary = read_json(suite_dir / "suite_summary.json")
    suite_runs = read_jsonl(suite_dir / "suite_runs.jsonl")

    metric_keys = [
        ("effective_attack_count_mean", "平均有效攻击数"),
        ("effective_attack_rate_mean", "平均有效攻击率"),
        ("malicious_defended_violation_rate_mean", "恶意样本防御后违背率均值"),
        ("malicious_violation_rate_reduction_mean", "恶意样本违背率下降均值"),
        ("malicious_defended_post_block_rate_mean", "恶意样本后置拦截率均值"),
        ("benign_false_refusal_rate_mean", "良性误拒率均值"),
    ]

    kpis = []
    for key, label in metric_keys:
        value = float(suite_summary.get(key, 0.0))
        display = rate(value) if "rate" in key else f"{value:.2f}"
        kpis.append((label, display))

    best_run = max(suite_runs, key=lambda row: row.get("malicious_violation_rate_reduction", 0.0))
    worst_run = max(suite_runs, key=lambda row: row.get("malicious_defended_violation_rate", 0.0))

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    @page {{ size: A4; margin: 16mm; }}
    body {{
      margin: 0;
      background: #f4f7fb;
      color: #152131;
      font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      line-height: 1.6;
      font-size: 13px;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      background: #fff;
      padding: 24px 28px 40px;
    }}
    h1, h2, h3 {{ color: #13365b; margin-top: 0; }}
    h2 {{
      margin-top: 26px;
      padding-bottom: 6px;
      border-bottom: 2px solid #d9e3f0;
    }}
    .meta, .kpis {{
      display: grid;
      gap: 12px;
    }}
    .meta {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .kpis {{ grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 14px; }}
    .card {{
      background: #f2f7ff;
      border: 1px solid #d9e5f6;
      border-radius: 12px;
      padding: 14px 16px;
    }}
    .card .label {{
      color: #5d7289;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .card .value {{
      color: #0d355f;
      font-weight: 700;
      font-size: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      word-break: break-word;
      margin: 10px 0 18px;
    }}
    th, td {{
      border: 1px solid #d8e2ee;
      padding: 8px 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{ background: #eef4fb; }}
    code {{
      background: #f1f4f8;
      border-radius: 4px;
      padding: 1px 4px;
      font-family: "SF Mono", "Menlo", monospace;
      font-size: 11px;
    }}
    .footnote {{ color: #5d7289; font-size: 12px; }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <div class="meta">
      <div class="card"><div class="label">总运行次数</div><div class="value">{suite_summary.get("run_count", 0)}</div></div>
      <div class="card"><div class="label">Base Model</div><div class="value" style="font-size:18px">{escape(str(suite_summary.get("base_model", "")))}</div></div>
      <div class="card"><div class="label">Judge Model</div><div class="value" style="font-size:18px">{escape(str(suite_summary.get("judge_model", "")))}</div></div>
    </div>

    <h2>均值概览</h2>
    <div class="kpis">
      {"".join(f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>' for label, value in kpis)}
    </div>

    <h2>总体结论</h2>
    <p>本套件共完成 <strong>{suite_summary.get("run_count", 0)}</strong> 次重复实验。每次实验都使用 <code>{escape(str(suite_summary.get("base_model", "")))}</code> 作为无防御 / 有防御的 base model，并使用同一组 3B 本地 judge 作为 GuardWeave 风险与输出审查器。每次实验完整扫描 <strong>{suite_summary.get("candidate_attack_pool_size", 0)}</strong> 条攻击候选，要求至少 <strong>{suite_summary.get("min_effective_attacks", 0)}</strong> 条无防御有效攻击，最终均值上达到了 <strong>{suite_summary.get("effective_attack_count_mean", 0):.2f}</strong> 条有效攻击。</p>
    <p>从重复实验结果看，GuardWeave 在恶意样本上的防御后违背率均值为 <strong>{rate(float(suite_summary.get("malicious_defended_violation_rate_mean", 0.0)))}</strong>，相对无防御基线的平均违背率下降为 <strong>{rate(float(suite_summary.get("malicious_violation_rate_reduction_mean", 0.0)))}</strong>；良性误拒率均值为 <strong>{rate(float(suite_summary.get("benign_false_refusal_rate_mean", 0.0)))}</strong>。</p>

    <h2>均值与波动</h2>
    {render_table(
        ["指标", "均值", "标准差", "最小值", "最大值"],
        [
            ["有效攻击数", f"{suite_summary.get('effective_attack_count_mean', 0):.2f}", f"{suite_summary.get('effective_attack_count_std', 0):.2f}", suite_summary.get('effective_attack_count_min', 0), suite_summary.get('effective_attack_count_max', 0)],
            ["有效攻击率", rate(float(suite_summary.get('effective_attack_rate_mean', 0.0))), rate(float(suite_summary.get('effective_attack_rate_std', 0.0))), rate(float(suite_summary.get('effective_attack_rate_min', 0.0))), rate(float(suite_summary.get('effective_attack_rate_max', 0.0)))],
            ["恶意样本防御后违背率", rate(float(suite_summary.get('malicious_defended_violation_rate_mean', 0.0))), rate(float(suite_summary.get('malicious_defended_violation_rate_std', 0.0))), rate(float(suite_summary.get('malicious_defended_violation_rate_min', 0.0))), rate(float(suite_summary.get('malicious_defended_violation_rate_max', 0.0)))],
            ["恶意样本违背率下降", rate(float(suite_summary.get('malicious_violation_rate_reduction_mean', 0.0))), rate(float(suite_summary.get('malicious_violation_rate_reduction_std', 0.0))), rate(float(suite_summary.get('malicious_violation_rate_reduction_min', 0.0))), rate(float(suite_summary.get('malicious_violation_rate_reduction_max', 0.0)))],
            ["恶意样本后置拦截率", rate(float(suite_summary.get('malicious_defended_post_block_rate_mean', 0.0))), rate(float(suite_summary.get('malicious_defended_post_block_rate_std', 0.0))), rate(float(suite_summary.get('malicious_defended_post_block_rate_min', 0.0))), rate(float(suite_summary.get('malicious_defended_post_block_rate_max', 0.0)))],
            ["良性误拒率", rate(float(suite_summary.get('benign_false_refusal_rate_mean', 0.0))), rate(float(suite_summary.get('benign_false_refusal_rate_std', 0.0))), rate(float(suite_summary.get('benign_false_refusal_rate_min', 0.0))), rate(float(suite_summary.get('benign_false_refusal_rate_max', 0.0)))],
        ],
    )}

    <h2>最佳与最弱运行</h2>
    {render_table(
        ["类型", "运行目录", "seed", "有效攻击数", "恶意样本防御后违背率", "恶意样本违背率下降", "良性误拒率"],
        [
            ["最佳防御运行", best_run.get("run_name", ""), best_run.get("seed", ""), best_run.get("effective_attack_count", 0), rate(float(best_run.get("malicious_defended_violation_rate", 0.0))), rate(float(best_run.get("malicious_violation_rate_reduction", 0.0))), rate(float(best_run.get("benign_false_refusal_rate", 0.0)))],
            ["最难运行", worst_run.get("run_name", ""), worst_run.get("seed", ""), worst_run.get("effective_attack_count", 0), rate(float(worst_run.get("malicious_defended_violation_rate", 0.0))), rate(float(worst_run.get("malicious_violation_rate_reduction", 0.0))), rate(float(worst_run.get("benign_false_refusal_rate", 0.0)))],
        ],
    )}

    <h2>逐次运行明细</h2>
    {render_table(
        ["运行", "seed", "有效攻击数", "恶意样本数", "恶意样本防御后违背率", "恶意样本违背率下降", "后置拦截率", "误拒率"],
        [
            [
                row.get("run_name", ""),
                row.get("seed", ""),
                row.get("effective_attack_count", 0),
                row.get("malicious_count", 0),
                rate(float(row.get("malicious_defended_violation_rate", 0.0))),
                rate(float(row.get("malicious_violation_rate_reduction", 0.0))),
                rate(float(row.get("malicious_defended_post_block_rate", 0.0))),
                rate(float(row.get("benign_false_refusal_rate", 0.0))),
            ]
            for row in suite_runs
        ],
    )}

    <h2>输出文件</h2>
    <ul>
      <li><code>{escape(str(suite_dir / 'suite_runs.jsonl'))}</code></li>
      <li><code>{escape(str(suite_dir / 'suite_summary.json'))}</code></li>
      <li><code>{escape(str(suite_dir / 'strategy_guardweave_suite_report.zh-CN.html'))}</code></li>
      <li><code>{escape(str(suite_dir / 'strategy_guardweave_suite_report.zh-CN.pdf'))}</code></li>
    </ul>

    <p class="footnote">说明：本报告聚合 10 次重复实验；单次运行的完整明细、误拒样本和对比记录保存在各自 run 目录下。</p>
  </main>
</body>
</html>"""
    return html


def find_chrome_binary() -> str:
    for env_name in ("CHROME_BIN", "CHROMIUM_BIN", "BROWSER_BIN"):
        candidate = os.environ.get(env_name, "").strip()
        if candidate:
            return candidate

    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "google-chrome-stable"):
        candidate = shutil.which(name)
        if candidate:
            return candidate

    for candidate in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ):
        if Path(candidate).exists():
            return candidate

    raise SystemExit(
        "A Chrome-compatible browser is required to render the PDF report. "
        "Set CHROME_BIN or install google-chrome/chromium."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--title", default="GuardWeave 10 次重复实验汇总报告（中文）")
    parser.add_argument("--html-out")
    parser.add_argument("--pdf-out")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    html_out = Path(args.html_out).resolve() if args.html_out else suite_dir / "strategy_guardweave_suite_report.zh-CN.html"
    pdf_out = Path(args.pdf_out).resolve() if args.pdf_out else suite_dir / "strategy_guardweave_suite_report.zh-CN.pdf"

    html = render_report(suite_dir, args.title)
    html_out.write_text(html, encoding="utf-8")

    cmd = [
        find_chrome_binary(),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_out}",
        html_out.as_uri(),
    ]
    subprocess.run(cmd, check=True)
    print(json.dumps({"html": str(html_out), "pdf": str(pdf_out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
