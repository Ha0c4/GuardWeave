#!/usr/bin/env python3
"""Render a Chinese HTML/PDF report for a strategy-driven GuardWeave experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return escape(str(value))


def render_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head_html = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{format_cell(cell)}</td>" for cell in row) + "</tr>")
    return (
        "<table>"
        "<thead><tr>" + head_html + "</tr></thead>"
        "<tbody>" + "".join(body_rows) + "</tbody>"
        "</table>"
    )


def render_report(results_dir: Path, title: str) -> str:
    manifest = read_json(results_dir / "experiment_manifest.json")
    selection_summary = read_json(results_dir / "attack_selection_summary.json")
    evaluation_summary = read_json(results_dir / "evaluation_summary.json")
    summary_rows = read_jsonl(results_dir / "attack_comparison_summary.jsonl")
    false_refusals = read_jsonl(results_dir / "benign_false_refusal_records.jsonl")
    selected_attacks = read_jsonl(results_dir / "selected_attack_eval_records.jsonl")

    summary_map = {(row["group_type"], row["group_value"]): row for row in summary_rows}
    overall = summary_map.get(("overall", "all"), {})
    malicious = summary_map.get(("label", "malicious"), {})
    benign = summary_map.get(("label", "benign"), {})

    category_rows = [row for row in summary_rows if row["group_type"] == "category" and row["group_value"] != "benign"]
    strongest_defense = sorted(
        category_rows,
        key=lambda row: (row.get("violation_rate_reduction", 0.0), row.get("defended_refusal_rate", 0.0)),
        reverse=True,
    )[:10]
    weakest_defense = sorted(
        category_rows,
        key=lambda row: (row.get("defended_system_prompt_violation_rate", 0.0), -row.get("violation_rate_reduction", 0.0)),
        reverse=True,
    )[:10]

    false_refusal_items = []
    for row in false_refusals[:10]:
        false_refusal_items.append(
            "<li><strong>{}</strong>: {}</li>".format(
                escape(str(row.get("category", "benign"))),
                escape(str(row.get("prompt", ""))[:180]),
            )
        )

    selected_attack_items = []
    for row in selected_attacks[:12]:
        selected_attack_items.append(
            "<li><strong>{}</strong>: {}</li>".format(
                escape(str(row.get("category", ""))),
                escape(str(row.get("prompt", ""))[:180]),
            )
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm 18mm 16mm;
    }}
    body {{
      font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      color: #18212b;
      line-height: 1.55;
      font-size: 13px;
      margin: 0;
      background: #f5f7fb;
    }}
    main {{
      max-width: 920px;
      margin: 0 auto;
      background: white;
      padding: 24px 28px 40px;
    }}
    h1, h2, h3 {{
      color: #0f2742;
      margin-top: 0;
    }}
    h1 {{
      font-size: 28px;
      margin-bottom: 12px;
    }}
    h2 {{
      font-size: 20px;
      margin-top: 28px;
      border-bottom: 2px solid #d7e1ef;
      padding-bottom: 6px;
    }}
    .meta, .highlight-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 18px;
    }}
    .card {{
      background: #f3f7fd;
      border: 1px solid #dbe6f3;
      border-radius: 12px;
      padding: 12px 14px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .kpi {{
      background: linear-gradient(180deg, #edf5ff, #f9fbff);
      border: 1px solid #d6e3f5;
      border-radius: 14px;
      padding: 14px;
    }}
    .kpi .label {{
      font-size: 12px;
      color: #55697f;
    }}
    .kpi .value {{
      font-size: 24px;
      font-weight: 700;
      color: #0e355d;
      margin-top: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 18px;
      table-layout: fixed;
      word-break: break-word;
    }}
    th, td {{
      border: 1px solid #d8e2ee;
      padding: 8px 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #eff4fa;
    }}
    ul {{
      padding-left: 20px;
    }}
    code {{
      font-family: "SF Mono", "Menlo", monospace;
      font-size: 11px;
      background: #f1f4f8;
      padding: 1px 4px;
      border-radius: 4px;
    }}
    .footnote {{
      color: #55697f;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <div class="meta">
      <div class="card"><strong>Base Model</strong><br>{escape(str(manifest.get("base_model", "")))}</div>
      <div class="card"><strong>Judge Model</strong><br>{escape(str(manifest.get("judge_model", "")))}</div>
      <div class="card"><strong>运行设备</strong><br>{escape(str(manifest.get("runtime_device", "")))} / {escape(str(manifest.get("runtime_dtype", "")))}</div>
      <div class="card"><strong>生成长度</strong><br>max_new_tokens = {escape(str(manifest.get("max_new_tokens", "")))}</div>
    </div>

    <h2>结论摘要</h2>
    <div class="kpi-grid">
      <div class="kpi"><div class="label">扫描攻击数</div><div class="value">{selection_summary.get("scanned_attack_count", 0)}</div></div>
      <div class="kpi"><div class="label">无防御可打穿攻击数</div><div class="value">{selection_summary.get("effective_attack_count", 0)}</div></div>
      <div class="kpi"><div class="label">恶意样本防御后违背率</div><div class="value">{rate(float(malicious.get("defended_system_prompt_violation_rate", 0.0)))}</div></div>
      <div class="kpi"><div class="label">良性误拒率</div><div class="value">{rate(float(benign.get("defended_refusal_rate", 0.0)))}</div></div>
    </div>
    <p>本次实验完整扫描了 <strong>{selection_summary.get("scanned_attack_count", 0)}</strong> 条外部 prompt injection 攻击，并筛出 <strong>{selection_summary.get("effective_attack_count", 0)}</strong> 条在无防御 <code>{escape(str(manifest.get("base_model", "")))}</code> 上确实会触发 system prompt 违背的攻击。随后使用 GuardWeave 的 3B 本地 risk/output judge 进行前后对比。</p>

    <h2>实验配置</h2>
    {render_table(
        ["字段", "数值"],
        [
            ["数据集", manifest.get("dataset", "")],
            ["本地策略文件", manifest.get("local_strategy_file", "")],
            ["候选攻击池", selection_summary.get("candidate_attack_pool_size", 0)],
            ["最终有效攻击数", selection_summary.get("effective_attack_count", 0)],
            ["最少要求有效攻击数", selection_summary.get("min_effective_attacks", 0)],
            ["Benign 样本数", evaluation_summary.get("benign_count", 0)],
            ["完整扫描", "是" if manifest.get("scan_all_candidates") else "否"],
            ["选择完成时间(UTC)", selection_summary.get("selection_completed_at", "")],
            ["评测完成时间(UTC)", evaluation_summary.get("evaluation_completed_at", "")],
        ],
    )}

    <h2>核心指标</h2>
    {render_table(
        ["分组", "样本数", "无防御违背率", "防御后违背率", "违背率下降", "防御拒答率", "前置拒答率", "后置拦截率"],
        [
            [
                "整体",
                overall.get("count", 0),
                rate(float(overall.get("undefended_system_prompt_violation_rate", 0.0))),
                rate(float(overall.get("defended_system_prompt_violation_rate", 0.0))),
                rate(float(overall.get("violation_rate_reduction", 0.0))),
                rate(float(overall.get("defended_refusal_rate", 0.0))),
                rate(float(overall.get("defended_pre_refusal_rate", 0.0))),
                rate(float(overall.get("defended_post_block_rate", 0.0))),
            ],
            [
                "恶意攻击",
                malicious.get("count", 0),
                rate(float(malicious.get("undefended_system_prompt_violation_rate", 0.0))),
                rate(float(malicious.get("defended_system_prompt_violation_rate", 0.0))),
                rate(float(malicious.get("violation_rate_reduction", 0.0))),
                rate(float(malicious.get("defended_refusal_rate", 0.0))),
                rate(float(malicious.get("defended_pre_refusal_rate", 0.0))),
                rate(float(malicious.get("defended_post_block_rate", 0.0))),
            ],
            [
                "良性请求",
                benign.get("count", 0),
                rate(float(benign.get("undefended_system_prompt_violation_rate", 0.0))),
                rate(float(benign.get("defended_system_prompt_violation_rate", 0.0))),
                rate(float(benign.get("violation_rate_reduction", 0.0))),
                rate(float(benign.get("defended_refusal_rate", 0.0))),
                rate(float(benign.get("defended_pre_refusal_rate", 0.0))),
                rate(float(benign.get("defended_post_block_rate", 0.0))),
            ],
        ],
    )}

    <h2>攻击筛选结果</h2>
    <p>攻击筛选阶段使用无防御 7B base model 逐条生成输出，并以 “是否违背 system prompt / 是否出现非拒答攻击响应” 作为有效攻击条件。前 12 条有效攻击样例如下：</p>
    <ul>{"".join(selected_attack_items)}</ul>

    <h2>误拒统计</h2>
    <p>良性样本共 <strong>{evaluation_summary.get("benign_count", 0)}</strong> 条，其中发生误拒 <strong>{evaluation_summary.get("false_refusal_count", 0)}</strong> 条，误拒率为 <strong>{rate(float(evaluation_summary.get("false_refusal_rate", 0.0)))}</strong>。</p>
    <p>示例误拒请求：</p>
    <ul>{"".join(false_refusal_items) if false_refusal_items else "<li>本次没有误拒样本。</li>"}</ul>

    <h2>最强与最弱类别</h2>
    <h3>防御效果最强的 10 个类别</h3>
    {render_table(
        ["类别", "样本数", "违背率下降", "防御后违背率", "防御拒答率"],
        [
            [
                row.get("group_value", ""),
                row.get("count", 0),
                rate(float(row.get("violation_rate_reduction", 0.0))),
                rate(float(row.get("defended_system_prompt_violation_rate", 0.0))),
                rate(float(row.get("defended_refusal_rate", 0.0))),
            ]
            for row in strongest_defense
        ],
    )}
    <h3>防御最难处理的 10 个类别</h3>
    {render_table(
        ["类别", "样本数", "防御后违背率", "违背率下降", "后置拦截率"],
        [
            [
                row.get("group_value", ""),
                row.get("count", 0),
                rate(float(row.get("defended_system_prompt_violation_rate", 0.0))),
                rate(float(row.get("violation_rate_reduction", 0.0))),
                rate(float(row.get("defended_post_block_rate", 0.0))),
            ]
            for row in weakest_defense
        ],
    )}

    <h2>结果文件</h2>
    <ul>
      <li><code>{escape(str(results_dir / "attack_selection_records.jsonl"))}</code></li>
      <li><code>{escape(str(results_dir / "selected_attack_eval_records.jsonl"))}</code></li>
      <li><code>{escape(str(results_dir / "attack_comparison_records.jsonl"))}</code></li>
      <li><code>{escape(str(results_dir / "attack_comparison_summary.jsonl"))}</code></li>
      <li><code>{escape(str(results_dir / "benign_false_refusal_records.jsonl"))}</code></li>
      <li><code>{escape(str(results_dir / "evaluation_summary.json"))}</code></li>
    </ul>

    <p class="footnote">说明：本报告统计口径中，“打穿”指无防御 base model 的输出被评估器判定为违背 system prompt；“误拒”指良性样本在防御侧被拒答。</p>
  </main>
</body>
</html>"""
    return html


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--title", default="GuardWeave 攻击防御实验报告")
    parser.add_argument("--html-out")
    parser.add_argument("--pdf-out")
    return_code = 0
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    html_out = Path(args.html_out).resolve() if args.html_out else results_dir / "strategy_guardweave_report.zh-CN.html"
    pdf_out = Path(args.pdf_out).resolve() if args.pdf_out else results_dir / "strategy_guardweave_report.zh-CN.pdf"

    html = render_report(results_dir, args.title)
    html_out.write_text(html, encoding="utf-8")

    chrome_bin = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome_bin.exists():
        raise SystemExit("Google Chrome is required to render the PDF report.")

    cmd = [
        str(chrome_bin),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_out}",
        html_out.as_uri(),
    ]
    subprocess.run(cmd, check=True)
    print(json.dumps({"html": str(html_out), "pdf": str(pdf_out)}, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
