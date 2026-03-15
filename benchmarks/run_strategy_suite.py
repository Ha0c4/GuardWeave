#!/usr/bin/env python3
"""Run a cached 10-run GuardWeave suite with one shared precompute pass."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import strategy_driven_guardweave_experiment as exp


SUITE_REPORT_SCRIPT = Path(__file__).resolve().parent / "render_strategy_suite_report.py"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_eval_records(
    test_records: Sequence[Dict[str, Any]],
    *,
    candidate_attack_pool_size: int,
    benign_eval_size: int,
    prefer_short_candidates: bool,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    malicious_test = [record for record in test_records if record["is_malicious"]]
    benign_test = [record for record in test_records if not record["is_malicious"]]
    candidate_source = list(malicious_test)
    benign_source = list(benign_test)

    if prefer_short_candidates:
        short_malicious = [record for record in malicious_test if record["length_type"] == "short"]
        short_benign = [record for record in benign_test if record["length_type"] == "short"]
        if len(short_malicious) >= candidate_attack_pool_size:
            candidate_source = short_malicious
        else:
            candidate_source = sorted(
                malicious_test,
                key=lambda record: (record["length_type"] != "short", len(record["prompt"])),
            )
        if len(short_benign) >= benign_eval_size:
            benign_source = short_benign
        else:
            benign_source = sorted(
                benign_test,
                key=lambda record: (record["length_type"] != "short", len(record["prompt"])),
            )

    candidate_attacks = exp.stratified_sample(
        candidate_source,
        candidate_attack_pool_size,
        seed=seed + 4,
        key_fields=("category", "length_type"),
    )
    benign_eval = exp.stratified_sample(
        benign_source,
        benign_eval_size,
        seed=seed + 5,
        key_fields=("length_type",),
    )
    if prefer_short_candidates:
        candidate_attacks = sorted(candidate_attacks, key=lambda record: (record["length_type"] != "short", len(record["prompt"])))
        benign_eval = sorted(benign_eval, key=lambda record: (record["length_type"] != "short", len(record["prompt"])))
    return candidate_attacks, benign_eval


def collect_run_metrics(run_dir: Path, *, seed: int, run_name: str) -> Dict[str, Any]:
    summary_rows = read_jsonl(run_dir / "attack_comparison_summary.jsonl")
    summary = {(row["group_type"], row["group_value"]): row for row in summary_rows}
    selection = read_json(run_dir / "attack_selection_summary.json")
    evaluation = read_json(run_dir / "evaluation_summary.json")
    malicious = summary[("label", "malicious")]
    benign = summary[("label", "benign")]
    overall = summary[("overall", "all")]
    manifest = read_json(run_dir / "experiment_manifest.json")
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "seed": seed,
        "base_model": manifest.get("base_model", ""),
        "judge_model": manifest.get("judge_model", ""),
        "candidate_attack_pool_size": selection.get("candidate_attack_pool_size", 0),
        "effective_attack_count": selection.get("effective_attack_count", 0),
        "effective_attack_rate": selection.get("effective_attack_rate", 0.0),
        "malicious_count": evaluation.get("malicious_count", 0),
        "benign_count": evaluation.get("benign_count", 0),
        "malicious_defended_violation_rate": malicious.get("defended_system_prompt_violation_rate", 0.0),
        "malicious_violation_rate_reduction": malicious.get("violation_rate_reduction", 0.0),
        "malicious_defended_post_block_rate": malicious.get("defended_post_block_rate", 0.0),
        "benign_false_refusal_rate": benign.get("defended_refusal_rate", 0.0),
        "overall_defended_violation_rate": overall.get("defended_system_prompt_violation_rate", 0.0),
    }


def build_suite_summary(run_rows: List[Dict[str, Any]], *, base_model: str, judge_model: str, candidate_attack_pool_size: int, min_effective_attacks: int) -> Dict[str, Any]:
    def values(key: str) -> List[float]:
        return [float(row.get(key, 0.0)) for row in run_rows]

    def mean(key: str) -> float:
        vals = values(key)
        return statistics.mean(vals) if vals else 0.0

    def stdev(key: str) -> float:
        vals = values(key)
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    def min_value(key: str) -> float:
        vals = values(key)
        return min(vals) if vals else 0.0

    def max_value(key: str) -> float:
        vals = values(key)
        return max(vals) if vals else 0.0

    return {
        "run_count": len(run_rows),
        "base_model": base_model,
        "judge_model": judge_model,
        "candidate_attack_pool_size": candidate_attack_pool_size,
        "min_effective_attacks": min_effective_attacks,
        "effective_attack_count_mean": mean("effective_attack_count"),
        "effective_attack_count_std": stdev("effective_attack_count"),
        "effective_attack_count_min": min_value("effective_attack_count"),
        "effective_attack_count_max": max_value("effective_attack_count"),
        "effective_attack_rate_mean": mean("effective_attack_rate"),
        "effective_attack_rate_std": stdev("effective_attack_rate"),
        "effective_attack_rate_min": min_value("effective_attack_rate"),
        "effective_attack_rate_max": max_value("effective_attack_rate"),
        "malicious_defended_violation_rate_mean": mean("malicious_defended_violation_rate"),
        "malicious_defended_violation_rate_std": stdev("malicious_defended_violation_rate"),
        "malicious_defended_violation_rate_min": min_value("malicious_defended_violation_rate"),
        "malicious_defended_violation_rate_max": max_value("malicious_defended_violation_rate"),
        "malicious_violation_rate_reduction_mean": mean("malicious_violation_rate_reduction"),
        "malicious_violation_rate_reduction_std": stdev("malicious_violation_rate_reduction"),
        "malicious_violation_rate_reduction_min": min_value("malicious_violation_rate_reduction"),
        "malicious_violation_rate_reduction_max": max_value("malicious_violation_rate_reduction"),
        "malicious_defended_post_block_rate_mean": mean("malicious_defended_post_block_rate"),
        "malicious_defended_post_block_rate_std": stdev("malicious_defended_post_block_rate"),
        "malicious_defended_post_block_rate_min": min_value("malicious_defended_post_block_rate"),
        "malicious_defended_post_block_rate_max": max_value("malicious_defended_post_block_rate"),
        "benign_false_refusal_rate_mean": mean("benign_false_refusal_rate"),
        "benign_false_refusal_rate_std": stdev("benign_false_refusal_rate"),
        "benign_false_refusal_rate_min": min_value("benign_false_refusal_rate"),
        "benign_false_refusal_rate_max": max_value("benign_false_refusal_rate"),
    }


def materialize_run(
    *,
    run_dir: Path,
    run_name: str,
    seed: int,
    selected_attacks: List[Dict[str, Any]],
    benign_records: List[Dict[str, Any]],
    candidate_attacks: List[Dict[str, Any]],
    manifest_template: Dict[str, Any],
    selection_summary: Dict[str, Any],
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = exp.summarize_records(list(selected_attacks) + list(benign_records))
    evaluation_summary = exp.build_evaluation_summary(list(selected_attacks) + list(benign_records))
    false_refusals = [row for row in benign_records if row.get("defended_refusal")]

    write_jsonl(run_dir / "attack_selection_records.jsonl", candidate_attacks)
    write_json(run_dir / "attack_selection_summary.json", selection_summary)
    write_jsonl(run_dir / "selected_attack_eval_records.jsonl", selected_attacks)
    write_jsonl(run_dir / "attack_comparison_records.jsonl", list(selected_attacks) + list(benign_records))
    write_jsonl(run_dir / "attack_comparison_summary.jsonl", summary_rows)
    write_jsonl(run_dir / "benign_false_refusal_records.jsonl", false_refusals)
    write_json(run_dir / "evaluation_summary.json", evaluation_summary)

    manifest = dict(manifest_template)
    manifest["seed"] = seed
    manifest["evaluation"] = {
        "records_path": str(run_dir / "attack_comparison_records.jsonl"),
        "summary_path": str(run_dir / "attack_comparison_summary.jsonl"),
        "false_refusal_path": str(run_dir / "benign_false_refusal_records.jsonl"),
        "evaluation_summary_path": str(run_dir / "evaluation_summary.json"),
        "selected_attack_count": len(selected_attacks),
        "benign_eval_count": len(benign_records),
        "device": manifest_template.get("runtime_device", ""),
        "judge_dtype": manifest_template.get("runtime_dtype", ""),
    }
    write_json(run_dir / "experiment_manifest.json", manifest)
    return collect_run_metrics(run_dir, seed=seed, run_name=run_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--existing-run-dir")
    parser.add_argument("--existing-seed", type=int, default=42)
    parser.add_argument("--start-seed", type=int, default=43)
    parser.add_argument("--total-runs", type=int, default=10)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--judge-backend", choices=("local", "gemini"), default="local")
    parser.add_argument("--risk-judge-dir")
    parser.add_argument("--output-judge-dir")
    parser.add_argument("--gemini-judge-model", default="gemini-2.5-flash")
    parser.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--gemini-api-base", default="https://generativelanguage.googleapis.com/v1beta")
    parser.add_argument("--gemini-timeout-s", type=float, default=20.0)
    parser.add_argument("--gemini-max-output-tokens", type=int, default=256)
    parser.add_argument("--candidate-attack-pool-size", type=int, default=200)
    parser.add_argument("--attack-eval-size", type=int, default=80)
    parser.add_argument("--min-effective-attacks", type=int, default=80)
    parser.add_argument("--benign-eval-size", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefer-short-candidates", action="store_true")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = suite_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    judge_config = exp.build_judge_config(
        judge_backend=args.judge_backend,
        risk_judge_dir=args.risk_judge_dir,
        output_judge_dir=args.output_judge_dir,
        judge_model=args.judge_model,
        gemini_model=args.gemini_judge_model,
        gemini_api_key_env=args.gemini_api_key_env,
        gemini_api_base=args.gemini_api_base,
        gemini_timeout_s=args.gemini_timeout_s,
        gemini_max_output_tokens=args.gemini_max_output_tokens,
    )
    judge_model_label = exp.resolve_judge_model_label(judge_config)

    run_rows: List[Dict[str, Any]] = []
    pending_run_specs: List[Tuple[int, int]] = []
    if args.existing_run_dir:
        existing_run_dir = Path(args.existing_run_dir).resolve()
        run1_dir = runs_dir / f"run_01_seed{args.existing_seed}"
        if not run1_dir.exists():
            shutil.copytree(existing_run_dir, run1_dir)
        existing_manifest = read_json(run1_dir / "experiment_manifest.json")
        existing_manifest["seed"] = args.existing_seed
        write_json(run1_dir / "experiment_manifest.json", existing_manifest)
        run_rows.append(collect_run_metrics(run1_dir, seed=args.existing_seed, run_name=run1_dir.name))
        pending_run_specs.extend(
            (run_index, args.start_seed + (run_index - 2))
            for run_index in range(2, args.total_runs + 1)
        )
    else:
        pending_run_specs.append((1, args.existing_seed))
        pending_run_specs.extend(
            (run_index, args.start_seed + (run_index - 2))
            for run_index in range(2, args.total_runs + 1)
        )

    if args.total_runs == 1 and run_rows:
        write_jsonl(suite_dir / "suite_runs.jsonl", run_rows)
        write_json(
            suite_dir / "suite_summary.json",
            build_suite_summary(
                run_rows,
                base_model=args.base_model,
                judge_model=judge_model_label,
                candidate_attack_pool_size=args.candidate_attack_pool_size,
                min_effective_attacks=args.min_effective_attacks,
            ),
        )
        subprocess.run([sys.executable, str(SUITE_REPORT_SCRIPT), "--suite-dir", str(suite_dir)], check=True)
        print(json.dumps({"suite_dir": str(suite_dir), "run_count": 1}, ensure_ascii=False))
        return 0

    dataset = exp.load_dataset(exp.DATASET_NAME)
    test_records = [exp.dataset_row_to_record(row, split="test") for row in dataset["test"]]

    run_specs: List[Dict[str, Any]] = []
    all_needed_records: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for run_index, seed in pending_run_specs:
        run_name = f"run_{run_index:02d}_seed{seed}"
        candidate_attacks, benign_eval = sample_eval_records(
            test_records,
            candidate_attack_pool_size=args.candidate_attack_pool_size,
            benign_eval_size=args.benign_eval_size,
            prefer_short_candidates=args.prefer_short_candidates,
            seed=seed,
        )
        run_specs.append(
            {
                "seed": seed,
                "run_name": run_name,
                "candidate_attacks": candidate_attacks,
                "benign_eval": benign_eval,
            }
        )
        for record in list(candidate_attacks) + list(benign_eval):
            all_needed_records[exp.build_record_key(record)] = record

    print(f"[suite-cached] unique prompts to precompute: {len(all_needed_records)}", flush=True)
    strategies = exp.load_local_strategies(exp.LOCAL_STRATEGIES_FILE)
    policy = exp.build_policy()
    system_prompt = exp.build_system_prompt(exp.SYSTEM_PROMPT_MARKERS)
    hidden_markers = exp.build_hidden_marker_list(exp.SYSTEM_PROMPT_MARKERS)
    precomputed_rows = exp.precompute_record_outcomes(
        base_model_name=args.base_model,
        judge_config=judge_config,
        records=list(all_needed_records.values()),
        system_prompt=system_prompt,
        policy=policy,
        hidden_markers=hidden_markers,
        max_new_tokens=args.max_new_tokens,
    )
    precomputed_map = {exp.build_record_key(row): row for row in precomputed_rows}
    cache_dir = suite_dir / "precompute"
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache_dir / "precomputed_records.jsonl", precomputed_rows)

    manifest_template = {
        "dataset": exp.DATASET_NAME,
        "base_model": args.base_model,
        "judge_backend": judge_config["backend"],
        "judge_model": judge_model_label,
        "local_strategy_file": str(exp.LOCAL_STRATEGIES_FILE),
        "system_prompt_markers": exp.SYSTEM_PROMPT_MARKERS,
        "prefer_short_candidates": bool(args.prefer_short_candidates),
        "scan_all_candidates": True,
        "min_effective_attacks": int(args.min_effective_attacks),
        "max_new_tokens": int(args.max_new_tokens),
        "runtime_device": exp.detect_device(),
        "runtime_dtype": exp.resolve_dtype_name(exp.detect_device()),
        "training": exp.build_training_payload(judge_config),
        "precompute_cache": str(cache_dir / "precomputed_records.jsonl"),
        "strategy_count": len(strategies),
    }

    for spec in run_specs:
        run_dir = runs_dir / spec["run_name"]
        candidate_rows = [dict(precomputed_map[exp.build_record_key(record)]) for record in spec["candidate_attacks"]]
        benign_rows = [dict(precomputed_map[exp.build_record_key(record)]) for record in spec["benign_eval"]]
        selected_attacks = [row for row in candidate_rows if row.get("undefended_violation")]
        if len(selected_attacks) < args.min_effective_attacks:
            raise RuntimeError(
                f"{spec['run_name']} has only {len(selected_attacks)} effective attacks; expected at least {args.min_effective_attacks}."
            )
        selection_summary = exp.build_selection_summary(
            scanned_rows=candidate_rows,
            selected_rows=selected_attacks,
            candidate_attack_pool_size=args.candidate_attack_pool_size,
            attack_eval_size=args.attack_eval_size,
            min_effective_attacks=args.min_effective_attacks,
            scan_all_candidates=True,
            base_model_name=args.base_model,
            device=exp.detect_device(),
            max_new_tokens=args.max_new_tokens,
        )
        run_rows.append(
            materialize_run(
                run_dir=run_dir,
                run_name=spec["run_name"],
                seed=spec["seed"],
                selected_attacks=selected_attacks,
                benign_records=benign_rows,
                candidate_attacks=candidate_rows,
                manifest_template=manifest_template,
                selection_summary=selection_summary,
            )
        )
        write_jsonl(suite_dir / "suite_runs.jsonl", run_rows)
        write_json(
            suite_dir / "suite_summary.json",
            build_suite_summary(
                run_rows,
                base_model=args.base_model,
                judge_model=judge_model_label,
                candidate_attack_pool_size=args.candidate_attack_pool_size,
                min_effective_attacks=args.min_effective_attacks,
            ),
        )
        print(
            f"[suite-cached] materialized {spec['run_name']} with {len(selected_attacks)} effective attacks and {len(benign_rows)} benign prompts",
            flush=True,
        )

    write_jsonl(suite_dir / "suite_runs.jsonl", run_rows)
    write_json(
        suite_dir / "suite_summary.json",
        build_suite_summary(
            run_rows,
            base_model=args.base_model,
            judge_model=judge_model_label,
            candidate_attack_pool_size=args.candidate_attack_pool_size,
            min_effective_attacks=args.min_effective_attacks,
        ),
    )
    subprocess.run([sys.executable, str(SUITE_REPORT_SCRIPT), "--suite-dir", str(suite_dir)], check=True)
    print(json.dumps({"suite_dir": str(suite_dir), "run_count": len(run_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
