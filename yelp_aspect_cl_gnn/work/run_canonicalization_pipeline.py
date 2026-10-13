#!/usr/bin/env python3
"""Run LLM labeling, multi-seed training, ablations, and aggregation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "work" / "train_aspect_contrastive_gnn.py"
LABELER = ROOT / "work" / "label_aspect_pairs.py"
METRICS = ("Recall@10", "NDCG@10", "Recall@20", "NDCG@20")


EXPERIMENTS: dict[str, list[str]] = {
    "baseline_full": ["--variants", "full"],
    "joint_llm_behavior": [
        "--variants", "joint",
        "--canonical-feature-mode", "all",
    ],
    "joint_semantic_only": [
        "--variants", "joint",
        "--canonical-feature-mode", "semantic",
    ],
    "joint_behavior_only": [
        "--variants", "joint",
        "--canonical-feature-mode", "behavior",
        "--canonical-prior-weight", "0",
    ],
    "joint_fixed_prior": [
        "--variants", "joint",
        "--canonical-feature-mode", "all",
        "--freeze-canonical-gates",
        "--canonical-prior-weight", "0",
        "--canonical-cluster-weight", "0",
        "--canonical-sparse-weight", "0",
    ],
    "joint_rec_only": [
        "--variants", "joint",
        "--canonical-feature-mode", "all",
        "--merge-strength", "1.0",
        "--canonical-prior-weight", "0",
        "--canonical-cluster-weight", "0",
        "--canonical-sparse-weight", "0",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "canonicalization_pipeline",
    )
    parser.add_argument(
        "--pair-file",
        type=Path,
        default=ROOT / "outputs" / "llm_aspect_pairs_v2.csv",
    )
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--seeds", default="42,2026,7")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--skip-labeling", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def stream_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if (
                line.startswith("{")
                or "selected_epoch=" in line
                or "early_stop" in line
                or "labeled=" in line
            ):
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_labeling(args: argparse.Namespace) -> None:
    if args.skip_labeling:
        if not args.pair_file.exists():
            raise FileNotFoundError(args.pair_file)
        return
    command = [
        sys.executable,
        str(LABELER),
        "--output", str(args.pair_file),
        "--model", args.model,
    ]
    stream_command(command, args.output_dir / "llm_labeling.log")


def training_command(
    args: argparse.Namespace,
    experiment: str,
    seed: int,
    run_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAINER),
        *EXPERIMENTS[experiment],
        "--output-dir", str(run_dir),
        "--epochs", str(args.epochs),
        "--min-epochs", str(args.min_epochs),
        "--patience", str(args.patience),
        "--embedding-dim", "64",
        "--layers", "2",
        "--learning-rate", "0.003",
        "--contrastive-weight", "0.02",
        "--temperature", "0.2",
        "--regularization", "0.0001",
        "--seed", str(seed),
        "--threads", str(args.threads),
    ]
    if experiment != "baseline_full":
        command.extend(["--aspect-pairs", str(args.pair_file)])
    return command


def run_experiments(args: argparse.Namespace) -> None:
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    selected = [value for value in args.experiments.split(",") if value]
    unknown = set(selected) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    for experiment in selected:
        for seed in seeds:
            run_dir = args.output_dir / experiment / f"seed_{seed}"
            result_path = run_dir / "results.json"
            if args.resume and result_path.exists():
                print(f"skip completed {experiment} seed={seed}", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"start {experiment} seed={seed}", flush=True)
            stream_command(
                training_command(args, experiment, seed, run_dir),
                run_dir / "train.log",
            )


def aggregate(args: argparse.Namespace) -> None:
    records = []
    for result_path in sorted(args.output_dir.glob("*/seed_*/results.json")):
        experiment = result_path.parents[1].name
        seed = int(result_path.parent.name.removeprefix("seed_"))
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        row: dict[str, object] = {
            "experiment": experiment,
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
        }
        for split in ("validation", "test"):
            for metric in METRICS:
                row[f"{split}_{metric.lower()}"] = result[split][metric]
        records.append(row)
    if not records:
        raise RuntimeError("No experiment results found")

    raw_path = args.output_dir / "all_runs.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary_rows = []
    experiments = sorted({str(row["experiment"]) for row in records})
    for experiment in experiments:
        subset = [row for row in records if row["experiment"] == experiment]
        summary: dict[str, object] = {
            "experiment": experiment,
            "runs": len(subset),
        }
        for split in ("validation", "test"):
            for metric in METRICS:
                key = f"{split}_{metric.lower()}"
                values = [float(row[key]) for row in subset]
                summary[f"{key}_mean"] = statistics.mean(values)
                summary[f"{key}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
        summary_rows.append(summary)

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    baseline = next(
        (row for row in summary_rows if row["experiment"] == "baseline_full"),
        None,
    )
    report = [
        "# Aspect canonicalization pipeline report",
        "",
        f"Seeds: {args.seeds}. Values are mean ± sample standard deviation.",
        "",
        "| Experiment | Test Recall@10 | Test NDCG@10 | Test Recall@20 | Test NDCG@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(
        summary_rows,
        key=lambda value: float(value["test_ndcg@10_mean"]),
        reverse=True,
    ):
        cells = []
        for metric in METRICS:
            key = f"test_{metric.lower()}"
            cells.append(
                f"{float(row[key + '_mean']):.6f} ± "
                f"{float(row[key + '_std']):.6f}"
            )
        report.append(f"| {row['experiment']} | " + " | ".join(cells) + " |")
    if baseline is not None:
        report.extend(["", "## Mean change from baseline", ""])
        for row in summary_rows:
            if row["experiment"] == "baseline_full":
                continue
            changes = []
            for metric in METRICS:
                key = f"test_{metric.lower()}_mean"
                base = float(baseline[key])
                value = float(row[key])
                changes.append(f"{metric}: {(value / base - 1) * 100:+.2f}%")
            report.append(f"- **{row['experiment']}** — " + "; ".join(changes))

    if args.pair_file.exists():
        pair_rows = list(csv.DictReader(args.pair_file.open(encoding="utf-8")))
        relation_counts: dict[str, int] = {}
        for row in pair_rows:
            relation = row["llm_relation"]
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        report.extend(
            [
                "",
                "## LLM candidate audit",
                "",
                f"- Candidate pairs: {len(pair_rows)}",
                "- Relation counts: "
                + ", ".join(
                    f"{key}={value}" for key, value in sorted(relation_counts.items())
                ),
                "- The LLM field is merge probability, validated against the "
                "relation label before training.",
            ]
        )

    gate_paths = sorted(
        (args.output_dir / "joint_llm_behavior").glob(
            "seed_*/learned_aspect_gates.csv"
        )
    )
    if gate_paths:
        by_pair: dict[tuple[str, str], dict[str, object]] = {}
        for path in gate_paths:
            for row in csv.DictReader(path.open(encoding="utf-8")):
                pair = (row["aspect_i"], row["aspect_j"])
                entry = by_pair.setdefault(
                    pair,
                    {
                        "prior": float(row["prior_confidence"]),
                        "gates": [],
                    },
                )
                gates = entry["gates"]
                assert isinstance(gates, list)
                gates.append(float(row["learned_gate"]))
        shifts = []
        for pair, entry in by_pair.items():
            gates = entry["gates"]
            assert isinstance(gates, list)
            prior = float(entry["prior"])
            shifts.append((statistics.mean(gates) - prior, pair, prior,
                           statistics.mean(gates)))
        shifts.sort()
        report.extend(
            [
                "",
                "## Recommendation-guided gate changes",
                "",
                "Largest mean decreases across seeds:",
                "",
            ]
        )
        for shift, pair, prior, gate in shifts[:8]:
            report.append(
                f"- `{pair[0]} / {pair[1]}`: {prior:.3f} → "
                f"{gate:.3f} ({shift:+.3f})"
            )
        report.extend(["", "Largest mean increases across seeds:", ""])
        for shift, pair, prior, gate in reversed(shifts[-8:]):
            report.append(
                f"- `{pair[0]} / {pair[1]}`: {prior:.3f} → "
                f"{gate:.3f} ({shift:+.3f})"
            )
        report.extend(
            [
                "",
                "With only three seeds, differences are descriptive and are not "
                "claimed as statistically significant.",
            ]
        )
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "REPORT.md", flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_labeling(args)
    run_experiments(args)
    aggregate(args)


if __name__ == "__main__":
    main()
