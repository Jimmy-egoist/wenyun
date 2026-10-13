#!/usr/bin/env python3
"""Generate cached LLM synonym priors for AspectCL-GNN candidate pairs."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import train_aspect_contrastive_gnn as trainer


ROOT = Path(__file__).resolve().parents[1]


class PairJudgment(BaseModel):
    pair_id: int
    relation: Literal[
        "semantic_equivalent",
        "related_but_distinct",
        "unrelated",
    ]
    merge_probability: float = Field(ge=0.0, le=1.0)
    canonical_name: str
    reason: str


class JudgmentBatch(BaseModel):
    judgments: list[PairJudgment]


INSTRUCTIONS = """You judge aspect canonicalization for a Yelp restaurant recommender.

For every input pair, decide whether the two phrases denote the same user-preference
dimension and can safely share one canonical aspect. Semantic relatedness is not
enough. Keep behaviorally distinct concepts separate (for example, food vs menu,
service vs wait time, or beer vs wine).

relation:
- semantic_equivalent: interchangeable or a direct lexical variant in this domain.
- related_but_distinct: related, but users can reasonably prefer one independently.
- unrelated: no useful canonical equivalence.

merge_probability must be the probability that the pair is semantic_equivalent:
- 0.90-1.00: virtually interchangeable lexical variants.
- 0.50-0.89: plausible equivalence with some ambiguity.
- 0.10-0.49: related but usually behaviorally distinct.
- 0.00-0.09: clearly distinct or unrelated.
It is NOT confidence in your classification. semantic_equivalent must have
merge_probability >= 0.50; all other relations must have merge_probability < 0.50.
Use collaborative similarities only as weak context; do not turn co-occurrence into
synonymy. Return exactly one judgment for every pair_id, preserve IDs, and keep
reason under 20 words."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=trainer.DEFAULT_DATA,
    )
    parser.add_argument(
        "--glove",
        type=Path,
        default=trainer.DEFAULT_GLOVE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "llm_aspect_pairs_v2.csv",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs" / "llm_aspect_pair_cache_v2.jsonl",
    )
    parser.add_argument("--provider", choices=("openai", "mock"), default="openai")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--merge-candidates", type=int, default=4)
    parser.add_argument("--merge-threshold", type=float, default=0.45)
    parser.add_argument("--semantic-dim", type=int, default=50)
    return parser.parse_args()


def candidate_records(args: argparse.Namespace) -> list[dict[str, object]]:
    data = trainer.load_interactions(args.data_dir)
    semantic_candidates = trainer.build_canonicalization_data(
        args.data_dir,
        data,
        args.glove,
        args.semantic_dim,
        args.merge_candidates,
        args.merge_threshold,
        None,
        "all",
    )
    all_pairs = trainer.build_canonicalization_data(
        args.data_dir,
        data,
        args.glove,
        args.semantic_dim,
        data.num_aspects - 1,
        -1.0,
        None,
        "all",
    )
    semantic_pair_set = {
        tuple(map(int, pair))
        for pair in semantic_candidates.candidate_pairs
    }
    selected_indices = []
    for index, (left, right) in enumerate(all_pairs.candidate_pairs):
        left_name = all_pairs.aspect_names[int(left)].lower()
        right_name = all_pairs.aspect_names[int(right)].lower()
        common_prefix = os.path.commonprefix((left_name, right_name))
        lexical_variant = (
            len(common_prefix) >= 4
            or difflib.SequenceMatcher(None, left_name, right_name).ratio() >= 0.72
        )
        if tuple(map(int, (left, right))) in semantic_pair_set or lexical_variant:
            selected_indices.append(index)
    records = []
    for pair_id, source_index in enumerate(selected_indices):
        left, right = all_pairs.candidate_pairs[source_index]
        features = all_pairs.pair_features[source_index]
        records.append(
            {
                "pair_id": pair_id,
                "aspect_i": all_pairs.aspect_names[int(left)],
                "aspect_j": all_pairs.aspect_names[int(right)],
                "semantic_similarity": round(float(features[0]), 6),
                "user_behavior_similarity": round(float(features[1]), 6),
                "item_behavior_similarity": round(float(features[2]), 6),
            }
        )
    return records


def load_cache(path: Path) -> dict[int, dict[str, object]]:
    cached: dict[int, dict[str, object]] = {}
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                cached[int(row["pair_id"])] = row
    return cached


def append_cache(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def mock_judgments(batch: list[dict[str, object]]) -> list[PairJudgment]:
    results = []
    for row in batch:
        probability = min(max(float(row["semantic_similarity"]), 0.0), 1.0)
        merge = probability >= 0.50
        results.append(
            PairJudgment(
                pair_id=int(row["pair_id"]),
                relation="semantic_equivalent" if merge else "related_but_distinct",
                merge_probability=probability,
                canonical_name=str(row["aspect_i"]),
                reason="Deterministic smoke-test judgment.",
            )
        )
    return results


def openai_judgments(
    batch: list[dict[str, object]],
    model: str,
    reasoning_effort: str,
) -> tuple[list[PairJudgment], dict[str, int]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        instructions=INSTRUCTIONS,
        input=json.dumps(batch, ensure_ascii=False),
        text_format=JudgmentBatch,
        reasoning={"effort": reasoning_effort},
        store=False,
        safety_identifier="aspectcl-canonicalization-research",
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("The model returned no parsed structured output")
    usage = {
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "total_tokens": int(response.usage.total_tokens),
    }
    return parsed.judgments, usage


def validate_batch(
    inputs: list[dict[str, object]],
    judgments: list[PairJudgment],
) -> None:
    expected = {int(row["pair_id"]) for row in inputs}
    actual = {row.pair_id for row in judgments}
    if expected != actual or len(actual) != len(judgments):
        raise ValueError(
            f"Pair ID mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for judgment in judgments:
        equivalent = judgment.relation == "semantic_equivalent"
        if equivalent != (judgment.merge_probability >= 0.5):
            raise ValueError(
                f"Inconsistent probability/relation for pair {judgment.pair_id}"
            )


def write_output(
    path: Path,
    candidates: list[dict[str, object]],
    cached: dict[int, dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        pair_id = int(candidate["pair_id"])
        judgment = cached[pair_id]
        rows.append(
            {
                **candidate,
                "llm_confidence": judgment["merge_probability"],
                "llm_merge": (
                    judgment["relation"] == "semantic_equivalent"
                ),
                "llm_relation": judgment["relation"],
                "canonical_name": judgment["canonical_name"],
                "llm_reason": judgment["reason"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    candidates = candidate_records(args)
    cached = load_cache(args.cache)
    missing = [row for row in candidates if int(row["pair_id"]) not in cached]
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    print(
        json.dumps(
            {
                "provider": args.provider,
                "model": args.model,
                "candidates": len(candidates),
                "cached": len(candidates) - len(missing),
                "missing": len(missing),
            }
        ),
        flush=True,
    )
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                if args.provider == "mock":
                    judgments = mock_judgments(batch)
                    usage = {key: 0 for key in total_usage}
                else:
                    judgments, usage = openai_judgments(
                        batch,
                        args.model,
                        args.reasoning_effort,
                    )
                validate_batch(batch, judgments)
                cache_rows = [
                    {
                        **judgment.model_dump(),
                        "model": args.model,
                    }
                    for judgment in judgments
                ]
                append_cache(args.cache, cache_rows)
                for row in cache_rows:
                    cached[int(row["pair_id"])] = row
                for key, value in usage.items():
                    total_usage[key] += value
                print(
                    f"labeled={min(start + len(batch), len(missing))}/"
                    f"{len(missing)} total_tokens={total_usage['total_tokens']}",
                    flush=True,
                )
                break
            except Exception as error:
                last_error = error
                if attempt == args.max_retries:
                    raise
                time.sleep(2 ** (attempt - 1))
        if last_error is not None and not all(
            int(row["pair_id"]) in cached for row in batch
        ):
            raise last_error
    write_output(args.output, candidates, cached)
    metadata = {
        "provider": args.provider,
        "model": args.model,
        "candidate_count": len(candidates),
        "cache": str(args.cache),
        "output": str(args.output),
        "usage": total_usage,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
