# End-to-end recommendation-guided canonicalization pipeline

## Stages

1. Audit the fixed Yelp-small temporal split.
2. Propose aspect pairs from GloVe nearest neighbors plus lexical variants.
3. Ask an LLM for a calibrated `merge_probability` using structured output.
4. Validate relation/probability consistency and cache every completed batch.
5. Build semantic and training-only collaborative pair features.
6. Train recommendation-guided soft gates with BPR and InfoNCE.
7. Run three seeds for the baseline and five ablations.
8. Aggregate means, sample standard deviations, paired changes, and gate shifts.

The API receives aspect names and similarity summaries only. Review text and API
credentials are not written to prompts, caches, logs, or result files.

## Environment

```bash
PYTHON=/net/ilabnas2/ilabhome/wenyun/miniconda3/envs/yelp-baselines-torch/bin/python
$PYTHON -m pip install "openai>=2.50"
export OPENAI_API_KEY="..."
```

Run the complete pipeline:

```bash
$PYTHON work/run_canonicalization_pipeline.py \
  --model gpt-5.6-terra \
  --pair-file outputs/llm_aspect_pairs_v2.csv \
  --output-dir results/canonicalization_pipeline \
  --seeds 42,2026,7 \
  --resume
```

After LLM labeling has completed, later training resumes do not need the API key:

```bash
unset OPENAI_API_KEY
$PYTHON work/run_canonicalization_pipeline.py \
  --skip-labeling --resume \
  --pair-file outputs/llm_aspect_pairs_v2.csv \
  --output-dir results/canonicalization_pipeline
```

## LLM audit

The final GPT-5.6 Terra run labeled 464 pairs:

- 4 `semantic_equivalent`;
- 380 `related_but_distinct`;
- 80 `unrelated`.

The first exploratory Luna labeling file is retained for audit only. Its
ambiguous `confidence` interpretation was detected automatically and it was
not used for formal training. Formal labels use the unambiguous
`merge_probability` field and enforce consistency with the relation.

## Three-seed result

The recommended model is `joint_llm_behavior` because it has the strongest
NDCG@10 while retaining calibrated semantic constraints.

| Experiment | Test Recall@10 | Test NDCG@10 | Test Recall@20 | Test NDCG@20 |
|---|---:|---:|---:|---:|
| baseline_full | 0.181023 ± 0.000546 | 0.090795 ± 0.000346 | 0.308599 ± 0.005231 | 0.122731 ± 0.000995 |
| joint_llm_behavior | 0.181167 ± 0.001355 | 0.091093 ± 0.000645 | 0.309395 ± 0.001565 | 0.123351 ± 0.000671 |
| joint_behavior_only | 0.181167 ± 0.001565 | 0.091051 ± 0.000882 | 0.308599 ± 0.000452 | 0.123119 ± 0.000499 |
| joint_rec_only | 0.181095 ± 0.001445 | 0.091026 ± 0.000666 | 0.310335 ± 0.000697 | 0.123523 ± 0.000594 |
| joint_fixed_prior | 0.180444 ± 0.002993 | 0.090647 ± 0.001501 | 0.309033 ± 0.002279 | 0.122974 ± 0.001295 |
| joint_semantic_only | 0.180299 ± 0.002837 | 0.090610 ± 0.001473 | 0.308961 ± 0.002170 | 0.122960 ± 0.001273 |

Relative to the baseline, `joint_llm_behavior` improves mean NDCG@10 by
0.33% and mean NDCG@20 by 0.51%. With only three seeds, these results are
descriptive and are not a statistical significance claim.

The unconstrained `joint_rec_only` model moves gates much more strongly and
performs well at K=20, but it also assigns high merge weights to many related
but non-equivalent concepts. It is therefore an optimization ablation, not the
recommended canonicalization model.

## Artifacts

- `outputs/llm_aspect_pairs_v2.csv`: validated LLM priors.
- `outputs/llm_aspect_pair_cache_v2.jsonl`: resumable per-pair cache.
- `results/canonicalization_pipeline/all_runs.csv`: all 18 runs.
- `results/canonicalization_pipeline/summary.csv`: aggregate metrics.
- `results/canonicalization_pipeline/REPORT.md`: metrics and gate analysis.
- Each `seed_*` directory contains its checkpoint, history, result JSON, log,
  and learned gate CSV where applicable.
