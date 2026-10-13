# Aspect canonicalization pipeline report

Seeds: 42,2026,7. Values are mean ± sample standard deviation.

| Experiment | Test Recall@10 | Test NDCG@10 | Test Recall@20 | Test NDCG@20 |
|---|---:|---:|---:|---:|
| joint_llm_behavior | 0.181167 ± 0.001355 | 0.091093 ± 0.000645 | 0.309395 ± 0.001565 | 0.123351 ± 0.000671 |
| joint_behavior_only | 0.181167 ± 0.001565 | 0.091051 ± 0.000882 | 0.308599 ± 0.000452 | 0.123119 ± 0.000499 |
| joint_rec_only | 0.181095 ± 0.001445 | 0.091026 ± 0.000666 | 0.310335 ± 0.000697 | 0.123523 ± 0.000594 |
| baseline_full | 0.181023 ± 0.000546 | 0.090795 ± 0.000346 | 0.308599 ± 0.005231 | 0.122731 ± 0.000995 |
| joint_fixed_prior | 0.180444 ± 0.002993 | 0.090647 ± 0.001501 | 0.309033 ± 0.002279 | 0.122974 ± 0.001295 |
| joint_semantic_only | 0.180299 ± 0.002837 | 0.090610 ± 0.001473 | 0.308961 ± 0.002170 | 0.122960 ± 0.001273 |

## Mean change from baseline

- **joint_behavior_only** — Recall@10: +0.08%; NDCG@10: +0.28%; Recall@20: +0.00%; NDCG@20: +0.32%
- **joint_fixed_prior** — Recall@10: -0.32%; NDCG@10: -0.16%; Recall@20: +0.14%; NDCG@20: +0.20%
- **joint_llm_behavior** — Recall@10: +0.08%; NDCG@10: +0.33%; Recall@20: +0.26%; NDCG@20: +0.51%
- **joint_rec_only** — Recall@10: +0.04%; NDCG@10: +0.25%; Recall@20: +0.56%; NDCG@20: +0.65%
- **joint_semantic_only** — Recall@10: -0.40%; NDCG@10: -0.20%; Recall@20: +0.12%; NDCG@20: +0.19%

## LLM candidate audit

- Candidate pairs: 464
- Relation counts: related_but_distinct=380, semantic_equivalent=4, unrelated=80
- The LLM field is merge probability, validated against the relation label before training.

## Recommendation-guided gate changes

Largest mean decreases across seeds:

- `soup / gumbo`: 0.320 → 0.253 (-0.067)
- `sushi / seafood`: 0.270 → 0.225 (-0.045)
- `location / town`: 0.700 → 0.661 (-0.039)
- `decor / patio`: 0.140 → 0.103 (-0.037)
- `breakfast / brunch`: 0.350 → 0.316 (-0.034)
- `fish / salmon`: 0.320 → 0.288 (-0.032)
- `neighborhood / town`: 0.280 → 0.252 (-0.028)
- `hotel / neighborhood`: 0.160 → 0.133 (-0.027)

Largest mean increases across seeds:

- `shrimp / gumbo`: 0.180 → 0.220 (+0.040)
- `bbq / gumbo`: 0.080 → 0.093 (+0.013)
- `shrimp / seafood`: 0.350 → 0.363 (+0.013)
- `vegetable / vegetarian`: 0.280 → 0.292 (+0.012)
- `gumbo / grits`: 0.060 → 0.072 (+0.012)
- `hotel / town`: 0.120 → 0.131 (+0.011)
- `ramen / mushroom`: 0.080 → 0.089 (+0.009)
- `sushi / lobster`: 0.080 → 0.087 (+0.007)

With only three seeds, differences are descriptive and are not claimed as statistically significant.
