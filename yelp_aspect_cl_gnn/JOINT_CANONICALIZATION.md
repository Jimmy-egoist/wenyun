# Recommendation-guided aspect canonicalization

The `joint` variant in `work/train_aspect_contrastive_gnn.py` keeps the
original 144 aspect nodes and learns a soft merge gate for each candidate
aspect pair. A gate controls aspect-to-aspect message propagation, so the BPR
ranking loss can increase or decrease the merge probability through ordinary
back-propagation.

Candidate features are:

- semantic similarity;
- training-only user behavior similarity;
- training-only item behavior similarity;
- user/item behavior disagreement;
- an LLM or semantic prior confidence.

The training objective adds prior consistency, gated representation
clustering, and sparsity terms to the original BPR and InfoNCE losses.

## LLM prior input

Pass `--aspect-pairs path/to/pairs.csv` to use LLM judgments. The CSV schema is:

```csv
aspect_i,aspect_j,llm_confidence
service,staff,0.92
food,menu,0.35
```

Aspect names or integer aspect indices are accepted. If this file is omitted,
GloVe cosine similarity is used as a reproducible semantic fallback. The
fallback was used for the included small-data result; it must not be described
as LLM supervision.

## Reproduce the small-data run

```bash
/net/ilabnas2/ilabhome/wenyun/miniconda3/envs/yelp-baselines-torch/bin/python \
  work/train_aspect_contrastive_gnn.py \
  --variants joint \
  --epochs 160 --min-epochs 30 --patience 20 \
  --embedding-dim 64 --layers 2 --learning-rate 0.003 \
  --contrastive-weight 0.02 --seed 42 --threads 16 \
  --output-dir results/joint_canonical
```

The selected checkpoint is epoch 125. Compared with the existing `full`
seed-42 result:

| Split / metric | Original full | Joint canonical | Relative change |
|---|---:|---:|---:|
| Validation Recall@10 | 0.190239 | 0.190889 | +0.34% |
| Validation NDCG@10 | 0.093976 | 0.094882 | +0.96% |
| Validation Recall@20 | 0.311063 | 0.315401 | +1.39% |
| Validation NDCG@20 | 0.124378 | 0.126163 | +1.44% |
| Test Recall@10 | 0.180516 | 0.180733 | +0.12% |
| Test NDCG@10 | 0.090892 | 0.091458 | +0.62% |
| Test Recall@20 | 0.303103 | 0.311564 | +2.79% |
| Test NDCG@20 | 0.121596 | 0.124282 | +2.21% |

These are single-seed results, so the gains are evidence for a successful
small-data integration, not a statistical significance claim. The learned
probabilities and their changes from the prior are in
`results/joint_canonical/learned_aspect_gates.csv`.
