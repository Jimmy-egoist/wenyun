#!/usr/bin/env python3
"""Train a two-view aspect-aware contrastive GNN recommender.

View 1 is a positive user-item interaction graph. View 2 augments the same
interaction graph with user-aspect and item-aspect edges extracted from
training reviews. Aspect relevance and signed sentiment are propagated with
separate normalized adjacency matrices. User and item representations from
the two views are aligned with an InfoNCE objective. The ``joint`` variant
adds differentiable aspect canonicalization gates whose merge probabilities
are directly optimized by the recommendation objective.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "outputs" / "small_validation_dataset"
DEFAULT_OUTPUT = ROOT / "outputs" / "aspect_contrastive_gnn"
DEFAULT_GLOVE = ROOT.parent / "yelp_official_baselines" / "embeddings" / "glove.6B.50d.txt"
POSITIVE_THRESHOLD = 4.0
K_VALUES = (10, 20)
TOKEN_RE = re.compile(r"[0-9A-Za-z]+")


@dataclass
class InteractionData:
    num_users: int
    num_items: int
    num_aspects: int
    train_pairs: np.ndarray
    train_positive_mask: np.ndarray
    train_user_positive: dict[int, set[int]]
    val_user_positive: dict[int, set[int]]
    test_user_positive: dict[int, set[int]]


@dataclass
class GraphBundle:
    ui_adj: torch.Tensor
    uia_relevance_adj: torch.Tensor
    uia_sentiment_adj: torch.Tensor
    num_ui_nodes: int
    num_uia_nodes: int
    ui_undirected_edges: int
    user_aspect_edges: int
    item_aspect_edges: int
    signed_aspect_edges: int


@dataclass
class CanonicalizationData:
    aspect_names: list[str]
    candidate_pairs: np.ndarray
    pair_features: np.ndarray
    semantic_prior: np.ndarray
    prior_source: str


@dataclass
class TrainConfig:
    embedding_dim: int
    layers: int
    epochs: int
    min_epochs: int
    patience: int
    learning_rate: float
    contrastive_weight: float
    temperature: float
    contrastive_batch: int
    regularization: float
    canonical_prior_weight: float
    canonical_cluster_weight: float
    canonical_sparse_weight: float
    merge_strength: float
    canonical_feature_mode: str
    freeze_canonical_gates: bool
    seed: int
    threads: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", default="full,no_cl,ui_only")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--contrastive-weight", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--contrastive-batch", type=int, default=1024)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--aspect-pairs", type=Path, default=None,
                        help="Optional CSV: aspect_i,aspect_j,llm_confidence")
    parser.add_argument("--glove", type=Path, default=DEFAULT_GLOVE)
    parser.add_argument("--semantic-dim", type=int, default=50)
    parser.add_argument("--merge-candidates", type=int, default=4)
    parser.add_argument("--merge-threshold", type=float, default=0.45)
    parser.add_argument("--merge-strength", type=float, default=0.5)
    parser.add_argument(
        "--canonical-feature-mode",
        choices=("all", "semantic", "behavior"),
        default="all",
    )
    parser.add_argument("--freeze-canonical-gates", action="store_true")
    parser.add_argument("--canonical-prior-weight", type=float, default=0.02)
    parser.add_argument("--canonical-cluster-weight", type=float, default=0.002)
    parser.add_argument("--canonical-sparse-weight", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def load_interactions(data_dir: Path) -> InteractionData:
    num_users = sum(1 for _ in read_csv(data_dir / "users.csv"))
    num_items = sum(1 for _ in read_csv(data_dir / "items.csv"))
    num_aspects = sum(1 for _ in read_csv(data_dir / "aspects.csv"))

    train_pairs: list[tuple[int, int]] = []
    split_positive: dict[str, dict[int, set[int]]] = {
        "train": defaultdict(set),
        "val": defaultdict(set),
        "test": defaultdict(set),
    }
    for row in read_csv(data_dir / "interactions.csv"):
        if float(row["rating"]) < POSITIVE_THRESHOLD:
            continue
        split = row["split"]
        user = int(row["user_idx"])
        item = int(row["item_idx"])
        split_positive[split][user].add(item)
        if split == "train":
            train_pairs.append((user, item))

    train_positive_mask = np.zeros((num_users, num_items), dtype=np.bool_)
    for user, items in split_positive["train"].items():
        train_positive_mask[user, list(items)] = True

    return InteractionData(
        num_users=num_users,
        num_items=num_items,
        num_aspects=num_aspects,
        train_pairs=np.asarray(train_pairs, dtype=np.int64),
        train_positive_mask=train_positive_mask,
        train_user_positive=dict(split_positive["train"]),
        val_user_positive=dict(split_positive["val"]),
        test_user_positive=dict(split_positive["test"]),
    )


def load_aspect_names(data_dir: Path, num_aspects: int) -> list[str]:
    rows = sorted(
        read_csv(data_dir / "aspects.csv"),
        key=lambda row: int(row["aspect_idx"]),
    )
    names = [row["aspect"] for row in rows]
    if len(names) != num_aspects:
        raise ValueError(f"Loaded {len(names)} aspect names, expected {num_aspects}")
    return names


def load_semantic_vectors(
    path: Path,
    names: list[str],
    dimension: int,
) -> np.ndarray:
    needed = {token for name in names for token in TOKEN_RE.findall(name.lower())}
    words: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) == dimension + 1 and parts[0] in needed:
                words[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    vectors = np.zeros((len(names), dimension), dtype=np.float32)
    for index, name in enumerate(names):
        matched = [
            words[token]
            for token in TOKEN_RE.findall(name.lower())
            if token in words
        ]
        if matched:
            vectors[index] = np.mean(matched, axis=0)
    return F.normalize(torch.from_numpy(vectors), dim=1).numpy()


def column_cosine(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    return normalized.T @ normalized


def build_canonicalization_data(
    data_dir: Path,
    data: InteractionData,
    glove_path: Path,
    semantic_dim: int,
    max_candidates: int,
    threshold: float,
    aspect_pairs_path: Path | None,
    feature_mode: str = "all",
) -> CanonicalizationData:
    """Create semantic candidates and train-only collaborative pair features."""
    names = load_aspect_names(data_dir, data.num_aspects)
    semantic = load_semantic_vectors(glove_path, names, semantic_dim)
    semantic_similarity = semantic @ semantic.T
    user_behavior = np.zeros((data.num_users, data.num_aspects), dtype=np.float32)
    item_behavior = np.zeros((data.num_items, data.num_aspects), dtype=np.float32)
    for row in read_csv(data_dir / "uia_review_aspect_edges.csv"):
        if row["split"] != "train":
            continue
        aspect = int(row["aspect_idx"])
        user_behavior[int(row["user_idx"]), aspect] += 1.0
        item_behavior[int(row["item_idx"]), aspect] += 1.0
    user_similarity = column_cosine(np.log1p(user_behavior))
    item_similarity = column_cosine(np.log1p(item_behavior))

    proposed: dict[tuple[int, int], float] = {}
    if aspect_pairs_path is not None:
        name_to_id = {name: index for index, name in enumerate(names)}
        for row in read_csv(aspect_pairs_path):
            left_value = row["aspect_i"].strip()
            right_value = row["aspect_j"].strip()
            left = (
                int(left_value)
                if left_value.isdigit()
                else name_to_id.get(left_value, -1)
            )
            right = (
                int(right_value)
                if right_value.isdigit()
                else name_to_id.get(right_value, -1)
            )
            if not (0 <= left < data.num_aspects and 0 <= right < data.num_aspects):
                raise ValueError(f"Unknown aspect pair: {left_value}, {right_value}")
            if left != right:
                proposed[tuple(sorted((left, right)))] = float(
                    row["llm_confidence"]
                )
        prior_source = f"LLM confidence file: {aspect_pairs_path}"
    else:
        # This reproducible fallback replaces only the unavailable LLM prior.
        # Recommendation gradients still make the final merge decisions.
        for left in range(data.num_aspects):
            neighbors = np.argsort(-semantic_similarity[left])
            kept = 0
            for right in neighbors:
                similarity = float(semantic_similarity[left, right])
                if right == left or similarity < threshold:
                    continue
                pair = tuple(sorted((left, int(right))))
                proposed[pair] = max(proposed.get(pair, 0.0), similarity)
                kept += 1
                if kept >= max_candidates:
                    break
        prior_source = f"GloVe-{semantic_dim} cosine fallback"

    if not proposed:
        raise ValueError("No aspect canonicalization candidates were generated")
    candidate_pairs = np.asarray(sorted(proposed), dtype=np.int64)
    prior = np.asarray(
        [proposed[tuple(pair)] for pair in candidate_pairs],
        dtype=np.float32,
    )
    pair_features = np.stack(
        [
            semantic_similarity[candidate_pairs[:, 0], candidate_pairs[:, 1]],
            user_similarity[candidate_pairs[:, 0], candidate_pairs[:, 1]],
            item_similarity[candidate_pairs[:, 0], candidate_pairs[:, 1]],
            np.abs(
                user_similarity[candidate_pairs[:, 0], candidate_pairs[:, 1]]
                - item_similarity[candidate_pairs[:, 0], candidate_pairs[:, 1]]
            ),
            prior,
        ],
        axis=1,
    ).astype(np.float32)
    if feature_mode == "semantic":
        pair_features[:, 1:4] = 0.0
    elif feature_mode == "behavior":
        pair_features[:, 0] = 0.0
        pair_features[:, 4] = 0.5
        prior.fill(0.5)
    elif feature_mode != "all":
        raise ValueError(f"Unknown canonical feature mode: {feature_mode}")
    return CanonicalizationData(
        aspect_names=names,
        candidate_pairs=candidate_pairs,
        pair_features=pair_features,
        semantic_prior=np.clip(prior, 0.05, 0.95),
        prior_source=f"{prior_source}; feature_mode={feature_mode}",
    )


def normalized_sparse_adjacency(
    num_nodes: int,
    src: list[int],
    dst: list[int],
    weights: list[float],
) -> torch.Tensor:
    src_array = np.asarray(src, dtype=np.int64)
    dst_array = np.asarray(dst, dtype=np.int64)
    weight_array = np.asarray(weights, dtype=np.float32)
    degree = np.bincount(
        src_array,
        weights=np.abs(weight_array),
        minlength=num_nodes,
    ).astype(np.float32)
    inv_sqrt_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = np.power(degree[nonzero], -0.5)
    normalized = weight_array * inv_sqrt_degree[src_array] * inv_sqrt_degree[dst_array]
    indices = torch.from_numpy(np.stack([src_array, dst_array]))
    values = torch.from_numpy(normalized)
    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
        dtype=torch.float32,
    ).coalesce()


def add_bidirectional(
    src: list[int],
    dst: list[int],
    weights: list[float],
    left: int,
    right: int,
    weight: float,
) -> None:
    src.extend((left, right))
    dst.extend((right, left))
    weights.extend((weight, weight))


def build_graphs(data_dir: Path, data: InteractionData) -> GraphBundle:
    ui_nodes = data.num_users + data.num_items
    uia_nodes = ui_nodes + data.num_aspects

    ui_src: list[int] = []
    ui_dst: list[int] = []
    ui_weight: list[float] = []
    relevance_src: list[int] = []
    relevance_dst: list[int] = []
    relevance_weight: list[float] = []
    sentiment_src: list[int] = []
    sentiment_dst: list[int] = []
    sentiment_weight: list[float] = []

    for user, item in data.train_pairs:
        item_node = data.num_users + int(item)
        add_bidirectional(ui_src, ui_dst, ui_weight, int(user), item_node, 1.0)
        add_bidirectional(
            relevance_src,
            relevance_dst,
            relevance_weight,
            int(user),
            item_node,
            1.0,
        )

    # Each aggregate stores total mentions and signed polarity count.
    user_aspect: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    item_aspect: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in read_csv(data_dir / "uia_review_aspect_edges.csv"):
        if row["split"] != "train":
            continue
        user = int(row["user_idx"])
        item = int(row["item_idx"])
        aspect = int(row["aspect_idx"])
        polarity = {"positive": 1, "neutral": 0, "negative": -1}.get(
            row["sentiment"], 0
        )
        user_stats = user_aspect[(user, aspect)]
        user_stats[0] += 1
        user_stats[1] += polarity
        item_stats = item_aspect[(item, aspect)]
        item_stats[0] += 1
        item_stats[1] += polarity

    signed_edges = 0

    def add_aspect_edges(
        aggregates: dict[tuple[int, int], list[int]],
        node_offset: int,
    ) -> None:
        nonlocal signed_edges
        aspect_offset = data.num_users + data.num_items
        for (entity, aspect), (mentions, signed_count) in aggregates.items():
            entity_node = node_offset + entity
            aspect_node = aspect_offset + aspect
            relevance = math.log1p(mentions)
            add_bidirectional(
                relevance_src,
                relevance_dst,
                relevance_weight,
                entity_node,
                aspect_node,
                relevance,
            )
            if signed_count:
                polarity = signed_count / mentions
                signed_value = relevance * polarity
                add_bidirectional(
                    sentiment_src,
                    sentiment_dst,
                    sentiment_weight,
                    entity_node,
                    aspect_node,
                    signed_value,
                )
                signed_edges += 1

    add_aspect_edges(user_aspect, node_offset=0)
    add_aspect_edges(item_aspect, node_offset=data.num_users)

    return GraphBundle(
        ui_adj=normalized_sparse_adjacency(ui_nodes, ui_src, ui_dst, ui_weight),
        uia_relevance_adj=normalized_sparse_adjacency(
            uia_nodes,
            relevance_src,
            relevance_dst,
            relevance_weight,
        ),
        uia_sentiment_adj=normalized_sparse_adjacency(
            uia_nodes,
            sentiment_src,
            sentiment_dst,
            sentiment_weight,
        ),
        num_ui_nodes=ui_nodes,
        num_uia_nodes=uia_nodes,
        ui_undirected_edges=len(data.train_pairs),
        user_aspect_edges=len(user_aspect),
        item_aspect_edges=len(item_aspect),
        signed_aspect_edges=signed_edges,
    )


class LightGraphEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        layers: int,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.layers = layers
        self.register_buffer("adjacency", adjacency)
        nn.init.normal_(self.embedding.weight, std=0.05)

    def forward(self) -> torch.Tensor:
        current = self.embedding.weight
        outputs = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(self.adjacency, current)
            outputs.append(current)
        return torch.stack(outputs, dim=0).mean(dim=0)


class AspectGraphEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        layers: int,
        relevance_adjacency: torch.Tensor,
        sentiment_adjacency: torch.Tensor,
        aspect_offset: int,
        num_aspects: int,
        canonicalization: CanonicalizationData | None = None,
        merge_strength: float = 0.0,
        freeze_canonical_gates: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.layers = layers
        self.aspect_offset = aspect_offset
        self.num_aspects = num_aspects
        self.merge_strength = merge_strength
        self.sentiment_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("relevance_adjacency", relevance_adjacency)
        self.register_buffer("sentiment_adjacency", sentiment_adjacency)
        self.gate_logit: nn.Parameter | None = None
        self.gate_mlp: nn.Sequential | None = None
        if canonicalization is not None:
            pairs = torch.from_numpy(canonicalization.candidate_pairs)
            features = torch.from_numpy(canonicalization.pair_features)
            prior = torch.from_numpy(canonicalization.semantic_prior)
            self.register_buffer("candidate_pairs", pairs)
            self.register_buffer("pair_features", features)
            self.register_buffer("semantic_prior", prior)
            prior_logit = torch.log(prior / (1.0 - prior))
            self.gate_logit = nn.Parameter(prior_logit.clone())
            self.gate_mlp = nn.Sequential(
                nn.Linear(features.shape[1], 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            nn.init.xavier_uniform_(self.gate_mlp[0].weight)
            nn.init.zeros_(self.gate_mlp[0].bias)
            # Initial gates equal the semantic/LLM prior exactly.
            nn.init.zeros_(self.gate_mlp[2].weight)
            nn.init.zeros_(self.gate_mlp[2].bias)
            if freeze_canonical_gates:
                self.gate_logit.requires_grad_(False)
                for parameter in self.gate_mlp.parameters():
                    parameter.requires_grad_(False)
        nn.init.normal_(self.embedding.weight, std=0.05)

    def merge_gates(self) -> torch.Tensor | None:
        if self.gate_logit is None or self.gate_mlp is None:
            return None
        residual = self.gate_mlp(self.pair_features).squeeze(1)
        return torch.sigmoid(self.gate_logit + residual)

    def aspect_messages(
        self,
        current: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        aspects = current[
            self.aspect_offset : self.aspect_offset + self.num_aspects
        ]
        left = self.candidate_pairs[:, 0]
        right = self.candidate_pairs[:, 1]
        messages = torch.zeros_like(aspects)
        degree = torch.ones(
            self.num_aspects,
            device=current.device,
            dtype=current.dtype,
        )
        messages.index_add_(0, left, gates[:, None] * aspects[right])
        messages.index_add_(0, right, gates[:, None] * aspects[left])
        degree.index_add_(0, left, gates)
        degree.index_add_(0, right, gates)
        output = torch.zeros_like(current)
        output[
            self.aspect_offset : self.aspect_offset + self.num_aspects
        ] = messages / degree[:, None]
        return output

    def forward(self) -> torch.Tensor:
        current = self.embedding.weight
        outputs = [current]
        sentiment_scale = torch.sigmoid(self.sentiment_logit)
        gates = self.merge_gates()
        for _ in range(self.layers):
            relevance_message = torch.sparse.mm(self.relevance_adjacency, current)
            sentiment_message = torch.sparse.mm(self.sentiment_adjacency, current)
            current = relevance_message + sentiment_scale * sentiment_message
            if gates is not None:
                current = (
                    current
                    + self.merge_strength * self.aspect_messages(outputs[-1], gates)
                )
            outputs.append(current)
        return torch.stack(outputs, dim=0).mean(dim=0)


class AspectContrastiveGNN(nn.Module):
    def __init__(
        self,
        data: InteractionData,
        graphs: GraphBundle,
        embedding_dim: int,
        layers: int,
        variant: str,
        canonicalization: CanonicalizationData | None,
        merge_strength: float,
        freeze_canonical_gates: bool,
    ) -> None:
        super().__init__()
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.variant = variant
        self.ui_encoder = LightGraphEncoder(
            graphs.num_ui_nodes,
            embedding_dim,
            layers,
            graphs.ui_adj,
        )
        self.aspect_encoder = None
        if variant != "ui_only":
            self.aspect_encoder = AspectGraphEncoder(
                graphs.num_uia_nodes,
                embedding_dim,
                layers,
                graphs.uia_relevance_adj,
                graphs.uia_sentiment_adj,
                aspect_offset=data.num_users + data.num_items,
                num_aspects=data.num_aspects,
                canonicalization=canonicalization if variant == "joint" else None,
                merge_strength=merge_strength,
                freeze_canonical_gates=freeze_canonical_gates,
            )
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self) -> dict[str, torch.Tensor | None]:
        ui_nodes = self.ui_encoder()
        ui_users = ui_nodes[: self.num_users]
        ui_items = ui_nodes[self.num_users : self.num_users + self.num_items]
        if self.aspect_encoder is None:
            return {
                "users": ui_users,
                "items": ui_items,
                "ui_users": ui_users,
                "ui_items": ui_items,
                "aspect_users": None,
                "aspect_items": None,
                "canonical_aspects": None,
                "merge_gates": None,
            }

        aspect_nodes = self.aspect_encoder()
        aspect_users = aspect_nodes[: self.num_users]
        aspect_items = aspect_nodes[
            self.num_users : self.num_users + self.num_items
        ]
        canonical_aspects = aspect_nodes[
            self.num_users + self.num_items :
        ]
        gate = torch.sigmoid(self.fusion_logit)
        users = gate * ui_users + (1.0 - gate) * aspect_users
        items = gate * ui_items + (1.0 - gate) * aspect_items
        return {
            "users": users,
            "items": items,
            "ui_users": ui_users,
            "ui_items": ui_items,
            "aspect_users": aspect_users,
            "aspect_items": aspect_items,
            "canonical_aspects": canonical_aspects,
            "merge_gates": self.aspect_encoder.merge_gates(),
        }


def sample_negatives(
    users: np.ndarray,
    positive_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, positive_mask.shape[1], size=len(users), dtype=np.int64)
    invalid = positive_mask[users, negatives]
    while invalid.any():
        negatives[invalid] = rng.integers(
            0,
            positive_mask.shape[1],
            size=int(invalid.sum()),
            dtype=np.int64,
        )
        invalid = positive_mask[users, negatives]
    return negatives


def symmetric_info_nce(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    first = F.normalize(first, dim=1)
    second = F.normalize(second, dim=1)
    logits = first @ second.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


def contrastive_loss(
    embeddings: dict[str, torch.Tensor | None],
    data: InteractionData,
    batch_size: int,
    temperature: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    ui_users = embeddings["ui_users"]
    ui_items = embeddings["ui_items"]
    aspect_users = embeddings["aspect_users"]
    aspect_items = embeddings["aspect_items"]
    assert isinstance(ui_users, torch.Tensor)
    assert isinstance(ui_items, torch.Tensor)
    assert isinstance(aspect_users, torch.Tensor)
    assert isinstance(aspect_items, torch.Tensor)

    user_count = min(batch_size, data.num_users)
    item_count = min(batch_size, data.num_items)
    user_ids = torch.from_numpy(
        rng.choice(data.num_users, size=user_count, replace=False)
    ).long()
    item_ids = torch.from_numpy(
        rng.choice(data.num_items, size=item_count, replace=False)
    ).long()
    user_loss = symmetric_info_nce(
        ui_users[user_ids], aspect_users[user_ids], temperature
    )
    item_loss = symmetric_info_nce(
        ui_items[item_ids], aspect_items[item_ids], temperature
    )
    return 0.5 * (user_loss + item_loss)


def canonicalization_losses(
    embeddings: dict[str, torch.Tensor | None],
    encoder: AspectGraphEncoder,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gates = embeddings["merge_gates"]
    aspects = embeddings["canonical_aspects"]
    assert isinstance(gates, torch.Tensor)
    assert isinstance(aspects, torch.Tensor)
    prior_loss = F.binary_cross_entropy(gates, encoder.semantic_prior)
    left = encoder.candidate_pairs[:, 0]
    right = encoder.candidate_pairs[:, 1]
    cluster_loss = (
        gates * (aspects[left] - aspects[right]).pow(2).mean(dim=1)
    ).mean()
    sparse_loss = gates.mean()
    return prior_loss, cluster_loss, sparse_loss


def write_learned_gates(
    path: Path,
    model: AspectContrastiveGNN,
    canonicalization: CanonicalizationData,
) -> dict[str, float | int | str]:
    assert model.aspect_encoder is not None
    with torch.no_grad():
        gate_tensor = model.aspect_encoder.merge_gates()
    assert gate_tensor is not None
    gates = gate_tensor.cpu().numpy()
    rows: list[dict[str, float | str]] = []
    for index, (left, right) in enumerate(canonicalization.candidate_pairs):
        features = canonicalization.pair_features[index]
        rows.append(
            {
                "aspect_i": canonicalization.aspect_names[int(left)],
                "aspect_j": canonicalization.aspect_names[int(right)],
                "semantic_similarity": float(features[0]),
                "user_behavior_similarity": float(features[1]),
                "item_behavior_similarity": float(features[2]),
                "behavior_disagreement": float(features[3]),
                "prior_confidence": float(canonicalization.semantic_prior[index]),
                "learned_gate": float(gates[index]),
                "gate_shift": float(
                    gates[index] - canonicalization.semantic_prior[index]
                ),
            }
        )
    rows.sort(key=lambda row: float(row["learned_gate"]), reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shifts = gates - canonicalization.semantic_prior
    return {
        "prior_source": canonicalization.prior_source,
        "num_candidates": len(gates),
        "mean_gate": float(gates.mean()),
        "min_gate": float(gates.min()),
        "max_gate": float(gates.max()),
        "mean_absolute_shift": float(np.abs(shifts).mean()),
        "num_above_0.5": int((gates >= 0.5).sum()),
        "artifact": str(path),
    }


def ranking_metrics(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    train_positive_mask: np.ndarray,
    eval_positive: dict[int, set[int]],
) -> dict[str, float | int]:
    with torch.no_grad():
        scores = (user_embeddings @ item_embeddings.T).cpu().numpy()
    scores[train_positive_mask] = -np.inf

    metric_values: dict[str, list[float]] = {
        f"Recall@{k}": [] for k in K_VALUES
    }
    metric_values.update({f"NDCG@{k}": [] for k in K_VALUES})
    for user, positives in sorted(eval_positive.items()):
        if not positives:
            continue
        order = np.argsort(-scores[user])
        for k in K_VALUES:
            topk = order[:k]
            hits = [rank for rank, item in enumerate(topk) if int(item) in positives]
            recall = len(hits) / len(positives)
            dcg = sum(1.0 / math.log2(rank + 2) for rank in hits)
            ideal = sum(
                1.0 / math.log2(rank + 2)
                for rank in range(min(len(positives), k))
            )
            metric_values[f"Recall@{k}"].append(recall)
            metric_values[f"NDCG@{k}"].append(dcg / ideal if ideal else 0.0)

    return {
        "evaluated_users": len(eval_positive),
        **{key: float(np.mean(values)) for key, values in metric_values.items()},
    }


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_variant(
    variant: str,
    data: InteractionData,
    graphs: GraphBundle,
    config: TrainConfig,
    output_dir: Path,
    canonicalization: CanonicalizationData | None,
) -> dict[str, object]:
    if variant == "joint" and canonicalization is None:
        raise ValueError("The joint variant requires canonicalization candidates")
    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    model = AspectContrastiveGNN(
        data,
        graphs,
        config.embedding_dim,
        config.layers,
        variant,
        canonicalization,
        config.merge_strength,
        config.freeze_canonical_gates,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    users_np = data.train_pairs[:, 0]
    positive_items_np = data.train_pairs[:, 1]
    users = torch.from_numpy(users_np).long()
    positive_items = torch.from_numpy(positive_items_np).long()

    checkpoint_path = output_dir / f"best_{variant}.pt"
    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_ndcg = -math.inf
    stale_epochs = 0
    start_time = time.time()
    cl_weight = config.contrastive_weight if variant in {"full", "joint"} else 0.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        negative_items_np = sample_negatives(
            users_np,
            data.train_positive_mask,
            rng,
        )
        negative_items = torch.from_numpy(negative_items_np).long()
        embeddings = model()
        user_embeddings = embeddings["users"]
        item_embeddings = embeddings["items"]
        assert isinstance(user_embeddings, torch.Tensor)
        assert isinstance(item_embeddings, torch.Tensor)

        selected_users = user_embeddings[users]
        selected_positive = item_embeddings[positive_items]
        selected_negative = item_embeddings[negative_items]
        positive_scores = (selected_users * selected_positive).sum(dim=1)
        negative_scores = (selected_users * selected_negative).sum(dim=1)
        bpr_loss = F.softplus(negative_scores - positive_scores).mean()

        cl_loss = torch.tensor(0.0)
        if cl_weight > 0:
            cl_loss = contrastive_loss(
                embeddings,
                data,
                config.contrastive_batch,
                config.temperature,
                rng,
            )

        prior_loss = torch.tensor(0.0)
        cluster_loss = torch.tensor(0.0)
        sparse_loss = torch.tensor(0.0)
        if variant == "joint":
            assert model.aspect_encoder is not None
            prior_loss, cluster_loss, sparse_loss = canonicalization_losses(
                embeddings,
                model.aspect_encoder,
            )

        raw_regularization = (
            model.ui_encoder.embedding(users).pow(2).mean()
            + model.ui_encoder.embedding(data.num_users + positive_items).pow(2).mean()
            + model.ui_encoder.embedding(data.num_users + negative_items).pow(2).mean()
        )
        if model.aspect_encoder is not None:
            raw_regularization = raw_regularization + (
                model.aspect_encoder.embedding(users).pow(2).mean()
                + model.aspect_encoder.embedding(
                    data.num_users + positive_items
                ).pow(2).mean()
                + model.aspect_encoder.embedding(
                    data.num_users + negative_items
                ).pow(2).mean()
            )
        loss = (
            bpr_loss
            + cl_weight * cl_loss
            + config.regularization * raw_regularization
            + config.canonical_prior_weight * prior_loss
            + config.canonical_cluster_weight * cluster_loss
            + config.canonical_sparse_weight * sparse_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_embeddings = model()
        validation = ranking_metrics(
            validation_embeddings["users"],
            validation_embeddings["items"],
            data.train_positive_mask,
            data.val_user_positive,
        )
        val_ndcg = float(validation["NDCG@10"])
        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "bpr_loss": float(bpr_loss.item()),
            "contrastive_loss": float(cl_loss.item()),
            "canonical_prior_loss": float(prior_loss.item()),
            "canonical_cluster_loss": float(cluster_loss.item()),
            "canonical_sparse_loss": float(sparse_loss.item()),
            "val_recall@10": float(validation["Recall@10"]),
            "val_ndcg@10": val_ndcg,
            "val_recall@20": float(validation["Recall@20"]),
            "val_ndcg@20": float(validation["NDCG@20"]),
        }
        history.append(row)

        if val_ndcg > best_ndcg + 1e-8:
            best_ndcg = val_ndcg
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "variant": variant,
                    "config": asdict(config),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        print(
            f"variant={variant} epoch={epoch:03d} loss={loss.item():.5f} "
            f"bpr={bpr_loss.item():.5f} cl={cl_loss.item():.5f} "
            f"canon={prior_loss.item():.5f}/{cluster_loss.item():.5f}/"
            f"{sparse_loss.item():.5f} "
            f"val_recall@10={validation['Recall@10']:.6f} "
            f"val_ndcg@10={val_ndcg:.6f} best_epoch={best_epoch}",
            flush=True,
        )
        if epoch >= config.min_epochs and stale_epochs >= config.patience:
            print(f"variant={variant} early_stop epoch={epoch}", flush=True)
            break

    write_history(output_dir / f"history_{variant}.csv", history)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        best_embeddings = model()
    test = ranking_metrics(
        best_embeddings["users"],
        best_embeddings["items"],
        data.train_positive_mask,
        data.test_user_positive,
    )
    elapsed = time.time() - start_time
    canonicalization_summary = None
    if variant == "joint":
        assert canonicalization is not None
        canonicalization_summary = write_learned_gates(
            output_dir / "learned_aspect_gates.csv",
            model,
            canonicalization,
        )
    print(
        f"variant={variant} selected_epoch={best_epoch} "
        f"test_recall@10={test['Recall@10']:.6f} "
        f"test_ndcg@10={test['NDCG@10']:.6f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "variant": variant,
        "model": {
            "full": "AspectCL-GNN",
            "joint": "JointCanonical-AspectCL-GNN",
            "no_cl": "AspectGNN-noCL",
            "ui_only": "UI-LightGCN",
        }[variant],
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation": checkpoint["validation"],
        "test": test,
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint_path),
        "canonicalization": canonicalization_summary,
    }


def write_results(
    output_dir: Path,
    data: InteractionData,
    graphs: GraphBundle,
    config: TrainConfig,
    results: list[dict[str, object]],
    canonicalization: CanonicalizationData | None,
) -> None:
    graph_stats = {
        "num_ui_nodes": graphs.num_ui_nodes,
        "num_uia_nodes": graphs.num_uia_nodes,
        "ui_undirected_edges": graphs.ui_undirected_edges,
        "user_aspect_edges": graphs.user_aspect_edges,
        "item_aspect_edges": graphs.item_aspect_edges,
        "signed_aspect_edges": graphs.signed_aspect_edges,
    }
    payload = {
        "setting": {
            "task": "implicit top-K recommendation",
            "positive_threshold": POSITIVE_THRESHOLD,
            "split": "per-user temporal train/validation/test",
            "selection_metric": "validation NDCG@10",
            "candidate_protocol": "all 599 items; training positives filtered",
            "num_users": data.num_users,
            "num_items": data.num_items,
            "num_aspects": data.num_aspects,
            "train_positive_pairs": len(data.train_pairs),
            "config": asdict(config),
            "canonicalization_candidates": ({
                "prior_source": canonicalization.prior_source,
                "count": len(canonicalization.candidate_pairs),
                "features": [
                    "semantic_similarity",
                    "user_behavior_similarity",
                    "item_behavior_similarity",
                    "behavior_disagreement",
                    "prior_confidence",
                ],
                "leakage_policy": "collaborative features use training review-aspect edges only",
            } if canonicalization is not None else None),
        },
        "graph": graph_stats,
        "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    rows = []
    for result in results:
        validation = result["validation"]
        test = result["test"]
        assert isinstance(validation, dict)
        assert isinstance(test, dict)
        rows.append(
            {
                "model": result["model"],
                "variant": result["variant"],
                "best_epoch": result["best_epoch"],
                "epochs_run": result["epochs_run"],
                "val_recall@10": validation["Recall@10"],
                "val_ndcg@10": validation["NDCG@10"],
                "val_recall@20": validation["Recall@20"],
                "val_ndcg@20": validation["NDCG@20"],
                "test_recall@10": test["Recall@10"],
                "test_ndcg@10": test["NDCG@10"],
                "test_recall@20": test["Recall@20"],
                "test_ndcg@20": test["NDCG@20"],
                "test_users": test["evaluated_users"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    with (output_dir / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    supported = {"joint", "full", "no_cl", "ui_only"}
    unknown = set(variants) - supported
    if unknown:
        raise ValueError(f"Unsupported variants: {sorted(unknown)}")

    torch.set_num_threads(args.threads)
    config = TrainConfig(
        embedding_dim=args.embedding_dim,
        layers=args.layers,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        contrastive_batch=args.contrastive_batch,
        regularization=args.regularization,
        canonical_prior_weight=args.canonical_prior_weight,
        canonical_cluster_weight=args.canonical_cluster_weight,
        canonical_sparse_weight=args.canonical_sparse_weight,
        merge_strength=args.merge_strength,
        canonical_feature_mode=args.canonical_feature_mode,
        freeze_canonical_gates=args.freeze_canonical_gates,
        seed=args.seed,
        threads=args.threads,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    data = load_interactions(args.data_dir)
    graphs = build_graphs(args.data_dir, data)
    canonicalization = None
    if "joint" in variants:
        canonicalization = build_canonicalization_data(
            args.data_dir,
            data,
            args.glove,
            args.semantic_dim,
            args.merge_candidates,
            args.merge_threshold,
            args.aspect_pairs,
            args.canonical_feature_mode,
        )
    print(
        json.dumps(
            {
                "num_users": data.num_users,
                "num_items": data.num_items,
                "num_aspects": data.num_aspects,
                "train_positive_pairs": len(data.train_pairs),
                "ui_edges": graphs.ui_undirected_edges,
                "user_aspect_edges": graphs.user_aspect_edges,
                "item_aspect_edges": graphs.item_aspect_edges,
                "signed_aspect_edges": graphs.signed_aspect_edges,
                "canonicalization_candidates": (
                    len(canonicalization.candidate_pairs)
                    if canonicalization is not None else 0
                ),
                "canonicalization_prior": (
                    canonicalization.prior_source
                    if canonicalization is not None else None
                ),
            },
            indent=2,
        ),
        flush=True,
    )

    results = [
        train_variant(
            variant,
            data,
            graphs,
            config,
            args.output_dir,
            canonicalization,
        )
        for variant in variants
    ]
    write_results(
        args.output_dir,
        data,
        graphs,
        config,
        results,
        canonicalization,
    )
    print(args.output_dir / "results.csv", flush=True)


if __name__ == "__main__":
    main()
