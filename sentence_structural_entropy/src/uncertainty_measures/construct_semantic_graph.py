from __future__ import annotations

import heapq
import itertools
import json
import logging
import os
import pickle
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from openai import OpenAI
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
try:
    # optional dependency provided in original code base
    from src.utils import utils

    utils.setup_logger()
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------------------------------------------------------------
# Global resources (models, tokenizer, device, OpenAI client …)
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- NLI model (entailment probabilities) --------------
NLI_MODEL_NAME = "microsoft/deberta-v2-xlarge-mnli"
with tqdm(total=2, desc="Loading NLI model") as bar:
    entailment_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_NAME
    ).to(device).eval()
    bar.update(1)
    entailment_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    bar.update(1)

# ---------------- Sentence-transformer model (embeddings) ------------
SENTENCE_EMB_MODEL_NAME = "tomaarsen/static-similarity-mrl-multilingual-v1"
with tqdm(total=1, desc="Loading sentence-embedding model") as bar:
    sentence_embedding_model = SentenceTransformer(SENTENCE_EMB_MODEL_NAME)
    bar.update(1)

# ---------------- OpenAI client --------------------------------------
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", ""),
    api_key=os.getenv("OPENAI_API_KEY", ""),
)

def page_rank(
    adjacency_matrix: np.ndarray,
    damping_factor: float = 0.85,
    max_iterations: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    n = adjacency_matrix.shape[0]
    out_degree = np.array(adjacency_matrix.sum(axis=0)).flatten()
    out_degree[out_degree == 0] = 1  # avoid division by zero

    P = csr_matrix(adjacency_matrix / out_degree)  # transition matrix

    rank = np.full(n, 1.0 / n)
    for _ in range(max_iterations):
        new_rank = (1 - damping_factor) / n + damping_factor * P.T @ rank
        if np.linalg.norm(new_rank - rank, 1) < tol:
            break
        rank = new_rank

    return rank


def is_connected(adjacency_matrix: np.ndarray) -> Tuple[bool, List[List[int]]]:

    n = adjacency_matrix.shape[0]
    parent = list(range(n))
    rank = [0] * n

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1

    for u, v in np.argwhere(adjacency_matrix > 0):
        union(int(u), int(v))

    roots = [find(i) for i in range(n)]
    comp_dict: Dict[int, List[int]] = {}
    for idx, r in enumerate(roots):
        comp_dict.setdefault(r, []).append(idx)

    components = list(comp_dict.values())
    return len(components) == 1, components


def make_connected(
    adjacency_matrix: np.ndarray, responses: List[str]
) -> np.ndarray:
    
    connected, components = is_connected(adjacency_matrix)
    if connected:
        return adjacency_matrix

    # --- representative per component (highest PageRank) -------------
    representatives: List[int] = []
    for comp in components:
        pr = page_rank(adjacency_matrix[np.ix_(comp, comp)])
        representatives.append(comp[int(np.argmax(pr))])

    # --- compute edge weights (entailment probabilities) -------------
    rep_pairs = list(itertools.combinations(representatives, 2))
    entail_pairs = [(responses[i], responses[j]) for i, j in rep_pairs]
    weights = run_textual_entailment(entail_pairs)  # list of [entail, neutral, contra]

    # Build heap of (weight, u, v) – negative weight for *maximum* weight
    #   because we later pick *smallest* items; we want highest entailment.
    heap: List[Tuple[float, int, int]] = []
    for (u, v), w in zip(rep_pairs, weights):
        heapq.heappush(heap, (-w[0], u, v))  # negative = maximum entailment first

    # --- Kruskal ------------------------------------------------------
    parent = list(range(adjacency_matrix.shape[0]))
    rank = [0] * len(parent)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        return True

    while heap:
        w, u, v = heapq.heappop(heap)
        if union(u, v):
            adjacency_matrix[u, v] = adjacency_matrix[v, u] = -w  # revert sign
            logging.info(
                f"Inserted edge ({u}, {v}) with weight { -w :.4f} to make graph connected"
            )

    return adjacency_matrix



def run_textual_entailment(
    batch_premise_hypothesis: List[Tuple[str, str]]
) -> List[List[float]]:
    """
    Compute entailment class probabilities for *pairs* of sentences.

    Returns
    -------
    probs : List[[entail, neutral, contradiction]]
    """
    model_inputs = entailment_tokenizer.batch_encode_plus(
        batch_premise_hypothesis,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = entailment_model(**model_inputs).logits  # [B, 3]

    probs = torch.softmax(logits, dim=1).cpu().tolist()
    # reorder to [entail, neutral, contradiction]
    return [[p[2], p[1], p[0]] for p in probs]



def build_prompt(question: str, responses: List[str]) -> str:
    """Build a rewriting prompt in English."""
    responses_section = "\n".join(f"{i+1}. {r}" for i, r in enumerate(responses))
    prompt = f"""
Task Description:
Given a question and a raw response set R, rewrite **each** response so that it
is explicit, self-contained and comprehensive.  Provide exactly
{len(responses)} rewritten responses.

Rules:
1. Finish incomplete sentences.
2. Replace vague pronouns with concrete references.
3. Expand or clarify under-developed points relevant to the question.
4. If a response is empty, return "I don't know." for that item.
5. Return ONLY the enhanced responses, one per line, prefixed with
   the original index (e.g. "1. ...").  No extra commentary.

Example:
Question: How many students became heroes?
R =
1. Andrew Willis, Chris Willis, Reece Galea.
2. Andrew and his two friends helped Nicholle Price.
3. These three.
4. I don't know.
Augmented Responses:
1. Andrew Willis, Chris Willis and Reece Galea helped Nicholle Price and
   became heroes.
2. Andrew Willis and his two friends helped Nicholle Price and became
   heroes.
3. The three students became heroes.
4. I don't know.

Execution:
Question: {question}
R =
{responses_section}
Augmented Responses:
""".strip()
    return prompt


def parse_enhanced_responses(
    enhanced_responses: str, expected_count: int
) -> List[str]:
    """Extract the list of rewritten answers from the LLM output."""
    lines = [l.strip() for l in enhanced_responses.strip().split("\n") if l.strip()]
    cleaned: List[str] = []
    for line in lines:
        if line.startswith("Rewritten Answers:"):
            continue
        match = re.match(r"^\s*(\d+)\.\s+(.*)", line)
        if match:
            cleaned.append(match.group(2).strip())

    if len(cleaned) != expected_count:
        raise ValueError(
            f"Expected {expected_count} items, received {len(cleaned)} : {lines}"
        )
    return cleaned


def enhancing_answers(responses: List[str], question: str) -> List[str]:
    """
    Rewrite every response so that it is explicit and self-contained.
    Retries up to `max_retries` times if parsing fails.
    """
    logging.info("Enhancing answers …")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            prompt = build_prompt(question, responses)

            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            enhanced_text = chat_completion.choices[0].message.content
            return parse_enhanced_responses(enhanced_text, len(responses))
        except Exception as exc:
            logging.warning("Enhancing attempt %d/%d failed: %s", attempt, max_retries, exc)

    logging.error("Enhancing failed – falling back to original responses.")
    return responses



def compute_sentence_transformer_similirities(
    strings_list: List[str],
) -> np.ndarray:
    """
    Compute cosine similarity matrix with SentenceTransformers embeddings.
    """
    embeddings = sentence_embedding_model.encode(
        strings_list, convert_to_numpy=True, normalize_embeddings=True
    )  # shape [n, d], already L2-normalised
    return embeddings @ embeddings.T  # cosine because of normalisation


def compute_entailment_scores(
    strings_list: List[str], batch_size: int = 32
) -> np.ndarray:
    """
    Produce a symmetric matrix of pair-wise entailment probabilities
    (class 'entail' from NLI model).
    """
    n = len(strings_list)
    scores = np.zeros((n, n), dtype=np.float32)

    pairs = list(itertools.combinations(range(n), 2))
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        sent_pairs = [(strings_list[a], strings_list[b]) for a, b in batch_pairs]
        batch_probs = run_textual_entailment(sent_pairs)
        for (a, b), prob in zip(batch_pairs, batch_probs):
            scores[a, b] = scores[b, a] = prob[0]  # entailment prob

    return scores


def get_semantic_clusters(
    strings_list: List[str],
    question: str,
    similarity_threshold: float = 0.5,
    w_entail: float = 0.5,
) -> Tuple[List[int], List[str]]:
    """
    Cluster responses with a hybrid similarity measure.

    Returns
    -------
    clusters
        List of cluster id per response.
    enhanced_strings
        The rewritten responses fed into clustering.
    """
    if not strings_list:
        raise ValueError("strings_list cannot be empty")

    enhanced = enhancing_answers(strings_list, question)

    # Hybrid similarity matrix
    cos_sim = compute_sentence_transformer_similirities(enhanced)
    entail_sim = compute_entailment_scores(enhanced)
    sim = np.clip(w_entail * entail_sim + (1 - w_entail) * cos_sim, 0.0, 1.0)

    # Convert to distance
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    non_diag = dist[~np.eye(len(dist), dtype=bool)]
    if np.all(non_diag == 0.0):
        return [0] * len(enhanced), enhanced
    if np.all(non_diag == 1.0):
        return list(range(len(enhanced))), enhanced

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - similarity_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clusterer.fit_predict(dist)
    uniq, counts = np.unique(labels, return_counts=True)
    logging.info("Clustering: %d clusters, sizes %s", len(uniq), counts.tolist())

    return labels.tolist(), enhanced



def convert_data_to_serializable(data: Dict[str, Any]) -> Dict[str, Any]:
    """Strip large / non-serialisable fields before JSON dump."""
    for _, example in data.items():
        mla = example.get("most_likely_answer")
        if isinstance(mla, dict):
            mla.pop("token_log_likelihoods", None)
            mla.pop("embedding", None)
        if isinstance(example.get("responses"), list):
            example["responses"] = [
                r[0].replace("</s>", "") if isinstance(r, tuple) else r.replace("</s>", "")
                for r in example["responses"]
            ]
    return data


def convert_formats(generations_path: str, json_path: str) -> None:
    """Convert pickle generations file to pretty-printed UTF-8 JSON."""
    with open(generations_path, "rb") as f:
        data = pickle.load(f)
    data = convert_data_to_serializable(data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info("Converted data saved to %s", json_path)



def build_semantic_graph(
    responses: List[str], question: str, batch_size: int = 128
) -> List[List[float]]:

    cluster_ids, enhanced = get_semantic_clusters(responses, question)
    logging.info("Cluster ids: %s", cluster_ids)

    n = len(enhanced)
    adj = np.zeros((n, n), dtype=np.float32)

    # Only compute edges inside the *same* cluster
    pairs = [
        (i, j)
        for i, j in itertools.combinations(range(n), 2)
        if cluster_ids[i] == cluster_ids[j]
    ]
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        sent_pairs = [(enhanced[a], enhanced[b]) for a, b in batch_pairs]
        batch_probs = run_textual_entailment(sent_pairs)
        for (a, b), prob in zip(batch_pairs, batch_probs):
            adj[a, b] = adj[b, a] = prob[0]

    adj = make_connected(adj, enhanced)
    return adj.tolist()


