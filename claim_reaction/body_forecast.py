"""Retrospective article-body evidence forecasts for held-out markets.

Body fetches can postdate the provisional GDELT event time. Their fetch times
are therefore recorded and the report remains explicitly retrospective.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .forecast import ForecastRow, add_relevance_features, evaluate_leave_one_market_out
from .retrieval import HuggingFaceEmbedder

_SCORER_VERSION = "entailment-only-v1"


class LocalNLIClassifier:
    """Local three-way NLI classifier used for signed market evidence."""
    def __init__(self, model_id: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli") -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for local NLI") from exc
        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def probabilities(self, premise: str, hypothesis: str) -> dict[str, float]:
        tokens = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**tokens).logits[0], dim=-1).cpu().tolist()
        labels = {str(k).lower(): int(v) for k, v in self.model.config.label2id.items()}
        def get(label: str, fallback: int) -> float:
            idx = next((i for name, i in labels.items() if label in name), fallback)
            return float(probs[idx])
        return {"entailment": get("entail", 0), "neutral": get("neutral", 1),
                "contradiction": get("contrad", 2)}


def extract_verbatim_sentences(body: str, min_characters: int = 20,
                               max_characters: int = 500, max_sentences: int = 200) -> list[str]:
    """Split text at sentence punctuation while preserving decimals and abbreviations."""
    if min_characters <= 0 or max_characters < min_characters or max_sentences <= 0:
        raise ValueError("invalid sentence extraction limits")
    abbreviations = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs",
                     "etc", "e.g", "i.e", "u.s", "u.k", "u.n", "u.s.a", "u.k.a",
                     "inc", "ltd", "co", "no", "fig", "approx", "dept"}
    boundaries: list[int] = []
    i = 0
    while i < len(body):
        char = body[i]
        if char == "\n":
            boundaries.append(i + 1)
            i += 1
            continue
        if char in "!?。！？":
            while i + 1 < len(body) and body[i + 1] in "!?。！？":
                i += 1
            if i + 1 == len(body) or body[i + 1].isspace():
                boundaries.append(i + 1)
            i += 1
            continue
        if char == "." and not (i > 0 and i + 1 < len(body)
                                and body[i - 1].isdigit() and body[i + 1].isdigit()):
            if i + 1 < len(body) and not body[i + 1].isspace():
                i += 1
                continue
            prefix = body[:i]
            word_match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$", prefix)
            token = word_match.group(1).lower() if word_match else ""
            initialism = bool(re.fullmatch(r"(?:[a-z]\.)+[a-z]?", token))
            if token not in abbreviations and not initialism:
                boundaries.append(i + 1)
        i += 1

    found: list[str] = []
    seen: set[str] = set()
    start = 0
    for end in [*boundaries, len(body)]:
        sentence = body[start:end].strip()
        start = end
        key = sentence.casefold()
        if not sentence:
            continue
        if len(sentence) < min_characters or len(sentence) > max_characters or key in seen:
            continue
        seen.add(key)
        found.append(sentence)
        if len(found) >= max_sentences:
            break
    return found


def signed_entailment_score(probabilities: dict[str, Any]) -> float:
    """Return signed evidence from paired YES/NO NLI probability mappings."""
    yes_p, no_p = probabilities["yes"], probabilities["no"]
    # Require a clear entailment for one direction. Ambiguous language often
    # contradicts both declarative outcomes, in which case the evidence is zero.
    y, n = yes_p["entailment"], no_p["entailment"]
    yes_entails = y > yes_p["neutral"] and y > yes_p["contradiction"]
    no_entails = n > no_p["neutral"] and n > no_p["contradiction"]
    if yes_entails and no_entails:
        # When both hypotheses are judged entailed, use the stronger score;
        # exact ties are uninformative.
        score = y if y > n else (-n if n > y else 0.0)
    elif yes_entails:
        score = y
    elif no_entails:
        score = -n
    else:
        score = 0.0
    return float(max(-1.0, min(1.0, score)))


def _score_pair(classifier: Any, sentence: str, yes: str, no: str) -> tuple[float, dict[str, Any]]:
    raw = {"yes": classifier.probabilities(sentence, yes),
           "no": classifier.probabilities(sentence, no)}
    return signed_entailment_score(raw), raw


def _read_documents(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    documents: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid documents JSONL at line {line_number}") from exc
            slug = row.get("market_slug")
            url = row.get("article_url") or row.get("url")
            if slug and url:
                documents[(str(slug), str(url))] = row
    return documents


def select_usable_body_rows(rows: list[ForecastRow], documents_path: str | Path
                            ) -> tuple[list[ForecastRow], dict[str, Any]]:
    """Select labelled rows using only successful sample-body availability.

    The outcome is never inspected for selection. A selected document must
    match both market slug and article URL, have status ``fetched``, and carry
    a nonempty body.
    """
    documents = _read_documents(documents_path)
    usable_keys = {
        key for key, document in documents.items()
        if document.get("body_fetch_status", document.get("status")) in
        ("fetched", "ok", "success", "complete")
        and isinstance(document.get("body"), str) and document["body"].strip()
    }
    selected = [row for row in rows if (row.market_slug, row.article_url) in usable_keys]
    rows_by_key: dict[tuple[str, str], list[ForecastRow]] = {}
    for row in rows:
        rows_by_key.setdefault((row.market_slug, row.article_url), []).append(row)
    successful_keys = usable_keys & rows_by_key.keys()
    failed_keys = set(documents) - usable_keys
    coverage = {
        "sample_document_pairs": len(documents),
        "successful_body_pairs": len(usable_keys),
        "failed_or_empty_body_pairs": len(failed_keys),
        "successful_pairs_with_forecast_rows": len(successful_keys),
        "selected_forecast_rows": len(selected),
        "forecast_rows_without_successful_sample_body": len(rows) - len(selected),
        "forecast_pairs_absent_from_sample": len(set(rows_by_key) - set(documents)),
        "failed_or_empty_pairs_with_forecast_rows": len(failed_keys & rows_by_key.keys()),
    }
    counts = {horizon: len({row.market_slug for row in selected
                            if row.horizon_minutes == horizon})
              for horizon in sorted({row.horizon_minutes for row in rows})}
    # Nested alpha selection in evaluate_leave_one_market_out needs two
    # training markets after the outer holdout, so three total markets are the
    # minimum for any evaluable horizon. Check before loading local models.
    insufficient = {horizon: count for horizon, count in counts.items() if count < 3}
    if insufficient:
        raise ValueError("Fewer than three markets have successfully fetched sample bodies "
                         f"for horizon(s): {insufficient}")
    coverage["usable_markets_by_horizon"] = {str(key): value for key, value in counts.items()}
    return selected, coverage


def _load_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {"version": 3, "sentences": {}, "rankings": {}, "scores": {}}
    value = json.loads(cache_path.read_text(encoding="utf-8"))
    if value.get("version") != 3 or not all(isinstance(value.get(key), dict)
                                              for key in ("sentences", "rankings", "scores")):
        raise ValueError("Unsupported or malformed body forecast cache")
    return value


def _save_cache(path: str | Path, cache: dict[str, Any]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=cache_path.parent,
                                         prefix=f".{cache_path.name}.", suffix=".tmp",
                                         delete=False) as output:
            temporary_path = output.name
            output.write(json.dumps(cache, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _cache_key(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def add_body_features(rows: list[ForecastRow], documents_path: str | Path,
                      embedder: Any, classifier: Any, hypotheses_path: str | Path,
                      cache_path: str | Path, min_detail_characters: int = 20,
                      max_sentences_per_article: int = 3
                      ) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Join bodies, rank exact sentences with E5, and score with signed NLI."""
    if min_detail_characters <= 0:
        raise ValueError("minimum detail length must be positive")
    docs = _read_documents(documents_path)
    if max_sentences_per_article <= 0:
        raise ValueError("max sentences per article must be positive")
    hypotheses = json.loads(Path(hypotheses_path).read_text(encoding="utf-8"))
    if not isinstance(hypotheses, dict):
        raise ValueError("hypotheses JSON must be a market-slug mapping")
    cache = _load_cache(cache_path)
    embedding_id = str(getattr(embedder, "model_id", type(embedder).__name__))
    classifier_id = str(getattr(classifier, "model_id", type(classifier).__name__))
    config_key = json.dumps({"min_chars": min_detail_characters,
                             "max_chars": 500, "top_k": max_sentences_per_article}, sort_keys=True)
    feature_rows: list[list[float]] = []
    provenance: list[dict[str, Any]] = []

    for row in rows:
        document = docs.get((row.market_slug, row.article_url))
        body = document.get("body") if document else None
        status = (document.get("body_fetch_status", document.get("status", "missing"))
                  if document else "missing")
        body_fetched_at = document.get("body_fetched_at") if document else None
        body_retrospective = document.get("body_retrospective") if document else None
        # Keep an unavailable body at score zero while retaining its match state.
        if not isinstance(body, str) or not body.strip() or status not in ("ok", "success", "fetched", "complete"):
            feature_rows.append([row.pre_price, 0.0])
            provenance.append({"claim_id": row.claim_id, "market_slug": row.market_slug,
                               "article_url": row.article_url, "model": classifier_id,
                               "embedding_model": embedding_id,
                               "status": status, "body_fetched_at": body_fetched_at,
                               "body_retrospective": body_retrospective,
                               "detail_count": 0, "signed_detail_scores": [],
                               "aggregate_signed_score": 0.0})
            continue

        body_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        market_hyp = hypotheses.get(row.market_slug)
        valid_hyp = (isinstance(market_hyp, dict) and isinstance(market_hyp.get("yes"), str)
                     and market_hyp["yes"].strip() and isinstance(market_hyp.get("no"), str)
                     and market_hyp["no"].strip())
        if not valid_hyp:
            feature_rows.append([row.pre_price, 0.0])
            provenance.append({"claim_id": row.claim_id, "market_slug": row.market_slug,
                               "article_url": row.article_url, "status": "missing_hypotheses",
                               "model": classifier_id, "detail_count": 0,
                               "aggregate_signed_score": 0.0, "body_sha256": body_digest})
            continue
        sentence_key = _cache_key(body_digest, config_key)
        if sentence_key not in cache["sentences"]:
            cache["sentences"][sentence_key] = extract_verbatim_sentences(body, min_detail_characters)
        sentences = cache["sentences"][sentence_key]
        rank_key = _cache_key(embedding_id, row.market_question, body_digest, config_key)
        if rank_key not in cache["rankings"]:
            qvec = embedder.encode([row.market_question], "query")[0] if sentences else []
            svecs = embedder.encode(sentences, "document") if sentences else []
            def cosine(a: list[float], b: list[float]) -> float:
                return sum(x*y for x,y in zip(a,b)) / max(1e-12, (sum(x*x for x in a)**.5)*(sum(y*y for y in b)**.5))
            ranked = sorted(((cosine(qvec, v), i) for i, v in enumerate(svecs)), reverse=True)
            cache["rankings"][rank_key] = [i for _, i in ranked[:max_sentences_per_article]]
        _save_cache(cache_path, cache)
        selected = [sentences[i] for i in cache["rankings"][rank_key]]
        scores: list[float] = []
        raw: list[dict[str, Any]] = []
        for sentence in selected:
            score_key = _cache_key(_SCORER_VERSION, classifier_id, config_key,
                                   market_hyp["yes"], market_hyp["no"], sentence)
            if score_key not in cache["scores"]:
                cache["scores"][score_key] = _score_pair(classifier, sentence,
                                                         market_hyp["yes"], market_hyp["no"])
            score, probs = cache["scores"][score_key]
            scores.append(float(score))
            raw.append({"sentence": sentence, "signed_score": float(score), "probabilities": probs})
        aggregate = float(sum(scores) / len(scores)) if scores else 0.0
        feature_rows.append([row.pre_price, aggregate])
        provenance.append({"claim_id": row.claim_id, "market_slug": row.market_slug,
                           "article_url": row.article_url, "model": classifier_id,
                           "embedding_model": embedding_id,
                           "status": status, "body_fetched_at": body_fetched_at,
                           "body_retrospective": body_retrospective,
                           "detail_count": len(selected), "selected_sentences": raw,
                           "signed_detail_scores": scores,
                           "aggregate_method": "arithmetic_mean_top_e5_sentences",
                           "aggregate_signed_score": aggregate,
                           "body_sha256": body_digest})
        # Persist each completed article so a later failure or interruption
        # retains all inference work completed before this point.
        _save_cache(cache_path, cache)

    # Also write the empty/fully unavailable case, ensuring the cache exists
    # and that future body results have the normal versioned structure.
    _save_cache(cache_path, cache)
    return np.asarray(feature_rows, dtype=float).reshape((len(rows), 2)), \
        ["pre_price", "body_mean_signed_nli_score"], provenance


def compare_body_and_headline(rows: list[ForecastRow], body_features: np.ndarray,
                              headline_features: np.ndarray,
                              alphas: tuple[float, ...] | list[float] = (0.01, 0.1, 1.0, 10.0, 100.0)
                              ) -> dict[str, Any]:
    """Compare body and headline feature models over exactly the same rows."""
    body = evaluate_leave_one_market_out(rows, body_features, alphas)
    headline = evaluate_leave_one_market_out(rows, headline_features, alphas)
    if not body["folds"] or not headline["folds"]:
        markets_by_horizon = {h: len({row.market_slug for row in rows
                                      if row.horizon_minutes == h})
                              for h in sorted({row.horizon_minutes for row in rows})}
        raise ValueError("The nested leave-one-market-out evaluator produced no folds; "
                         "at least three usable markets per horizon are needed. "
                         f"Available: {markets_by_horizon}")
    # Baselines are identical across both evaluations; expose them once from
    # the same folds and retain each model's per-fold predictions/MAE.
    return {"limitation": "Retrospective GDELT observations and fetched article bodies are not live-arriving evidence.",
            "rows": len(rows), "body_based": body, "headline_only": headline,
            "same_rows": True}


__all__ = ["LocalNLIClassifier", "extract_verbatim_sentences", "add_body_features", "compare_body_and_headline",
           "add_relevance_features", "select_usable_body_rows"]
