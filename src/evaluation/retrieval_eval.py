"""
Evaluation du retriever.

Regle centrale (section 54) : mesurer AVANT d'ajouter une technique.
Cette baseline sert de référence aux améliorations successives.

Dataset : evaluation/dataset.jsonl
  {"question": "...", "expected_sources": ["id1"], "category": "software"}
  Une question sans expected_sources est une question SANS REPONSE :
  le systeme doit s'abstenir.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import path
from ..chunking import pipeline as chunking
from ..retrieval import retriever
from ..schema import CanonicalDocument

QUESTION_TYPES = {
    "direct",
    "paraphrase",
    "procedurale",
    "case_pratique",
    "ambigue",
    "inconnue",
}


def load_dataset(file: str = "evaluation/dataset.jsonl") -> list[dict]:
    return [
        json.loads(line)
        for line in path(file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_dataset(
    dataset: list[dict[str, Any]] | None = None,
    documents: list[CanonicalDocument] | None = None,
) -> dict[str, Any]:
    """Valide le jeu d’évaluation sans lancer le retriever."""

    dataset = dataset if dataset is not None else load_dataset()
    documents = documents if documents is not None else chunking.load()
    source_domains: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        source_domains[document.source_id].add(document.domain)

    errors: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for index, item in enumerate(dataset, 1):
        question = str(item.get("question", "")).strip()
        category = item.get("category")
        expected = item.get("expected_sources")
        question_type = item.get("question_type")
        answerable = item.get("answerable")
        item_errors: list[str] = []
        if not question:
            item_errors.append("question_absente")
        elif question.casefold() in seen_questions:
            item_errors.append("question_dupliquee")
        seen_questions.add(question.casefold())
        if category not in {"software", "hotel"}:
            item_errors.append("categorie_invalide")
        if not isinstance(expected, list) or any(
            not isinstance(value, str) or not value for value in (expected or [])
        ):
            item_errors.append("expected_sources_invalides")
            expected = []
        if question_type not in QUESTION_TYPES:
            item_errors.append("question_type_invalide")
        if not isinstance(answerable, bool) or answerable != bool(expected):
            item_errors.append("answerable_incoherent")
        for source_id in expected or []:
            if source_id not in source_domains:
                item_errors.append(f"source_absente:{source_id}")
            elif category not in source_domains[source_id]:
                item_errors.append(f"source_mauvais_domaine:{source_id}")
        if item_errors:
            errors.append({"line": index, "question": question, "errors": item_errors})

    categories = Counter(item.get("category") for item in dataset)
    types = Counter(item.get("question_type") for item in dataset)
    answerable_count = sum(bool(item.get("expected_sources")) for item in dataset)
    balance_valid = categories == Counter({"software": 25, "hotel": 25})
    return {
        "questions": len(dataset),
        "by_category": dict(sorted(categories.items())),
        "by_question_type": dict(sorted(types.items())),
        "answerable": answerable_count,
        "unanswerable": len(dataset) - answerable_count,
        "unique_expected_sources": len(
            {
                source
                for item in dataset
                for source in item.get("expected_sources", [])
            }
        ),
        "errors": errors,
        "balanced_25_25": balance_valid,
        "ready_for_evaluation": len(dataset) == 50 and balance_valid and not errors,
    }


def evaluate(
    k: int = 5, hybrid: bool = False, file: str = "evaluation/dataset.jsonl"
) -> dict:
    dataset = load_dataset(file)
    per_cat = defaultdict(lambda: {"n": 0, "hits": 0, "recall": 0.0, "mrr": 0.0})
    failures = []
    query_results = []
    unanswerable_n = 0
    abstained = 0

    for item in dataset:
        expected = set(item.get("expected_sources", []))
        cat = item.get("category", "?")
        if not expected:
            hits = retriever.search(item["question"], top_k=k, hybrid=hybrid)
            unanswerable_n += 1
            if not hits:
                abstained += 1
            else:
                failures.append(
                    {
                        "question": item["question"],
                        "type": "false_answerable",
                        "attendu": [],
                        "trouve": [hit.document.source_id for hit in hits],
                    }
                )
            query_results.append(
                _query_result(item, hits, expected, rank=None, recall=0.0)
            )
            continue

        hits = retriever.search(item["question"], top_k=k, hybrid=hybrid)
        found = [h.document.source_id for h in hits]

        found_expected = expected.intersection(found)
        rank = next(
            (i for i, source in enumerate(found, 1) if source in expected), None
        )
        per_cat[cat]["n"] += 1
        per_cat[cat]["recall"] += len(found_expected) / len(expected)
        if rank:
            per_cat[cat]["hits"] += 1
            per_cat[cat]["mrr"] += 1 / rank
        else:
            failures.append(
                {
                    "question": item["question"],
                    "type": "miss",
                    "attendu": list(expected),
                    "trouve": found,
                }
            )
        query_results.append(
            _query_result(
                item,
                hits,
                expected,
                rank=rank,
                recall=len(found_expected) / len(expected),
            )
        )

    results = {}
    total_n = total_hits = 0
    total_recall = total_mrr = 0.0
    for cat, v in per_cat.items():
        results[cat] = {
            f"hit_rate@{k}": round(v["hits"] / v["n"], 3) if v["n"] else 0.0,
            f"recall@{k}": round(v["recall"] / v["n"], 3) if v["n"] else 0.0,
            "mrr": round(v["mrr"] / v["n"], 3) if v["n"] else 0.0,
            "n": v["n"],
        }
        total_n += v["n"]
        total_hits += v["hits"]
        total_recall += v["recall"]
        total_mrr += v["mrr"]

    results["GLOBAL"] = {
        f"hit_rate@{k}": round(total_hits / total_n, 3) if total_n else 0.0,
        f"recall@{k}": round(total_recall / total_n, 3) if total_n else 0.0,
        "mrr": round(total_mrr / total_n, 3) if total_n else 0.0,
        "n": total_n,
    }
    results["UNANSWERABLE"] = {
        "abstention_accuracy": round(abstained / unanswerable_n, 3)
        if unanswerable_n
        else None,
        "n": unanswerable_n,
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "retriever": "hybrid_rrf" if hybrid else "dense",
            "embedding_model": "BAAI/bge-m3",
            "k": k,
            "dataset": file,
        },
        "metrics": results,
        "failure_count": len(failures),
        "failures": failures,
        "queries": query_results,
    }


def _query_result(
    item: dict[str, Any],
    hits: list[Any],
    expected: set[str],
    *,
    rank: int | None,
    recall: float,
) -> dict[str, Any]:
    return {
        "question": item["question"],
        "category": item.get("category"),
        "question_type": item.get("question_type"),
        "answerable": bool(expected),
        "expected_sources": sorted(expected),
        "hit": rank is not None,
        "recall": recall,
        "reciprocal_rank": 1 / rank if rank else 0.0,
        "retrieved": [
            {
                "rank": index,
                "document_id": hit.document.id,
                "source_id": hit.document.source_id,
                "score": round(float(hit.score), 6),
                "title": hit.document.title,
            }
            for index, hit in enumerate(hits, 1)
        ],
    }


def save_run(result: dict[str, Any], output: str | None = None) -> Path:
    """Conserve une mesure complète et traçable dans l'espace de données."""

    if output:
        destination = path(output)
    else:
        stamp = datetime.fromisoformat(result["generated_at"]).strftime("%Y%m%dT%H%M%SZ")
        config = result["configuration"]
        destination = path(
            "data/exports/retrieval-evaluations/"
            f"{stamp}-{config['retriever']}-k{config['k']}.json"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination
