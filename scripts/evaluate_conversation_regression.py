"""Régression sur des questions observées avec l’interface Streamlit."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.config import path, settings
from src.generation import answer as generation
from src.retrieval import retriever, vector_store
from src.retrieval.query_planner import plan_query


def _source_identity(source_id: str) -> str:
    """Tolere le renommage neutre d'une meme source documentaire."""

    faq = re.fullmatch(r"(?:generic_software|software)-faq-(\d+)", source_id)
    if faq:
        return f"faq-{faq.group(1)}"
    marker = "lexique-hotelier-"
    return source_id[source_id.index(marker) :] if marker in source_id else source_id


def _expected_source_found(expected: set[str], actual: list[str] | set[str]) -> bool:
    expected_ids = {_source_identity(value) for value in expected}
    actual_ids = {_source_identity(value) for value in actual}
    return bool(expected_ids & actual_ids)


def _normalized_text(value: str) -> str:
    """Rend les contrôles robustes aux guillemets, accents et ponctuations."""

    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()


def _load_cases(dataset: Path | None = None) -> list[dict]:
    dataset = dataset or path("evaluation/conversation_regression.jsonl")
    dataset = dataset if dataset.is_absolute() else path(dataset.as_posix())
    cases = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [case.get("id") for case in cases]
    if any(not case_id for case_id in ids):
        raise ValueError(f"Chaque cas doit avoir un id: {dataset}")
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"IDs dupliques dans {dataset}: {', '.join(duplicates)}")
    return cases


def _missing_source_groups(
    groups: list[list[str]], actual: list[str] | set[str]
) -> list[list[str]]:
    """Chaque groupe exige au moins une source equivalente dans les resultats."""

    return [
        group
        for group in groups
        if group and not _expected_source_found(set(group), actual)
    ]


def _history(case: dict) -> list[dict[str, str]]:
    previous = case.get("prior_user_question")
    return [{"role": "user", "content": previous}] if previous else []


def _retrieval_result(case: dict) -> dict:
    plan = plan_query(case["question"], history=_history(case))
    search_fn = (
        retriever.search_balanced if plan.route == "both" else retriever.search
    )
    search_kwargs = {
        "top_k": 5,
        "hybrid": False,
        "max_chunks_per_source": plan.max_chunks_per_source,
        "exclude_subtopics": plan.excluded_subtopics,
    }
    if plan.route != "both":
        search_kwargs["domain"] = plan.domain_filter
    hits = search_fn(plan.retrieval_query, **search_kwargs)
    source_ids = [hit.document.source_id for hit in hits]
    domains = sorted({hit.document.domain for hit in hits})
    expected_sources = set(case.get("expected_sources", []))
    required_source_groups = case.get("required_source_groups", [])
    expected_domains = set(case.get("expected_domains", []))
    failures = []
    if plan.route != case["expected_route"]:
        failures.append(f"route={plan.route}, attendu={case['expected_route']}")
    if (
        case["answerable"]
        and expected_sources
        and not _expected_source_found(expected_sources, source_ids)
    ):
        failures.append("aucune source attendue dans le top 5")
    missing_groups = _missing_source_groups(required_source_groups, source_ids)
    if case["answerable"] and missing_groups:
        failures.append(f"groupes de sources absents du top 5 : {missing_groups}")
    if case["answerable"] and not expected_domains.issubset(domains):
        failures.append("domaines attendus absents du top 5")
    return {
        "id": case["id"],
        "question": case["question"],
        "route": plan.route,
        "used_history": plan.used_history,
        "retrieval_query": plan.retrieval_query,
        "sensitive": plan.sensitive,
        "explicit_support_required": plan.explicit_support_required,
        "procedural_detail": plan.procedural_detail,
        "max_chunks_per_source": plan.max_chunks_per_source,
        "excluded_subtopics": list(plan.excluded_subtopics),
        "automatic_pass": not failures,
        "failures": failures,
        "hits": [
            {
                "rank": rank,
                "score": hit.score,
                "source_id": hit.document.source_id,
                "domain": hit.document.domain,
                "source_type": hit.document.source_type,
                "title": hit.document.title,
            }
            for rank, hit in enumerate(hits, 1)
        ],
    }


def _generation_result(case: dict) -> dict:
    started = time.perf_counter()
    answer = generation.ask(case["question"], top_k=5, history=_history(case))
    cited = answer.cited_hits()
    cited_sources = {hit.document.source_id for hit in cited}
    cited_domains = {hit.document.domain for hit in cited}
    expected_sources = set(case.get("expected_sources", []))
    required_source_groups = case.get("required_source_groups", [])
    expected_domains = set(case.get("expected_domains", []))
    forbidden_sources = set(case.get("forbidden_sources", []))
    failures = []
    if case["answerable"]:
        if answer.abstained:
            failures.append("abstention sur une question répondable")
        if not answer.citations_valid or not cited:
            failures.append("citations absentes ou invalides")
        if expected_sources and not _expected_source_found(
            expected_sources, cited_sources
        ):
            failures.append("aucune source attendue citée")
        missing_groups = _missing_source_groups(required_source_groups, cited_sources)
        if missing_groups:
            failures.append(f"groupes de sources non cites : {missing_groups}")
        if not expected_domains.issubset(cited_domains):
            failures.append("domaines attendus non cités")
        forbidden_identities = {
            _source_identity(source_id) for source_id in forbidden_sources
        }
        unexpected = {
            source_id
            for source_id in cited_sources
            if _source_identity(source_id) in forbidden_identities
        }
        if unexpected:
            failures.append(
                "sources interdites citées : " + ", ".join(sorted(unexpected))
            )
    elif not answer.abstained:
        failures.append("réponse servie alors que les preuves sont insuffisantes")
    normalized_answer = _normalized_text(answer.text)
    for phrase in case.get("required_phrases", []):
        if _normalized_text(phrase) not in normalized_answer:
            failures.append(f"formulation attendue absente : {phrase}")
    for phrase in case.get("forbidden_phrases", []):
        if _normalized_text(phrase) in normalized_answer:
            failures.append(f"formulation interne exposée : {phrase}")
    return {
        "id": case["id"],
        "question": case["question"],
        "route": answer.route,
        "used_history": answer.used_history,
        "answer": answer.text,
        "abstained": answer.abstained,
        "citations_valid": answer.citations_valid,
        "verification_decision": answer.verification_decision,
        "expected": {
            "answerable": bool(case["answerable"]),
            "expected_sources": sorted(expected_sources),
            "required_source_groups": required_source_groups,
            "expected_domains": sorted(expected_domains),
            "forbidden_sources": sorted(forbidden_sources),
            "required_phrases": list(case.get("required_phrases", [])),
            "forbidden_phrases": list(case.get("forbidden_phrases", [])),
            "reference_facts": list(case.get("reference_facts", [])),
        },
        "cited_sources": sorted(cited_sources),
        "cited_domains": sorted(cited_domains),
        # Le juge independant doit voir les preuves exactes, pas uniquement les
        # identifiants de citation. Cet export reste interne au projet.
        "evidence": [
            {
                "index": index,
                "cited": hit.document.source_id in cited_sources,
                "source_id": hit.document.source_id,
                "domain": hit.document.domain,
                "source_type": hit.document.source_type,
                "title": hit.document.title,
                "citation": hit.document.citation(),
                "content": hit.document.content,
            }
            for index, hit in enumerate(answer.hits, 1)
        ],
        "automatic_pass": not failures,
        "failures": failures,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/conversation_regression.jsonl"),
        help="Jeu JSONL a evaluer (relatif a la racine du projet ou absolu)",
    )
    args = parser.parse_args()
    cases = _load_cases(args.dataset)
    if args.ids:
        selected = set(args.ids)
        cases = [case for case in cases if case["id"] in selected]
    if args.limit:
        cases = cases[: args.limit]

    started = datetime.now(timezone.utc)
    mode = "retrieval" if args.retrieval_only else "generation"
    results = []
    try:
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
            result = (
                _retrieval_result(case)
                if args.retrieval_only
                else _generation_result(case)
            )
            results.append(result)
            print("PASS" if result["automatic_pass"] else "FAIL", result["failures"], flush=True)
    finally:
        vector_store.close()
        if not args.retrieval_only:
            generation.unload()

    cfg = settings()
    report = {
        "evaluation": "conversation_regression",
        "dataset": str(args.dataset),
        "mode": mode,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "model": cfg["hw"]["llm_model"] if not args.retrieval_only else cfg["embeddings"]["model"],
        "passed": sum(result["automatic_pass"] for result in results),
        "total": len(results),
        "all_passed": all(result["automatic_pass"] for result in results),
        "results": results,
    }
    if args.output:
        output = args.output if args.output.is_absolute() else path(args.output.as_posix())
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = path("data/exports/conversation-evaluations")
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = started.strftime("%Y%m%dT%H%M%SZ")
        output = output_dir / f"{stamp}-{mode}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT {output}", flush=True)
    print(f"RESULT {report['passed']}/{report['total']}", flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
