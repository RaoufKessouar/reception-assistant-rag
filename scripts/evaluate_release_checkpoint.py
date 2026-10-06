"""Évaluation bout en bout de la version de référence."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.config import path, settings
from src.generation import answer as generation
from src.retrieval import vector_store


CASES = [
    {
        "id": "software",
        "requirement": "logiciel de gestion hôtelière question -> answer + source",
        "question": "Comment passer un dossier en débiteur dans logiciel de gestion hôtelière ?",
        "expected_domains": {"software"},
    },
    {
        "id": "hotel",
        "requirement": "Hotel question -> answer + source",
        "question": "Un client veut libérer sa chambre à 14 h : combien doit-il payer ?",
        "expected_domains": {"hotel"},
        "expected_text": "50",
    },
    {
        "id": "mixed",
        "requirement": "Mixed question -> answer",
        "question": (
            "Le client n'est pas arrivé et ne répond pas : quand le déclarer "
            "no-show et comment le traiter dans logiciel de gestion hôtelière ?"
        ),
        "expected_domains": {"hotel", "software"},
    },
    {
        "id": "unknown",
        "requirement": "Unknown question -> abstain",
        "question": "L'hôtel accepte-t-il les paiements en Bitcoin ?",
        "expect_abstention": True,
    },
    {
        "id": "video",
        "requirement": "Video question -> answer + timestamp + screenshot",
        "question": (
            "Dans logiciel de gestion hôtelière, où puis-je modifier les dates, la prestation, "
            "la pré-attribution, l'occupation, les tarifs et les options "
            "d'une réservation ?"
        ),
        "expected_domains": {"software"},
        "require_video_evidence": True,
    },
]


def _existing(relative: str | None) -> bool:
    return bool(relative and path(relative).exists())


def _evaluate_case(case: dict) -> dict:
    print(f"START {case['id']}: {case['question']}", flush=True)
    started = time.perf_counter()
    answer = generation.ask(case["question"], top_k=5, hybrid=False)
    cited = answer.cited_hits()
    cited_domains = {hit.document.domain for hit in cited}
    cited_videos = [hit for hit in cited if hit.document.source_type == "video"]

    failures: list[str] = []
    if case.get("expect_abstention"):
        if not answer.abstained:
            failures.append("La question inconnue n'a pas déclenché l'abstention.")
    else:
        if answer.abstained:
            failures.append("Le système s'est abstenu sur une question répondable.")
        if not answer.citations_valid or not cited:
            failures.append("La réponse n'a pas de citation valide.")
        expected_domains = case.get("expected_domains", set())
        missing_domains = expected_domains - cited_domains
        if missing_domains:
            failures.append(
                "Domaine(s) non cité(s) : " + ", ".join(sorted(missing_domains))
            )
        expected_text = case.get("expected_text")
        if expected_text and expected_text not in answer.text:
            failures.append(f"Valeur attendue absente : {expected_text}")

    if case.get("require_video_evidence"):
        if not cited_videos:
            failures.append("Aucune source vidéo n'est citée.")
        elif not any(
            hit.document.timestamp_start is not None
            and any(
                _existing(value)
                for value in [hit.document.screenshot, *hit.document.screenshots]
            )
            for hit in cited_videos
        ):
            failures.append("La vidéo citée n'a pas de timestamp et capture disponibles.")

    hit_rows = []
    cited_ids = {hit.document.id for hit in cited}
    for rank, hit in enumerate(answer.hits, 1):
        doc = hit.document
        screenshots = [value for value in [doc.screenshot, *doc.screenshots] if value]
        hit_rows.append(
            {
                "rank": rank,
                "score": hit.score,
                "cited": doc.id in cited_ids,
                "document_id": doc.id,
                "source_id": doc.source_id,
                "domain": doc.domain,
                "source_type": doc.source_type,
                "authority": doc.authority,
                "status": doc.status,
                "citation": doc.citation(),
                "timestamp_start": doc.timestamp_start,
                "screenshot": doc.screenshot,
                "screenshot_exists": any(_existing(value) for value in screenshots),
                "source_path": doc.source_path,
                "source_exists": _existing(doc.source_path),
            }
        )

    result = {
        "id": case["id"],
        "requirement": case["requirement"],
        "question": case["question"],
        "answer": answer.text,
        "abstained": answer.abstained,
        "citations_valid": answer.citations_valid,
        "cited_domains": sorted(cited_domains),
        "automatic_pass": not failures,
        "failures": failures,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "hits": hit_rows,
    }
    print(
        f"DONE {case['id']}: {'PASS' if result['automatic_pass'] else 'FAIL'} "
        f"({result['duration_seconds']} s)",
        flush=True,
    )
    return result


def main() -> int:
    cfg = settings()
    started = datetime.now(timezone.utc)
    results = []
    try:
        for case in CASES:
            results.append(_evaluate_case(case))
    finally:
        vector_store.close()
        generation.unload()

    finished = datetime.now(timezone.utc)
    report = {
        "checkpoint": "reference-release",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "model": cfg["hw"]["llm_model"],
        "llm_backend": cfg["hw"]["llm_backend"],
        "retrieval": {
            "mode": "dense",
            "top_k": 5,
            "min_score": cfg["qdrant"].get("min_score"),
            "candidate_multiplier": cfg["qdrant"].get("candidate_multiplier"),
            "max_chunks_per_source": cfg["qdrant"].get("max_chunks_per_source"),
        },
        "passed": sum(result["automatic_pass"] for result in results),
        "total": len(results),
        "all_passed": all(result["automatic_pass"] for result in results),
        "cases": results,
    }

    output_dir = path("data/exports/release-checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    output = output_dir / f"{stamp}-release-checkpoint.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT {output}", flush=True)
    print(f"RESULT {report['passed']}/{report['total']}", flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
