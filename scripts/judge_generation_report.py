"""Evaluation indépendante des réponses RAG à partir des preuves récupérées."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field


class JudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["pass", "fail"]
    score: int = Field(ge=0, le=4)
    grounded: bool
    relevant: bool
    citations_correct: bool
    abstention_appropriate: bool
    reason: str = Field(min_length=1, max_length=800)


SYSTEM = """Tu es un évaluateur indépendant et strict d'un assistant de réception hôtelière.
Tu dois juger uniquement avec les preuves fournies, sans utiliser tes connaissances externes.
Une réponse affirmative doit être entièrement soutenue par les extraits et ses citations [n]
doivent viser les bons extraits. Une abstention est correcte si la question est non répondable
ou si les preuves sont réellement insuffisantes. Signale toute information inventée.

Barème: 4 = correct, complet et bien cité; 3 = correct avec omission mineure;
2 = partiellement soutenu; 1 = erreur importante; 0 = dangereux ou inventé.
Le verdict vaut pass seulement si grounded, relevant, citations_correct et
abstention_appropriate sont vrais, avec un score d'au moins 3.
Retourne uniquement le JSON demandé."""


def _prompt(item: dict) -> str:
    evidence = [
        {
            "index": row.get("index"),
            "source_id": row.get("source_id"),
            "domain": row.get("domain"),
            "title": row.get("title"),
            "content": row.get("content"),
        }
        for row in item.get("evidence", [])
    ]
    payload = {
        "question": item.get("question"),
        "answer": item.get("answer"),
        "expected_test_constraints": item.get("expected", {}),
        "evidence": evidence,
    }
    return (
        f"CAS A EVALUER:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "SCHEMA JSON:\n"
        f"{json.dumps(JudgeDecision.model_json_schema(), ensure_ascii=False)}"
    )


def _extract_json(value: str) -> dict:
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Le juge n'a pas retourné de JSON")
    return json.loads(value[start : end + 1])


def _judge(
    client: OpenAI,
    model: str,
    item: dict,
    *,
    provider: str = "local_vllm",
) -> tuple[JudgeDecision, dict]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            deepseek_max_tokens = 8192 if attempt == 0 else 4096
            request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _prompt(item)},
                ],
                # En mode raisonnement, ce budget inclut aussi les tokens de
                # reflexion. 900 pouvait epuiser le budget avant le JSON final.
                "max_tokens": (
                    deepseek_max_tokens if provider == "deepseek" else 900
                ),
            }
            if provider == "deepseek":
                request["response_format"] = {"type": "json_object"}
                if attempt < 2:
                    request.update(
                        reasoning_effort="high" if attempt == 0 else "low",
                        extra_body={"thinking": {"type": "enabled"}},
                    )
                else:
                    request.update(
                        temperature=0.0,
                        extra_body={"thinking": {"type": "disabled"}},
                    )
            else:
                request.update(
                    temperature=0.0,
                    extra_body={
                        "guided_json": JudgeDecision.model_json_schema(),
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            completion = client.chat.completions.create(**request)
            raw = completion.choices[0].message.content or ""
            usage = completion.usage.model_dump() if completion.usage else {}
            return JudgeDecision.model_validate(_extract_json(raw)), usage
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"Réponse invalide du juge: {last_error}")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Nom de la variable d'environnement contenant la cle API",
    )
    parser.add_argument(
        "--provider", choices=("local_vllm", "deepseek"), default="local_vllm"
    )
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    items = source.get("results") or source.get("cases") or []
    api_key = os.getenv(args.api_key_env, "") if args.api_key_env else "EMPTY"
    if args.api_key_env and not api_key:
        parser.error(f"Variable d'environnement absente: {args.api_key_env}")
    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=300.0,
        max_retries=5,
    )
    existing = {}
    if args.output.exists():
        try:
            previous = json.loads(args.output.read_text(encoding="utf-8"))
            existing = {
                row.get("id"): row
                for row in previous.get("results", [])
                if row.get("id") and row.get("judge")
            }
        except (OSError, json.JSONDecodeError):
            existing = {}

    rows = []
    for index, item in enumerate(items, 1):
        if item.get("id") in existing:
            print(f"JUGE {index}/{len(items)}: {item.get('id')} (deja fait)", flush=True)
            row = existing[item["id"]]
        else:
            print(f"JUGE {index}/{len(items)}: {item.get('id')}", flush=True)
            try:
                decision, usage = _judge(
                    client, args.model, item, provider=args.provider
                )
                row = {
                    "id": item.get("id"),
                    "automatic_pass": item.get("automatic_pass"),
                    "judge": decision.model_dump(),
                    "usage": usage,
                }
            except (OSError, RuntimeError, OpenAIError) as exc:
                row = {
                    "id": item.get("id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        rows.append(row)
        judged = [row for row in rows if row.get("judge")]
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "candidate_model": source.get("model"),
            "judge_model": args.model,
            "judge_provider": args.provider,
            "source_report": str(args.input),
            "summary": {
                "expected": len(items),
                "completed": len(judged),
                "errors": sum("error" in row for row in rows),
                "passed": sum(row["judge"]["verdict"] == "pass" for row in judged),
                "mean_score": round(
                    sum(row["judge"]["score"] for row in judged) / len(judged), 3
                )
                if judged
                else 0.0,
                "prompt_tokens": sum(
                    int(row.get("usage", {}).get("prompt_tokens") or 0)
                    for row in judged
                ),
                "completion_tokens": sum(
                    int(row.get("usage", {}).get("completion_tokens") or 0)
                    for row in judged
                ),
            },
            "results": rows,
        }
        _write(args.output, report)

    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return 0 if report["summary"]["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
