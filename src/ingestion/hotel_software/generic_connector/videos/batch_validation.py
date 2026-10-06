"""Validation traçable des étapes vidéo à partir des paquets de revue.

Le contrôle par lot ne prétend pas que chaque étape a été relue séparément.
Il conserve explicitement la méthode ``batch_sampled`` et refuse d'appliquer
un lot incomplet, modifié après génération ou contenant une erreur majeure.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from .....config import path
from .....schema import ProceduralStep
from .common import atomic_write_json, read_json

CALIBRATED_FIELDS = {
    "action_type",
    "action_target",
    "action_location",
    "screen_before",
    "visual_description",
}
ALLOWED_DECISIONS = {"correct", "incorrect"}
MINOR_RESOLUTIONS = {"prefer_original_primary_evidence"}


def _read_jsonl(file: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _item_key(item: dict[str, Any]) -> tuple[str, int]:
    return str(item["video_id"]), int(item["step_number"])


def _unique_items(items: list[dict[str, Any]], label: str) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for item in items:
        key = _item_key(item)
        if key in indexed:
            raise ValueError(f"{label} contient un doublon : {key[0]} étape {key[1]}")
        indexed[key] = item
    return indexed


def validate_results(
    results_file: Path,
    manifest_file: Path,
    *,
    expected_review_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Vérifie l'identité, l'exhaustivité et les empreintes d'un export HTML."""

    results = read_json(results_file)
    manifest = read_json(manifest_file)
    if results.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        raise ValueError("Version de paquet de revue non prise en charge")
    if results.get("batch_id") != manifest.get("batch_id"):
        raise ValueError("Le résultat ne correspond pas au manifeste de revue")
    if results.get("review_type") != expected_review_type:
        raise ValueError(
            f"Type de revue attendu : {expected_review_type}, reçu : "
            f"{results.get('review_type')}"
        )
    if not str(results.get("reviewer") or "").strip():
        raise ValueError("Le reviewer est absent")
    if not results.get("reviewed_at"):
        raise ValueError("La date de revue est absente")

    expected = _unique_items(list(manifest.get("items", [])), "Le manifeste")
    actual = _unique_items(list(results.get("items", [])), "Le résultat")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"Revue incomplète ou étrangère : manquants={missing}, extras={extra}")

    for key, item in actual.items():
        reference = expected[key]
        if item.get("decision") not in ALLOWED_DECISIONS:
            raise ValueError(f"Décision invalide pour {key}")
        if item.get("proposal_hash") != reference.get("proposal_hash"):
            raise ValueError(f"Proposition modifiée après génération pour {key}")
        if item.get("risk") != reference.get("risk"):
            raise ValueError(f"Niveau de risque modifié pour {key}")
        if item.get("decision") == "incorrect" and not str(item.get("note") or "").strip():
            raise ValueError(f"Une correction sans remarque est présente pour {key}")
    return results, manifest


def _load_resolutions(
    file: Path,
    *,
    batch_id: str,
    incorrect_keys: set[tuple[str, int]],
) -> dict[tuple[str, int], dict[str, Any]]:
    payload = read_json(file)
    if payload.get("schema_version") != 1 or payload.get("batch_id") != batch_id:
        raise ValueError("Le fichier de résolution ne correspond pas au lot")
    resolutions = _unique_items(list(payload.get("resolutions", [])), "Les résolutions")
    if set(resolutions) != incorrect_keys:
        raise ValueError(
            "Chaque élément incorrect doit avoir exactement une résolution : "
            f"attendus={sorted(incorrect_keys)}, reçus={sorted(resolutions)}"
        )
    for key, resolution in resolutions.items():
        if resolution.get("severity") != "minor":
            raise ValueError(f"Une erreur non mineure interdit la validation par lot : {key}")
        if resolution.get("resolution") not in MINOR_RESOLUTIONS:
            raise ValueError(f"Résolution inconnue pour {key}")
    return resolutions


def _load_steps(videos_root: Path) -> tuple[
    dict[tuple[str, int], ProceduralStep], dict[str, tuple[Path, dict[str, Any]]]
]:
    steps: dict[tuple[str, int], ProceduralStep] = {}
    payloads: dict[str, tuple[Path, dict[str, Any]]] = {}
    for file in sorted(videos_root.glob("*/steps.json")):
        payload = read_json(file)
        payloads[str(payload["video_id"])] = (file, payload)
        for row in payload.get("steps", []):
            step = ProceduralStep(**row)
            steps[(step.video_id, step.step_number)] = step
    return steps, payloads


def _resolve_minor_issue(
    values: dict[str, Any],
    original: ProceduralStep,
    resolution: dict[str, Any],
) -> None:
    if resolution["resolution"] != "prefer_original_primary_evidence":
        raise ValueError(f"Résolution non gérée : {resolution['resolution']}")

    values["visual_description"] = original.visual_description
    values["screenshot_before"] = original.screenshot_before
    supporting = list(original.supporting_screenshots)
    if original.screenshot_after and original.screenshot_after not in supporting:
        supporting.append(original.screenshot_after)
    values["supporting_screenshots"] = supporting
    values["screenshot_after"] = None


def apply_sampled_batch(
    *,
    results_file: Path,
    manifest_file: Path,
    resolutions_file: Path,
    suggestions_file: Path | None = None,
    videos_root: Path | None = None,
    audit_output: Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Prépare ou applique la validation échantillonnée aux étapes non exceptionnelles."""

    suggestions_file = suggestions_file or path(
        "data/interim/video_calibration_suggestions.jsonl"
    )
    videos_root = videos_root or path("data/interim/videos")
    results, manifest = validate_results(
        results_file,
        manifest_file,
        expected_review_type="independent_stratified_sample",
    )
    result_items = _unique_items(list(results["items"]), "Le résultat")
    incorrect_keys = {
        key for key, item in result_items.items() if item["decision"] == "incorrect"
    }
    resolutions = _load_resolutions(
        resolutions_file,
        batch_id=str(results["batch_id"]),
        incorrect_keys=incorrect_keys,
    )

    suggestion_rows = _read_jsonl(suggestions_file)
    proposals = {
        _item_key(row): row
        for row in suggestion_rows
        if row.get("status") == "proposal_only"
    }
    exceptions = {
        _item_key(row): row for row in suggestion_rows if row.get("status") == "error"
    }
    steps, payloads = _load_steps(videos_root)
    reviewer = str(results["reviewer"]).strip()
    reviewed_at = results["reviewed_at"]
    sample_size = len(result_items)
    correct_count = sum(item["decision"] == "correct" for item in result_items.values())
    note = (
        f"Validation par échantillonnage stratifié {results['batch_id']} : "
        f"{sample_size} contrôles, {correct_count} corrects, "
        f"{len(incorrect_keys)} correction(s) mineure(s), 0 erreur majeure."
    )

    updates: dict[tuple[str, int], ProceduralStep] = {}
    calibrated = 0
    unchanged_low_risk = 0
    for key, original in steps.items():
        if original.status != "review_required" or key in exceptions:
            continue
        values = original.model_dump()
        suggestion = proposals.get(key)
        if suggestion:
            proposal = suggestion["proposal"]
            for field in CALIBRATED_FIELDS:
                values[field] = proposal.get(field)
            calibrated += 1
        else:
            unchanged_low_risk += 1
        if key in resolutions:
            _resolve_minor_issue(values, original, resolutions[key])
        values.update(
            status="verified",
            reviewed_by=reviewer,
            reviewed_at=reviewed_at,
            review_notes=note,
            validation_method="batch_sampled",
            validation_batch_id=results["batch_id"],
            validation_model=(suggestion.get("model") if suggestion else None),
        )
        updates[key] = ProceduralStep(**values)

    report = {
        "schema_version": 1,
        "batch_id": results["batch_id"],
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "validation_method": "batch_sampled",
        "sample": {
            "size": sample_size,
            "correct": correct_count,
            "minor_corrections": len(incorrect_keys),
            "major_errors": 0,
        },
        "population": {
            "eligible": len(updates),
            "calibrated": calibrated,
            "kept_original": unchanged_low_risk,
            "pending_exceptions": len(exceptions),
        },
        "resolutions": list(resolutions.values()),
        "applied": apply,
        "applied_at": datetime.now(timezone.utc).isoformat() if apply else None,
    }
    if not apply:
        return report

    backup_root = path(f"data/interim/video_batch_validation_backups/{results['batch_id']}")
    for video_id, (file, payload) in payloads.items():
        changed = False
        output_steps = []
        for row in payload.get("steps", []):
            key = (video_id, int(row["step_number"]))
            if key in updates:
                output_steps.append(updates[key].model_dump(mode="json"))
                changed = True
            else:
                output_steps.append(row)
        if not changed:
            continue
        backup = backup_root / video_id / "steps.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, backup)
        payload["steps"] = output_steps
        atomic_write_json(file, payload)

    audit_output = audit_output or path(
        f"data/interim/video_batch_validation_{results['batch_id']}.json"
    )
    atomic_write_json(audit_output, report)
    return report


def apply_exception_reviews(
    *,
    results_file: Path,
    manifest_file: Path,
    videos_root: Path | None = None,
    audit_output: Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Applique uniquement les exceptions contrôlées individuellement."""

    videos_root = videos_root or path("data/interim/videos")
    results, _ = validate_results(
        results_file,
        manifest_file,
        expected_review_type="exception_individual_review",
    )
    items = _unique_items(list(results["items"]), "Le résultat")
    steps, payloads = _load_steps(videos_root)
    updates: dict[tuple[str, int], ProceduralStep] = {}
    for key, item in items.items():
        original = steps.get(key)
        if original is None:
            raise ValueError(f"Étape exception absente : {key}")
        if item["decision"] != "correct":
            continue
        values = original.model_dump()
        values.update(
            status="verified",
            reviewed_by=str(results["reviewer"]).strip(),
            reviewed_at=results["reviewed_at"],
            review_notes=f"Exception contrôlée individuellement dans {results['batch_id']}.",
            validation_method="human_individual",
            validation_batch_id=results["batch_id"],
            validation_model=None,
        )
        updates[key] = ProceduralStep(**values)

    report = {
        "schema_version": 1,
        "batch_id": results["batch_id"],
        "reviewer": results["reviewer"],
        "reviewed_at": results["reviewed_at"],
        "validation_method": "human_individual",
        "reviewed": len(items),
        "verified": len(updates),
        "still_review_required": len(items) - len(updates),
        "applied": apply,
        "applied_at": datetime.now(timezone.utc).isoformat() if apply else None,
    }
    if not apply:
        return report

    backup_root = path(f"data/interim/video_batch_validation_backups/{results['batch_id']}")
    for video_id, (file, payload) in payloads.items():
        changed = False
        output_steps = []
        for row in payload.get("steps", []):
            key = (video_id, int(row["step_number"]))
            if key in updates:
                output_steps.append(updates[key].model_dump(mode="json"))
                changed = True
            else:
                output_steps.append(row)
        if not changed:
            continue
        backup = backup_root / video_id / "steps.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, backup)
        payload["steps"] = output_steps
        atomic_write_json(file, payload)

    audit_output = audit_output or path(
        f"data/interim/video_exception_validation_{results['batch_id']}.json"
    )
    atomic_write_json(audit_output, report)
    return report
