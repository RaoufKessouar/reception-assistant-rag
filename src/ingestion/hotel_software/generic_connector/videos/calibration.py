"""Second passage multimodal calibre sur les etapes verifiees par un humain.

Ce module ne valide jamais une etape automatiquement. Il produit des propositions
separees afin de mesurer le gain sur un jeu de reference humain avant tout traitement
du reste du corpus.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import math
import re
import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .....config import path, settings
from .....schema import ProceduralStep
from .common import artifact_path, atomic_write_json, atomic_write_jsonl, read_json


CALIBRATION_RULES = """Regles tirees de la revue humaine :
- reconstruire une instruction autonome avec les libelles exacts visibles ;
- une valeur montree dans le tutoriel est un exemple, pas une regle generale ;
- le curseur aide a localiser une cible mais son absence ne rend pas une capture inutile ;
- une capture dite apres ne prouve un resultat que si l'etat obtenu est visible ;
- les captures peuvent etre mal ordonnees ou appartenir a l'etape voisine ;
- ne garder que les captures apportant une preuve distincte et utile ;
- un libelle visible ne prouve pas qu'une case est cochee ou qu'une action a reussi ;
- quality_flags contient uniquement des problemes, jamais une confirmation positive ;
- ne jamais inventer un libelle, un resultat, une politique ou une action ;
- une correction de transcription n'est permise que si le texte de l'interface la prouve ;
- signaler toute incertitude restante ;
- cette sortie reste une proposition en review_required, jamais une validation humaine.
"""


class CalibrationDraft(BaseModel):
    """Proposition structuree du second passage multimodal."""

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    action_type: (
        Literal["click", "type", "select", "scroll", "hover", "none"] | None
    ) = None
    action_target: str | None = None
    action_location: str | None = None
    screen_before: str | None = None
    screen_after: str | None = None
    visual_description: str | None = None
    relevant_frame_indices: list[int] = Field(default_factory=list)
    result_frame_index: int | None = None
    confident: bool = False
    quality_flags: list[str] = Field(default_factory=list)


def _read_steps(file: Path) -> list[ProceduralStep]:
    payload = read_json(file)
    rows = payload.get("steps", []) if isinstance(payload, dict) else payload
    return [ProceduralStep(**row) for row in rows]


def load_all_steps(videos_root: Path | None = None) -> list[ProceduralStep]:
    videos_root = videos_root or path("data/interim/videos")
    steps: list[ProceduralStep] = []
    for file in sorted(videos_root.glob("*/steps.json")):
        steps.extend(_read_steps(file))
    return steps


def load_reference_steps(videos_root: Path | None = None) -> list[ProceduralStep]:
    """Charge uniquement la vérité terrain validée pour l’évaluation."""

    return [step for step in load_all_steps(videos_root) if step.status == "verified"]


def _reference_payload(step: ProceduralStep) -> dict[str, Any]:
    return {
        "video_id": step.video_id,
        "step_number": step.step_number,
        "spoken_instruction": step.spoken_instruction,
        "instruction_verifiee": step.instruction,
        "action_type": step.action_type,
        "action_target": step.action_target,
        "action_location": step.action_location,
        "screen_before": step.screen_before,
        "screen_after": step.screen_after,
        "visual_description": step.visual_description,
        "nombre_captures_retenues": len(_frame_paths(step)),
        "review_notes": step.review_notes,
    }


def write_reference_set(
    output: Path | None = None, videos_root: Path | None = None
) -> tuple[Path, list[dict[str, Any]]]:
    output = output or path("data/interim/video_verified_reference_set.json")
    references = [_reference_payload(step) for step in load_reference_steps(videos_root)]
    atomic_write_json(
        output,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status_filter": "verified",
            "references": references,
        },
    )
    return output, references


_STOPWORDS = {
    "avec",
    "cette",
    "dans",
    "depuis",
    "ensuite",
    "etape",
    "faire",
    "pour",
    "sous",
    "vous",
}


def _tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _token_f1(first: str | None, second: str | None) -> float:
    left = _tokens(first)
    right = _tokens(second)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    common = len(left & right)
    precision = common / len(left)
    recall = common / len(right)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def select_references(
    step: ProceduralStep,
    references: list[ProceduralStep],
    limit: int = 3,
) -> list[ProceduralStep]:
    """Selectionne des exemples humains proches sans reutiliser la cible."""

    query = f"{step.spoken_instruction} {step.instruction} {step.procedure}"
    candidates = [
        reference
        for reference in references
        if (reference.video_id, reference.step_number)
        != (step.video_id, step.step_number)
    ]
    return sorted(
        candidates,
        key=lambda reference: (
            _token_f1(
                query,
                f"{reference.spoken_instruction} {reference.instruction} "
                f"{reference.procedure}",
            ),
            reference.video_id,
            reference.step_number,
        ),
        reverse=True,
    )[:limit]


def _frame_paths(step: ProceduralStep) -> list[str]:
    return list(
        dict.fromkeys(
            value
            for value in [
                step.screenshot_before,
                *step.supporting_screenshots,
                step.screenshot_after,
            ]
            if value
        )
    )


def _prompt(
    step: ProceduralStep,
    frames: list[str],
    references: list[ProceduralStep],
) -> str:
    current = {
        "procedure": step.procedure,
        "spoken_instruction": step.spoken_instruction,
        "instruction_actuelle": step.instruction,
        "action_type_actuel": step.action_type,
        "action_target_actuel": step.action_target,
        "action_location_actuelle": step.action_location,
        "screen_before_actuel": step.screen_before,
        "screen_after_actuel": step.screen_after,
        "visual_description_actuelle": step.visual_description,
        "quality_flags_actuels": step.quality_flags,
        "captures": [f"F{index}" for index in range(1, len(frames) + 1)],
    }
    examples = [_reference_payload(reference) for reference in references]
    schema = CalibrationDraft.model_json_schema()
    return (
        "Tu effectues un second controle multimodal d'une etape de tutoriel logiciel de gestion hôtelière.\n\n"
        f"{CALIBRATION_RULES}\n"
        "Chaque image est identifiee F1, F2, etc. relevant_frame_indices contient "
        "uniquement les numeros des images directement utiles. result_frame_index "
        "vaut null si aucune image ne prouve l'etat obtenu apres l'action. "
        "Ne te fie pas au nom avant/apres d'origine.\n\n"
        f"ETAPE A CONTROLER:\n{json.dumps(current, ensure_ascii=False, indent=2)}\n\n"
        "EXEMPLES VERIFIES PAR UN HUMAIN, UTILISES SEULEMENT COMME REGLES DE FORME:\n"
        f"{json.dumps(examples, ensure_ascii=False, indent=2)}\n\n"
        "Retourne uniquement un objet JSON conforme a ce schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


def _extract_json(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Le VLM de calibration n'a pas produit d'objet JSON")
    return json.loads(raw[start : end + 1])


def _per_frame_pixel_budget(
    frame_count: int,
    *,
    min_pixels: int,
    max_pixels: int,
    total_pixels: int,
) -> int:
    """Evite qu'une etape riche en captures fasse exploser l'attention GPU."""

    if frame_count < 1:
        raise ValueError("frame_count doit etre positif")
    return max(min_pixels, min(max_pixels, total_pixels // frame_count))


def _encoded_resized_image(value: str, max_pixels: int) -> str:
    """Redimensionne en memoire pour le backend vLLM, sans toucher au fichier."""

    from PIL import Image

    with Image.open(artifact_path(value)) as source:
        image = source.convert("RGB")
        pixels = image.width * image.height
        if pixels > max_pixels:
            scale = math.sqrt(max_pixels / pixels)
            image = image.resize(
                (
                    max(28, int(image.width * scale)),
                    max(28, int(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _normalise_draft(draft: CalibrationDraft, frame_count: int) -> CalibrationDraft:
    relevant = list(
        dict.fromkeys(
            index
            for index in draft.relevant_frame_indices
            if 1 <= index <= frame_count
        )
    )
    result = draft.result_frame_index
    if result is not None and not 1 <= result <= frame_count:
        result = None
    if result is not None and result not in relevant:
        relevant.append(result)
    flags = sorted(set(flag.strip() for flag in draft.quality_flags if flag.strip()))
    confident = bool(draft.confident and relevant and not flags)
    return draft.model_copy(
        update={
            "relevant_frame_indices": relevant,
            "result_frame_index": result,
            "quality_flags": flags,
            "confident": confident,
        }
    )


def _draft_openai_compatible(prompt: str, frames: list[str]) -> CalibrationDraft:
    from openai import OpenAI

    cfg = settings()
    video_cfg = cfg["video"]
    per_frame_max = _per_frame_pixel_budget(
        len(frames),
        min_pixels=int(video_cfg["vlm_min_pixels"]),
        max_pixels=int(video_cfg["vlm_max_pixels"]),
        total_pixels=int(video_cfg["calibration_total_pixels"]),
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, value in enumerate(frames, 1):
        encoded = _encoded_resized_image(value, per_frame_max)
        content.extend(
            [
                {"type": "text", "text": f"Capture F{index}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ]
        )
    api_cfg = cfg.get("vlm_api", cfg["llm"])
    client = OpenAI(
        base_url=api_cfg["base_url"],
        api_key=api_cfg["api_key"],
        timeout=240.0,
        max_retries=5,
    )
    request: dict[str, Any] = {
        "model": cfg["hw"]["vlm_model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 900,
    }
    if api_cfg.get("mode") == "remote_json":
        request["response_format"] = {"type": "json_object"}
    else:
        request["extra_body"] = {
            "guided_json": CalibrationDraft.model_json_schema()
        }
    completion = client.chat.completions.create(**request)
    return CalibrationDraft.model_validate(
        _extract_json(completion.choices[0].message.content or "")
    )


def _draft_transformers(prompt: str, frames: list[str]) -> CalibrationDraft:
    import torch
    from qwen_vl_utils import process_vision_info

    from .s3_describe import _load_vlm

    cfg = settings()
    video_cfg = cfg["video"]
    per_frame_max = _per_frame_pixel_budget(
        len(frames),
        min_pixels=int(video_cfg["vlm_min_pixels"]),
        max_pixels=int(video_cfg["vlm_max_pixels"]),
        total_pixels=int(video_cfg["calibration_total_pixels"]),
    )
    model, processor = _load_vlm()
    image_inputs = None
    video_inputs = None
    inputs = None
    generated = None
    try:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index, value in enumerate(frames, 1):
            content.extend(
                [
                    {"type": "text", "text": f"Capture F{index}"},
                    {
                        "type": "image",
                        "image": str(artifact_path(value)),
                        "min_pixels": int(video_cfg["vlm_min_pixels"]),
                        "max_pixels": per_frame_max,
                    },
                ]
            )
        messages = [{"role": "user", "content": content}]
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=900, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1] :]
        raw = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return CalibrationDraft.model_validate(_extract_json(raw))
    finally:
        del generated, inputs, image_inputs, video_inputs
        gc.collect()
        torch.cuda.empty_cache()


def propose(
    step: ProceduralStep,
    references: list[ProceduralStep],
) -> CalibrationDraft:
    frames = _frame_paths(step)
    if not frames:
        raise ValueError("Aucune capture disponible pour la calibration")
    selected = select_references(step, references)
    prompt = _prompt(step, frames, selected)
    backend = settings()["hw"]["vlm_backend"]
    if backend in {"vllm", "openai"}:
        draft = _draft_openai_compatible(prompt, frames)
    elif backend == "transformers":
        draft = _draft_transformers(prompt, frames)
    else:
        raise ValueError(f"Backend VLM inconnu : {backend}")
    return _normalise_draft(draft, len(frames))


def _candidate_from_review_item(item: dict[str, Any]) -> ProceduralStep:
    screenshots = item.get("screenshots", [])
    before = next(
        (row["source"] for row in screenshots if row.get("role") == "avant"), None
    )
    after = next(
        (row["source"] for row in screenshots if row.get("role") == "apres"), None
    )
    supporting = [
        row["source"]
        for row in screenshots
        if str(row.get("role", "")).startswith("support-")
    ]
    return ProceduralStep(
        procedure=item["procedure"],
        step_number=int(item["step_number"]),
        total_steps=item.get("total_steps"),
        timestamp_start=float(item["timestamp_start"]),
        timestamp_end=float(item["timestamp_end"]),
        spoken_instruction=item["spoken_instruction"],
        instruction=item["instruction"],
        action_type=item.get("action_type"),
        action_target=item.get("action_target"),
        action_location=item.get("action_location"),
        screen_before=item.get("screen_before"),
        screen_after=item.get("screen_after"),
        visual_description=item.get("visual_description"),
        screenshot_before=before,
        screenshot_after=after,
        supporting_screenshots=supporting,
        confident=False,
        quality_flags=item.get("quality_flags", []),
        video_id=item["video_id"],
        video_title=item.get("video_title"),
        status="review_required",
    )


def _draft_values(draft: CalibrationDraft, frames: list[str]) -> dict[str, Any]:
    selected = [frames[index - 1] for index in draft.relevant_frame_indices]
    result = (
        frames[draft.result_frame_index - 1]
        if draft.result_frame_index is not None
        else None
    )
    non_result = [value for value in selected if value != result]
    before = non_result[0] if non_result else None
    supporting = non_result[1:]
    return {
        "instruction": draft.instruction,
        "action_type": draft.action_type,
        "action_target": draft.action_target,
        "action_location": draft.action_location,
        "screen_before": draft.screen_before,
        "screen_after": draft.screen_after if result else None,
        "visual_description": draft.visual_description,
        "screenshot_before": before,
        "supporting_screenshots": supporting,
        "screenshot_after": result,
        "confident": draft.confident,
        "quality_flags": draft.quality_flags,
    }


def _apply_measured_policy(
    proposal: dict[str, Any], original: ProceduralStep
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ne conserve que les champs dont le gain a ete mesure sur la verite terrain."""

    model_only = dict(proposal)
    safe = dict(proposal)
    safe.update(
        instruction=original.instruction,
        screen_after=original.screen_after,
        screenshot_before=original.screenshot_before,
        supporting_screenshots=list(original.supporting_screenshots),
        screenshot_after=original.screenshot_after,
        confident=original.confident,
        quality_flags=list(original.quality_flags),
    )
    return safe, model_only


def _set_f1(first: list[str], second: list[str]) -> float:
    left = set(first)
    right = set(second)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    common = len(left & right)
    precision = common / len(left)
    recall = common / len(right)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def _comparison(
    original: ProceduralStep,
    gold: ProceduralStep,
    proposal: dict[str, Any],
) -> dict[str, dict[str, float]]:
    original_frames = _frame_paths(original)
    gold_frames = _frame_paths(gold)
    proposed_frames = list(
        dict.fromkeys(
            value
            for value in [
                proposal.get("screenshot_before"),
                *proposal.get("supporting_screenshots", []),
                proposal.get("screenshot_after"),
            ]
            if value
        )
    )
    return {
        "instruction_token_f1": {
            "baseline": _token_f1(original.instruction, gold.instruction),
            "proposal": _token_f1(proposal.get("instruction"), gold.instruction),
        },
        "action_type_exact": {
            "baseline": float(original.action_type == gold.action_type),
            "proposal": float(proposal.get("action_type") == gold.action_type),
        },
        "action_target_token_f1": {
            "baseline": _token_f1(original.action_target, gold.action_target),
            "proposal": _token_f1(proposal.get("action_target"), gold.action_target),
        },
        "action_location_token_f1": {
            "baseline": _token_f1(original.action_location, gold.action_location),
            "proposal": _token_f1(
                proposal.get("action_location"), gold.action_location
            ),
        },
        "screen_before_token_f1": {
            "baseline": _token_f1(original.screen_before, gold.screen_before),
            "proposal": _token_f1(proposal.get("screen_before"), gold.screen_before),
        },
        "visual_description_token_f1": {
            "baseline": _token_f1(
                original.visual_description, gold.visual_description
            ),
            "proposal": _token_f1(
                proposal.get("visual_description"), gold.visual_description
            ),
        },
        "evidence_frame_f1": {
            "baseline": _set_f1(original_frames, gold_frames),
            "proposal": _set_f1(proposed_frames, gold_frames),
        },
        "screenshot_after_exact": {
            "baseline": float(original.screenshot_after == gold.screenshot_after),
            "proposal": float(proposal.get("screenshot_after") == gold.screenshot_after),
        },
        "after_state_presence_exact": {
            "baseline": float(bool(original.screen_after) == bool(gold.screen_after)),
            "proposal": float(
                bool(proposal.get("screen_after")) == bool(gold.screen_after)
            ),
        },
        "after_state_token_f1": {
            "baseline": _token_f1(original.screen_after, gold.screen_after),
            "proposal": _token_f1(proposal.get("screen_after"), gold.screen_after),
        },
    }


def _summary(items: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    compared = [item for item in items if item.get("comparison")]
    metrics: dict[str, Any] = {}
    if compared:
        for name in compared[0]["comparison"]:
            baseline = mean(item["comparison"][name]["baseline"] for item in compared)
            proposal = mean(item["comparison"][name]["proposal"] for item in compared)
            metrics[name] = {
                "baseline": round(baseline, 4),
                "proposal": round(proposal, 4),
                "delta": round(proposal - baseline, 4),
            }
    return {
        "expected": expected,
        "completed": len(compared),
        "errors": sum(bool(item.get("error")) for item in items),
        "metrics": metrics,
    }


def evaluate_reference_sample(
    *,
    packet_file: Path | None = None,
    output: Path | None = None,
    videos_root: Path | None = None,
    force: bool = False,
    proposer: Callable[[ProceduralStep, list[ProceduralStep]], CalibrationDraft]
    | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Mesure le second passage sur l'echantillon corrige par l'utilisateur."""

    packet_file = packet_file or path(
        "data/exports/video_review_sample/review_items.json"
    )
    output = output or path("data/interim/video_calibration_evaluation.json")
    packet = read_json(packet_file)
    review_items = packet.get("items", [])
    references = load_reference_steps(videos_root)
    gold = {(step.video_id, step.step_number): step for step in references}
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if output.exists() and not force:
        previous = read_json(output)
        completed = {
            (str(item["video_id"]), int(item["step_number"])): item
            for item in previous.get("items", [])
            if item.get("proposal")
        }

    infer = proposer or propose
    results: list[dict[str, Any]] = []
    for index, item in enumerate(review_items, 1):
        key = (str(item["video_id"]), int(item["step_number"]))
        original = _candidate_from_review_item(item)
        if key in completed:
            previous = completed[key]
            proposal, model_only = _apply_measured_policy(
                dict(previous["proposal"]), original
            )
            results.append(
                {
                    "video_id": key[0],
                    "step_number": key[1],
                    "proposal": proposal,
                    "model_only_proposal": previous.get(
                        "model_only_proposal", model_only
                    ),
                    "comparison": _comparison(original, gold[key], proposal),
                }
            )
            continue
        if key not in gold:
            results.append(
                {
                    "video_id": key[0],
                    "step_number": key[1],
                    "error": "Etape absente du jeu verified",
                }
            )
            continue
        frames = _frame_paths(original)
        print(f"Calibration {index}/{len(review_items)}: {key[0]} #{key[1]}", flush=True)
        try:
            draft = infer(original, references)
            raw_proposal = _draft_values(
                _normalise_draft(draft, len(frames)), frames
            )
            proposal, model_only = _apply_measured_policy(
                raw_proposal, original
            )
            result = {
                "video_id": key[0],
                "step_number": key[1],
                "proposal": proposal,
                "model_only_proposal": model_only,
                "comparison": _comparison(original, gold[key], proposal),
            }
        except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            result = {
                "video_id": key[0],
                "step_number": key[1],
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        atomic_write_json(
            output,
            {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "rules": CALIBRATION_RULES.strip().splitlines()[1:],
                "summary": _summary(results, len(review_items)),
                "items": results,
            },
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": CALIBRATION_RULES.strip().splitlines()[1:],
        "summary": _summary(results, len(review_items)),
        "items": results,
    }
    atomic_write_json(output, report)
    return output, report


def propose_remaining(
    *,
    audit_file: Path | None = None,
    output: Path | None = None,
    videos_root: Path | None = None,
    risk: Literal["high", "medium", "all"] = "high",
    limit: int | None = None,
    force: bool = False,
    proposer: Callable[[ProceduralStep, list[ProceduralStep]], CalibrationDraft]
    | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Produit des propositions separees, sans modifier steps.json."""

    audit_file = audit_file or path("data/interim/video_quality_audit.json")
    output = output or path("data/interim/video_calibration_suggestions.jsonl")
    report = read_json(audit_file)
    allowed = {"high", "medium"} if risk == "all" else {risk}
    wanted = {
        (str(item["video_id"]), int(item["step_number"]))
        for item in report.get("step_findings", [])
        if item.get("status") == "review_required" and item.get("risk") in allowed
    }
    all_steps = load_all_steps(videos_root)
    references = [step for step in all_steps if step.status == "verified"]
    candidates = [
        step
        for step in all_steps
        if (step.video_id, step.step_number) in wanted
    ]
    candidates.sort(key=lambda step: (step.video_id, step.step_number))
    if limit is not None:
        candidates = candidates[:limit]

    existing: dict[tuple[str, int], dict[str, Any]] = {}
    if output.exists():
        existing = {
            (str(item["video_id"]), int(item["step_number"])): item
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for item in [json.loads(line)]
        }
    infer = proposer or propose
    suggestions: list[dict[str, Any]] = []
    for index, step in enumerate(candidates, 1):
        key = (step.video_id, step.step_number)
        if (
            key in existing
            and not force
            and existing[key].get("status") != "error"
        ):
            suggestions.append(existing[key])
            continue
        print(f"Proposition {index}/{len(candidates)}: {step.video_id} #{step.step_number}", flush=True)
        try:
            frames = _frame_paths(step)
            draft = infer(step, references)
            raw_proposal = _draft_values(
                _normalise_draft(draft, len(frames)), frames
            )
            proposal, model_only = _apply_measured_policy(
                raw_proposal, step
            )
            item = {
                "video_id": step.video_id,
                "step_number": step.step_number,
                "status": "proposal_only",
                "model": settings()["hw"]["vlm_model"],
                "proposal": proposal,
                "model_only_proposal": model_only,
            }
        except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            item = {
                "video_id": step.video_id,
                "step_number": step.step_number,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        existing[key] = item
        suggestions.append(item)
        atomic_write_jsonl(
            output,
            [existing[key] for key in sorted(existing)],
        )
    atomic_write_jsonl(
        output,
        [existing[key] for key in sorted(existing)],
    )
    return output, suggestions


def propose_verified_corpus(
    *,
    output: Path | None = None,
    videos_root: Path | None = None,
    limit: int | None = None,
    force: bool = False,
    proposer: Callable[[ProceduralStep, list[ProceduralStep]], CalibrationDraft]
    | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Second passage sur tout le corpus verified, sans modifier les sources."""

    output = output or path(
        "data/exports/model-upgrades/qwen3-vl-full-suggestions.jsonl"
    )
    references = load_reference_steps(videos_root)
    candidates = sorted(
        references, key=lambda step: (step.video_id, step.step_number)
    )
    if limit is not None:
        candidates = candidates[:limit]

    existing: dict[tuple[str, int], dict[str, Any]] = {}
    if output.exists():
        existing = {
            (str(item["video_id"]), int(item["step_number"])): item
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for item in [json.loads(line)]
        }

    infer = proposer or propose
    tracked_fields = (
        "instruction",
        "action_type",
        "action_target",
        "action_location",
        "screen_before",
        "screen_after",
        "visual_description",
        "screenshot_before",
        "supporting_screenshots",
        "screenshot_after",
        "confident",
        "quality_flags",
    )
    current = dict(existing)
    selected: list[dict[str, Any]] = []
    for index, step in enumerate(candidates, 1):
        key = (step.video_id, step.step_number)
        previous = current.get(key)
        if previous and not force and previous.get("status") == "proposal_only":
            selected.append(previous)
            continue

        print(
            f"Corpus VLM {index}/{len(candidates)}: "
            f"{step.video_id} #{step.step_number}",
            flush=True,
        )
        original = {field: getattr(step, field) for field in tracked_fields}
        try:
            frames = _frame_paths(step)
            draft = infer(step, references)
            proposal = _draft_values(
                _normalise_draft(draft, len(frames)), frames
            )
            changed_fields = [
                field
                for field in tracked_fields
                if proposal.get(field) != original.get(field)
            ]
            item = {
                "video_id": step.video_id,
                "step_number": step.step_number,
                "status": "proposal_only",
                "source_status": step.status,
                "model": settings()["hw"]["vlm_model"],
                "original": original,
                "proposal": proposal,
                "changed_fields": changed_fields,
            }
        except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            item = {
                "video_id": step.video_id,
                "step_number": step.step_number,
                "status": "error",
                "source_status": step.status,
                "model": settings()["hw"]["vlm_model"],
                "error": f"{type(exc).__name__}: {exc}",
            }
        current[key] = item
        selected.append(item)
        atomic_write_jsonl(
            output,
            [current[item_key] for item_key in sorted(current)],
        )

    return output, selected


def unload() -> None:
    """Libere le VLM partage avec l'etape 3."""

    from .s3_describe import unload as unload_vlm

    unload_vlm()
