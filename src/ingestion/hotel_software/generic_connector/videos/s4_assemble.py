"""Etape 4 : fusion audio/visuel en etapes soumises a revue humaine."""

from __future__ import annotations

import gc
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from .....config import settings
from .....schema import ProceduralStep, StepDraft, TranscriptSegment
from .common import atomic_write_json, read_json, work_dir
from .s3_describe import _EXPLICIT_ACTION, _label_matches_spoken

SYSTEM = """Tu transformes un tutoriel de logiciel de gestion hôtelière en une etape de procedure autonome.

Contraintes absolues :
- utilise seulement la phrase prononcee et les observations visuelles fournies ;
- remplace "ici", "la" et "ce bouton" uniquement si le libelle exact est visible ;
- n'invente jamais de bouton, d'ecran, de resultat ou de politique hotel ;
- action_location provient uniquement des observations visuelles fiables, meme
  lorsque la phrase prononcee ne donne aucun emplacement ;
- si la parole et l'image se contredisent sur gauche/droite ou haut/bas,
  privilegie l'image et ajoute audio_visual_location_conflict ;
- confident=false des qu'une preuve visuelle est incertaine ;
- ajoute un quality_flag pour chaque information manquante ou contradictoire ;
- reponds uniquement avec l'objet JSON demande, en francais.
"""


def output_file(video_id: str) -> Path:
    return work_dir(video_id) / "steps.json"


def load(video_id: str) -> list[ProceduralStep]:
    file = output_file(video_id)
    if not file.exists():
        raise FileNotFoundError(f"Etapes absentes pour {video_id} : {file}")
    payload = read_json(file)
    rows = payload.get("steps", []) if isinstance(payload, dict) else payload
    return [ProceduralStep(**row) for row in rows]


def _extract_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Le LLM d'assemblage n'a pas produit d'objet JSON")
    return json.loads(raw[start : end + 1])


def _groups(descriptions: list[dict]) -> list[dict]:
    groups: dict[int, dict] = {}
    for description in descriptions:
        segment = description.get("segment")
        if segment is None:
            continue
        group = groups.setdefault(
            int(segment),
            {
                "segment": int(segment),
                "text": description.get("text", ""),
                "scenes": [],
            },
        )
        moment = description.get("moment")
        if moment in {"scene", "detail"}:
            group["scenes"].append(description)
        elif moment in {"before", "after"}:
            group[moment] = description
    return [
        groups[index]
        for index in sorted(groups)
        if "before" in groups[index] or "after" in groups[index]
    ]


def _analysis(frame: dict | None) -> dict:
    analysis = dict((frame or {}).get("analysis", {}))
    if analysis and not analysis.get("confident", False):
        for key in (
            "screen_name",
            "ui_element",
            "location",
            "action_type",
            "expected_result",
            "visual_description",
        ):
            analysis[key] = None
    return analysis


def _spatial_terms(value: str | None) -> set[str]:
    """Normalise les directions utiles a la comparaison audio/visuel."""

    if not value:
        return set()
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    tokens = set(re.findall(r"[a-z]+", normalized))
    aliases = {
        "gauche": "left",
        "left": "left",
        "droite": "right",
        "right": "right",
        "haut": "top",
        "superieur": "top",
        "superieure": "top",
        "top": "top",
        "bas": "bottom",
        "inferieur": "bottom",
        "inferieure": "bottom",
        "bottom": "bottom",
        "centre": "center",
        "milieu": "center",
        "center": "center",
    }
    return {aliases[token] for token in tokens if token in aliases}


def _opposing_location(first: str, second: str) -> bool:
    first_terms = _spatial_terms(first)
    second_terms = _spatial_terms(second)
    return bool(
        ({"left", "right"} <= first_terms | second_terms)
        or ({"top", "bottom"} <= first_terms | second_terms)
    )


def _guard_draft(group: dict, draft: StepDraft) -> StepDraft:
    """Retire toute précision qui n'est soutenue par aucune preuve fiable."""

    guarded = draft.model_copy(deep=True)
    before_trusted = bool(
        _analysis(group.get("before")).get("confident", False)
    )
    after_trusted = bool(_analysis(group.get("after")).get("confident", False))
    trusted_analyses = [
        analysis
        for frame in [group.get("before"), group.get("after"), *group["scenes"]]
        if (analysis := _analysis(frame)).get("confident", False)
    ]
    visual_locations = [
        analysis["location"]
        for analysis in trusted_analyses
        if analysis.get("location")
    ]
    evidence = " ".join(
        [group["text"]]
        + [analysis.get("ui_element") or "" for analysis in trusted_analyses]
    )
    flags = list(guarded.quality_flags)

    if guarded.action_target and not _label_matches_spoken(
        guarded.action_target, evidence
    ):
        guarded.action_target = None
        guarded.action_location = None
        guarded.instruction = group["text"].strip()
        flags.append("unsupported_action_target")
    if guarded.action_type not in {None, "none"} and not _EXPLICIT_ACTION.search(
        group["text"]
    ):
        guarded.action_type = "none"
        flags.append("unsupported_action_type")
    if visual_locations:
        visual_location = visual_locations[0]
        if guarded.action_location and _opposing_location(
            guarded.action_location, visual_location
        ):
            guarded.action_location = visual_location
            flags.append("audio_visual_location_conflict")
        elif not guarded.action_location:
            guarded.action_location = visual_location
    elif guarded.action_location:
        guarded.action_location = None
        flags.append("missing_visual_location")
    if not trusted_analyses:
        guarded.action_location = None
        guarded.visual_description = None
    if not before_trusted:
        guarded.before_state = None
    if not after_trusted:
        guarded.after_state = None

    guarded.confident = guarded.confident and before_trusted and after_trusted
    guarded.quality_flags = sorted(set(flags))
    return guarded


def _automatic_flags(group: dict) -> list[str]:
    flags = []
    before = _analysis(group.get("before"))
    after = _analysis(group.get("after"))
    if not before:
        flags.append("missing_before_frame")
    if not after:
        flags.append("missing_after_frame")
    if before and not before.get("confident", False):
        flags.append("uncertain_before_visual")
    if after and not after.get("confident", False):
        flags.append("uncertain_after_visual")
    if not before.get("ui_element") and not after.get("ui_element"):
        flags.append("missing_exact_ui_label")
    return flags


def _final_confidence(
    draft_confident: bool, visual_confident: bool, flags: list[str]
) -> bool:
    return bool(draft_confident and visual_confident and not flags)


def _payload(group: dict) -> dict:
    return {
        "phrase_prononcee": group["text"],
        "capture_avant": _analysis(group.get("before")),
        "capture_apres": _analysis(group.get("after")),
        "transitions_associees": [_analysis(frame) for frame in group["scenes"]],
    }


def _draft_vllm(client, model: str, group: dict) -> StepDraft:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": json.dumps(_payload(group), ensure_ascii=False, indent=2),
            },
        ],
        extra_body={"guided_json": StepDraft.model_json_schema()},
        temperature=0.0,
        max_tokens=600,
    )
    return StepDraft.model_validate(
        _extract_json(completion.choices[0].message.content or "")
    )


@lru_cache(maxsize=1)
def _load_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hw = settings()["hw"]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[hw["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        hw["llm_model"],
        dtype=dtype,
        attn_implementation=hw["attn_implementation"],
        device_map={"": hw["llm_device"]},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hw["llm_model"], use_fast=True)
    return model, tokenizer


def unload() -> None:
    _load_llm.cache_clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


def _draft_transformers(group: dict) -> StepDraft:
    import torch

    model, tokenizer = _load_llm()
    schema = json.dumps(StepDraft.model_json_schema(), ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Donnees:\n{json.dumps(_payload(group), ensure_ascii=False, indent=2)}"
                f"\n\nSchema JSON exact a respecter:\n{schema}"
            ),
        },
    ]
    last_error: Exception | None = None
    for attempt in range(2):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg["llm"].get("enable_thinking", False),
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        try:
            return StepDraft.model_validate(_extract_json(raw))
        except ValueError as exc:
            last_error = exc
            messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Le JSON precedent est invalide. Corrige-le et retourne "
                            "uniquement un objet conforme au schema, sans markdown."
                        ),
                    },
                ]
            )
            print(f"Nouvel essai JSON LLM ({attempt + 1}/2): {exc}", flush=True)
    raise ValueError(f"Le LLM n'a pas produit un StepDraft valide: {last_error}")


def _draft(client, model: str, group: dict) -> StepDraft:
    backend = settings()["hw"].get("llm_backend", "vllm")
    if backend == "vllm":
        return _draft_vllm(client, model, group)
    if backend == "transformers":
        return _draft_transformers(group)
    raise ValueError(f"Backend LLM inconnu: {backend}")


def run(
    video_id: str,
    descriptions: list[dict],
    segments: list[TranscriptSegment],
    procedure: str,
    video_title: str,
    video_path: str,
    force: bool = False,
) -> list[ProceduralStep]:
    file = output_file(video_id)
    previous = [] if force or not file.exists() else load(video_id)
    completed = {
        step.source_segment_index: step
        for step in previous
        if step.source_segment_index is not None
    }

    cfg = settings()
    client = None
    if cfg["hw"].get("llm_backend", "vllm") == "vllm":
        from openai import OpenAI

        client = OpenAI(
            base_url=cfg["llm"]["base_url"],
            api_key=cfg["llm"]["api_key"],
            timeout=180.0,
        )
    groups = _groups(descriptions)
    for index, group in enumerate(groups, start=1):
        segment_index = group["segment"]
        if segment_index in completed:
            print(f"LLM {index}/{len(groups)}: deja assemble", flush=True)
            continue
        if segment_index >= len(segments):
            raise ValueError(
                f"Segment {segment_index} absent de la transcription {video_id}"
            )

        print(f"LLM {index}/{len(groups)}: segment {segment_index}", flush=True)
        draft = _guard_draft(
            group, _draft(client, cfg["hw"]["llm_model"], group)
        )
        flags = sorted(set(draft.quality_flags + _automatic_flags(group)))
        visual_confident = bool(
            group.get("before")
            and group.get("after")
            and _analysis(group["before"]).get("confident", False)
            and _analysis(group["after"]).get("confident", False)
        )
        source_segment = segments[segment_index]
        before = group.get("before")
        after = group.get("after")
        step = ProceduralStep(
            procedure=procedure,
            step_number=len(completed) + 1,
            total_steps=len(groups),
            source_segment_index=segment_index,
            timestamp_start=source_segment.start,
            timestamp_end=source_segment.end,
            spoken_instruction=source_segment.text,
            instruction=draft.instruction,
            screen_before=draft.before_state,
            screen_after=draft.after_state,
            action_type=draft.action_type,
            action_target=draft.action_target,
            action_location=draft.action_location,
            visual_description=draft.visual_description,
            screenshot_before=before.get("file") if before else None,
            screenshot_after=after.get("file") if after else None,
            supporting_screenshots=[
                frame["file"] for frame in group["scenes"] if frame.get("file")
            ],
            confident=_final_confidence(draft.confident, visual_confident, flags),
            quality_flags=flags,
            video_id=video_id,
            video_title=video_title,
            video_path=video_path,
            status="review_required",
        )
        completed[segment_index] = step
        ordered = [completed[index] for index in sorted(completed)]
        for number, item in enumerate(ordered, 1):
            item.step_number = number
            item.total_steps = len(groups)
        atomic_write_json(
            file,
            {
                "schema_version": 2,
                "video_id": video_id,
                "steps": [item.model_dump(mode="json") for item in ordered],
            },
        )

    return [completed[index] for index in sorted(completed)]


def revalidate(video_id: str) -> list[ProceduralStep]:
    """Réapplique les invariants sans relancer le LLM."""

    file = output_file(video_id)
    steps = load(video_id)
    for step in steps:
        if step.quality_flags:
            step.confident = False
    atomic_write_json(
        file,
        {
            "schema_version": 2,
            "video_id": video_id,
            "steps": [step.model_dump(mode="json") for step in steps],
        },
    )
    return steps
