"""Etape 3 : lecture factuelle des captures avec un VLM."""

from __future__ import annotations

import base64
import gc
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from .....config import settings
from .....schema import ScreenAnalysis
from .common import artifact_path, atomic_write_json, read_json, work_dir

PROMPT = """Tu analyses une capture d'ecran d'un logiciel de gestion hôtelière hotelier.
Le narrateur dit a cet instant :

{spoken}

Type de capture : {moment}

Retourne uniquement UN objet JSON avec exactement ces huit cles :
{{
  "screen_name": "texte ou null",
  "ui_element": "libelle exact ou null",
  "location": "position ou null",
  "action_type": "click, type, select, scroll, hover, none ou null",
  "expected_result": "texte ou null",
  "visual_description": "texte factuel ou null",
  "confident": false,
  "uncertainty_reason": "texte ou null"
}}

Regles :
- n'ajoute aucune autre cle et ne retourne jamais de liste ;
- la parole donne du contexte, mais ne prouve aucun libelle ni resultat visible ;
- screen_name est uniquement un titre d'ecran ou de fenetre reellement lisible ;
- recopie ui_element exactement comme il apparait a l'ecran, jamais une
  expression generique comme "reservation existante" ;
- choisis uniquement l'element lie a la phrase prononcee ; ignore les boutons,
  dates et menus visibles mais sans rapport ;
- observe en priorite le pointeur de souris, l'element survole, l'infobulle et
  la fenetre active pour identifier la cible ;
- ui_element contient seulement le texte recopie, sans ajouter "le bouton",
  "la date", un article ou une date reconstruite ;
- si la cible est une icone sans texte lisible, mets ui_element=null et decris
  seulement sa position ;
- location vient uniquement de l'image, jamais de la phrase prononcee ;
- meme si la phrase ne donne aucun emplacement, decris la position de la cible
  par rapport a une section ou un libelle visible (par exemple "a droite de la
  reservation" ou "sous la colonne Prestations") ;
- si une direction prononcee contredit l'image, conserve la position visible et
  signale la contradiction dans uncertainty_reason ;
- ne deduis jamais un libelle masque ou illisible ;
- action_type vaut "none" si aucune action n'est visible ou impliquee ;
- expected_result est toujours null ; decris l'etat deja visible uniquement
  dans visual_description ;
- visual_description decrit seulement les elements visibles et ne dit jamais
  que l'utilisateur clique, saisit ou va obtenir un resultat ;
- confident=true uniquement si chaque affirmation est directement visible et
  si tout libelle nomme est lisible exactement ;
- en cas de doute, utilise null, confident=false et explique uncertainty_reason.
"""


def _prompt_for(frame: dict) -> str:
    return PROMPT.format(
        spoken=frame["text"] or "(aucune parole)", moment=frame["moment"]
    )


def output_file(video_id: str) -> Path:
    return work_dir(video_id) / "descriptions.json"


def load(video_id: str) -> list[dict]:
    file = output_file(video_id)
    if not file.exists():
        raise FileNotFoundError(f"Descriptions VLM absentes pour {video_id} : {file}")
    payload = read_json(file)
    return payload.get("descriptions", []) if isinstance(payload, dict) else payload


def _extract_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Le VLM n'a pas produit d'objet JSON")
    return json.loads(raw[start : end + 1])


def _validated_analysis(payload: dict) -> dict:
    return ScreenAnalysis.model_validate(payload).model_dump(mode="json")


def _failure(reason: str) -> dict:
    return ScreenAnalysis(confident=False, uncertainty_reason=reason[:500]).model_dump(
        mode="json"
    )


_ACTION_DESCRIPTION = re.compile(
    r"\b(?:cliquez?|saisissez?|selectionnez?|sélectionnez?|l'utilisateur|va |"
    r"permet de|pour modifier|pour ouvrir)\b",
    flags=re.IGNORECASE,
)
_GENERIC_UI_PREFIX = re.compile(
    r"^(?:le |la |un |une |l')?(?:bouton|date|icone|icône|reservation existante|"
    r"réservation existante)\b",
    flags=re.IGNORECASE,
)
_DATE_LIKE_UI = re.compile(r"^\d{1,2}(?:[/.-]\d{1,4}){0,2}$")
_DATE_CUE = re.compile(r"\b(?:date|jour|arrivee|arrivée|depart|départ)\b", re.I)
_EXPLICIT_ACTION = re.compile(
    r"\b(?:cliquez|cliquer|selectionnez|sélectionnez|selectionner|sélectionner|"
    r"saisissez|saisir|tapez|taper|ouvrez|ouvrir|appuyez|appuyer|survolez|"
    r"survoler|choisissez|choisir|rendez-vous|allez|modifiez|modifier|annulez|"
    r"annuler|supprimez|supprimer|transferez|transférez|transferer|transférer|"
    r"ajoutez|ajouter|affichez|afficher)\b",
    re.IGNORECASE,
)
_LABEL_STOPWORDS = {
    "avec",
    "dans",
    "pour",
    "sous",
    "texte",
    "bouton",
    "champ",
    "selectionnes",
}


def _semantic_tokens(value: str) -> set[str]:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    return {
        token
        for token in re.findall(r"[a-z]+", normalized)
        if len(token) >= 4 and token not in _LABEL_STOPWORDS
    }


def _label_matches_spoken(element: str, spoken: str) -> bool:
    element_tokens = _semantic_tokens(element)
    spoken_tokens = _semantic_tokens(spoken)
    return bool(
        element_tokens
        and any(
            element_token[:5] == spoken_token[:5]
            for element_token in element_tokens
            for spoken_token in spoken_tokens
        )
    )


def _semantic_guard(frame: dict, analysis: dict) -> dict:
    """Dégrade les affirmations qui ne décrivent pas un fait visible."""

    guarded = dict(analysis)
    reasons: list[str] = []
    spoken = frame.get("text", "")
    if guarded.get("expected_result") is not None:
        guarded["expected_result"] = None

    description = guarded.get("visual_description") or ""
    if _ACTION_DESCRIPTION.search(description):
        guarded["visual_description"] = None
        reasons.append("description d'action au lieu d'un etat visible")

    element = guarded.get("ui_element") or ""
    if element and _GENERIC_UI_PREFIX.search(element):
        reasons.append("element UI reformule au lieu d'un libelle exact")
    if (
        element
        and _DATE_LIKE_UI.fullmatch(element.strip())
        and not re.search(rf"(?<!\d){re.escape(element.strip())}(?!\d)", spoken)
        and not _DATE_CUE.search(spoken)
    ):
        guarded["ui_element"] = None
        guarded["location"] = None
        reasons.append("date visible sans rapport avec la cible prononcee")
    elif element and not _label_matches_spoken(element, spoken):
        guarded["ui_element"] = None
        guarded["location"] = None
        reasons.append("libelle visible sans correspondance avec la phrase prononcee")

    action_type = guarded.get("action_type")
    if action_type not in {None, "none"} and not _EXPLICIT_ACTION.search(spoken):
        guarded["action_type"] = "none"
        reasons.append("aucune action explicite dans la phrase prononcee")

    if reasons:
        guarded["confident"] = False
        previous = guarded.get("uncertainty_reason")
        guarded["uncertainty_reason"] = "; ".join(
            ([previous] if previous else []) + reasons
        )[:500]
    return _validated_analysis(guarded)


def revalidate(video_id: str) -> list[dict]:
    """Réapplique les garde-fous sans relancer le modèle GPU."""

    file = output_file(video_id)
    descriptions = load(video_id)
    guarded = [
        {**item, "analysis": _semantic_guard(item, item["analysis"])}
        for item in descriptions
    ]
    atomic_write_json(
        file,
        {"schema_version": 2, "video_id": video_id, "descriptions": guarded},
    )
    return guarded


def _describe_vllm(frame: dict) -> dict:
    from openai import OpenAI

    cfg = settings()
    client = OpenAI(
        base_url=cfg["llm"]["base_url"], api_key=cfg["llm"]["api_key"], timeout=120.0
    )
    image_bytes = artifact_path(frame["file"]).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    completion = client.chat.completions.create(
        model=cfg["hw"]["vlm_model"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                    {
                        "type": "text",
                        "text": _prompt_for(frame),
                    },
                ],
            }
        ],
        extra_body={"guided_json": ScreenAnalysis.model_json_schema()},
        temperature=0.0,
        max_tokens=500,
    )
    return _validated_analysis(
        _extract_json(completion.choices[0].message.content or "")
    )


@lru_cache(maxsize=1)
def _load_vlm():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    cfg = settings()
    hw = cfg["hw"]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[hw["dtype"]]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hw["vlm_model"],
        dtype=dtype,
        attn_implementation=hw["attn_implementation"],
        device_map={"": hw["vlm_device"]},
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        hw["vlm_model"],
        min_pixels=int(cfg["video"]["vlm_min_pixels"]),
        max_pixels=int(cfg["video"]["vlm_max_pixels"]),
        use_fast=False,
    )
    return model, processor


def unload() -> None:
    _load_vlm.cache_clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


def _describe_transformers(frame: dict) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = _load_vlm()
    image_path = artifact_path(frame["file"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {
                    "type": "text",
                    "text": _prompt_for(frame),
                },
            ],
        }
    ]
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
        generated = model.generate(**inputs, max_new_tokens=350, do_sample=False)
    generated = generated[:, inputs.input_ids.shape[1] :]
    raw = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return _validated_analysis(_extract_json(raw))


def run(video_id: str, frames: list[dict], force: bool = False) -> list[dict]:
    file = output_file(video_id)
    previous = [] if force or not file.exists() else load(video_id)
    completed = {item["frame_id"]: item for item in previous}
    backend = settings()["hw"]["vlm_backend"]
    if backend not in {"vllm", "transformers"}:
        raise ValueError(f"Backend VLM inconnu : {backend}")

    total = len(frames)
    for index, frame in enumerate(frames, start=1):
        if frame["frame_id"] in completed:
            print(f"VLM {index}/{total}: deja analyse ({frame['frame_id']})", flush=True)
            continue
        print(f"VLM {index}/{total}: {frame['frame_id']}", flush=True)
        try:
            analysis = (
                _describe_vllm(frame)
                if backend == "vllm"
                else _describe_transformers(frame)
            )
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            analysis = _failure(f"{type(exc).__name__}: {exc}")
        analysis = _semantic_guard(frame, analysis)
        completed[frame["frame_id"]] = {**frame, "analysis": analysis}
        ordered = [
            completed[item["frame_id"]]
            for item in frames
            if item["frame_id"] in completed
        ]
        atomic_write_json(
            file,
            {"schema_version": 2, "video_id": video_id, "descriptions": ordered},
        )

    return [completed[frame["frame_id"]] for frame in frames]
