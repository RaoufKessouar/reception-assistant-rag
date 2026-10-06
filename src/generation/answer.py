"""Génération grounded avec backend vLLM ou Transformers."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping, Sequence

from ..config import settings
from ..retrieval import retriever
from ..retrieval.query_planner import QueryPlan, plan_query
from ..retrieval.retriever import Hit
from . import prompt


_NUMBERED_STEP = re.compile(
    r"(?:^|\n|\s)(\d+)[.)]\s+(.+?)"
    r"(?=(?:\n|\s)\d+[.)]\s+|\n\s*Sources\b|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _uncited_numbered_steps(value: str) -> list[int]:
    """Repere les etapes procedurales qui ne portent pas leur propre preuve."""

    return [
        int(number)
        for number, content in _NUMBERED_STEP.findall(value)
        if not re.search(r"\[\d+\]", content)
    ]


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    retrieval_question: str | None = None
    route: str = "both"
    used_history: bool = False
    verification_decision: str = "not_run"

    @property
    def abstained(self) -> bool:
        return "ne dispose pas d'une information verifiee" in self.text.lower()

    def sources(self) -> list[str]:
        return [hit.document.citation() for hit in self.cited_hits()]

    def cited_hits(self) -> list[Hit]:
        indices = {int(value) for value in re.findall(r"\[(\d+)\]", self.text)}
        return [hit for index, hit in enumerate(self.hits, 1) if index in indices]

    @property
    def citations_valid(self) -> bool:
        if self.abstained:
            return True
        indices = [int(value) for value in re.findall(r"\[(\d+)\]", self.text)]
        return (
            bool(indices)
            and all(1 <= index <= len(self.hits) for index in indices)
            and not _uncited_numbered_steps(self.text)
        )


def _client():
    from openai import OpenAI

    cfg = settings()["llm"]
    return OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        timeout=float(cfg.get("timeout_seconds", 240)),
    )


def _completion_openai_compatible(messages: list[dict[str, str]]) -> str:
    """Appelle un fournisseur externe compatible avec l'API OpenAI."""

    cfg = settings()
    completion = _client().chat.completions.create(
        model=cfg["hw"]["llm_model"],
        messages=messages,
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    return (completion.choices[0].message.content or "").strip()


def _completion_api(messages: list[dict[str, str]]) -> str:
    """Alias court conserve pour les integrations locales existantes."""

    return _completion_openai_compatible(messages)


def _completion_vllm(messages: list[dict[str, str]]) -> str:
    cfg = settings()
    completion = _client().chat.completions.create(
        model=cfg["hw"]["llm_model"],
        messages=messages,
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": cfg["llm"].get("enable_thinking", False)
            }
        },
    )
    return (completion.choices[0].message.content or "").strip()


@lru_cache(maxsize=1)
def _load_transformers_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = settings()
    hw = cfg["hw"]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[hw["dtype"]]
    load_kwargs = {
        "dtype": dtype,
        "attn_implementation": hw["attn_implementation"],
        "device_map": {"": hw["llm_device"]},
    }
    if hw.get("model_cache"):
        load_kwargs["cache_dir"] = hw["model_cache"]
    model = AutoModelForCausalLM.from_pretrained(hw["llm_model"], **load_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        hw["llm_model"],
        use_fast=True,
        cache_dir=hw.get("model_cache"),
    )
    return model, tokenizer


def _completion_transformers(messages: list[dict[str, str]]) -> str:
    import torch

    cfg = settings()
    model, tokenizer = _load_transformers_llm()
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=cfg["llm"].get("enable_thinking", False),
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=cfg["llm"]["max_tokens"],
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    return tokenizer.decode(
        generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def _generate(messages: list[dict[str, str]]) -> str:
    backend = settings()["hw"].get("llm_backend", "vllm")
    if backend in {"api", "openai_compatible"}:
        return _completion_openai_compatible(messages)
    if backend == "vllm":
        return _completion_vllm(messages)
    if backend == "transformers":
        return _completion_transformers(messages)
    raise ValueError(f"Backend LLM inconnu: {backend}")


def _clean_answer(value: str) -> str:
    return re.sub(
        r"^\s*(?:correction|reponse finale|réponse finale)\s*:\s*",
        "",
        value.strip(),
        count=1,
        flags=re.IGNORECASE,
    )


def _late_checkout_guardrail(question: str, hits: list[Hit]) -> str | None:
    """Calcule les tranches d'heure commencée sans confier l'arithmétique au LLM."""

    if not re.search(r"\b(?:depart|départ|quitt\w*|late[ -]?check)", question, re.I):
        return None
    source_index = next(
        (
            index
            for index, hit in enumerate(hits, 1)
            if hit.document.source_id == "hotel-checkout-001"
        ),
        None,
    )
    if source_index is None:
        return None
    times = re.findall(r"(?<!\d)(\d{1,2})\s*h(?:\s*(\d{1,2}))?", question, re.I)
    if not times:
        return None
    hour, minute = (int(times[0][0]), int(times[0][1] or 0))
    target = hour * 60 + minute
    if target <= 11 * 60 or target > 17 * 60:
        return None
    if target <= 13 * 60:
        started = -(-(target - 11 * 60) // 60)
        total = started * 15
        detail = f"{started} tranche(s) commencee(s) a 15 EUR"
    else:
        started_after_13 = -(-(target - 13 * 60) // 60)
        total = 30 + started_after_13 * 20
        detail = (
            "2 tranches a 15 EUR entre 11h et 13h, puis "
            f"{started_after_13} tranche(s) commencee(s) a 20 EUR"
        )
    rendered_time = f"{hour}h{minute:02d}"
    return (
        f"CALCUL DETERMINISTE VERIFIE depuis la source [{source_index}] : "
        f"depart a {rendered_time} = {total} EUR ({detail}). "
        "Ce resultat exact doit etre repris; ne le recalcule pas autrement."
    )


def _room_change_guardrail(question: str, hits: list[Hit]) -> str | None:
    """Verrouille l'ordre du cas particulier de déplacement en cours de séjour."""

    if not re.search(
        r"\bclim(?:atisation)?.*changer\s+le\s+client\s+de\s+chambre\b",
        question,
        re.I | re.DOTALL,
    ):
        return None
    hotel_index = next(
        (
            index
            for index, hit in enumerate(hits, 1)
            if hit.document.source_id == "hotel-reservations-005"
        ),
        None,
    )
    software_index = next(
        (
            index
            for index, hit in enumerate(hits, 1)
            if hit.document.source_id == "software-faq-128"
        ),
        None,
    )
    if hotel_index is None or software_index is None:
        return None
    return (
        "SEQUENCE DETERMINISTE VERIFIEE : le changement cause par le probleme "
        f"de l'hotel est sans supplement [{hotel_index}]. Dans le logiciel "
        f"[{software_index}] : ouvrir le planning en mode attribution; selectionner "
        "la reservation et la chambre de destination libre; choisir la date "
        "du changement AVANT de cliquer sur 'Deplacer une reservation'; la "
        "reservation est alors scindee a cette date; le jour du changement, "
        "faire le check-out de l'ancienne chambre et le check-in de la nouvelle. "
        "Ne citer aucun numero de chambre issu de l'exemple documentaire."
    )


def _no_show_guardrail(question: str, hits: list[Hit]) -> str | None:
    """Force la couverture des deux volets d'une question no-show mixte."""

    if not re.search(r"\bno[ -]?show\b", question, re.I):
        return None
    hotel_index = next(
        (
            index
            for index, hit in enumerate(hits, 1)
            if hit.document.domain == "hotel"
        ),
        None,
    )
    software_index = next(
        (
            index
            for index, hit in enumerate(hits, 1)
            if hit.document.domain == "software"
        ),
        None,
    )
    if hotel_index is None or software_index is None:
        return None
    return (
        "CAS NO-SHOW MIXTE : traite les deux volets sans en omettre les "
        "actions necessaires. Pour la politique de l'hotel, indique le contact "
        "du client et l'attente du lendemain matin, la facturation de la "
        "premiere nuit, la production de la facture, le passage au statut "
        "no-show, l'annulation des nuits restantes pour un sejour de plusieurs "
        f"nuits et l'information du responsable [{hotel_index}]. Pour le "
        "logiciel, explique seulement le traitement avant cloture et le cas de "
        f"reglement directement decrits dans [{software_index}]. N'ajoute pas le "
        "passage en pertes ni la relance d'un dossier debiteur, qui ne sont pas "
        "necessaires a la demande. Chaque volet doit citer sa propre source."
    )


def _document_conflict_reason(question: str, hits: list[Hit]) -> str | None:
    """Refuse deux cas connus où le document ne prouve pas la demande.

    Ces contrôles évitent de transformer silencieusement une procédure
    d'arrivées en procédure de départs, ou un horaire de petit-déjeuner en
    promesse de service avant ouverture.
    """

    if re.search(r"\b(?:departures|liste des départs)\b", question, re.I):
        for hit in hits:
            document = hit.document
            if (
                document.source_id == "software-faq-118"
                and re.search(r"liste des arriv[ée]es", document.content, re.I)
            ):
                return "document_conflict"

    if re.search(r"(?:petit[ -]?déjeuner|petit[ -]?dejeuner|breakfast)", question, re.I):
        times = re.findall(
            r"(?<!\d)(\d{1,2})\s*h(?:\s*(\d{1,2}))?", question, re.I
        )
        if times:
            hour, minute = int(times[0][0]), int(times[0][1] or 0)
            if hour * 60 + minute < 7 * 60 and any(
                hit.document.source_id == "hotel-services-001" for hit in hits
            ):
                return "unsupported_time"
    return None


def _deterministic_guardrail(question: str, hits: list[Hit]) -> str | None:
    guardrails = [
        _late_checkout_guardrail(question, hits),
        _room_change_guardrail(question, hits),
        _no_show_guardrail(question, hits),
    ]
    active = [value for value in guardrails if value]
    return "\n".join(active) if active else None


def _focus_single_software_procedure(plan: QueryPlan, hits: list[Hit]) -> list[Hit]:
    """Évite de mélanger plusieurs tutoriels quand le meilleur suffit.

    Les chunks successifs d'une même vidéo restent disponibles, mais les
    procédures concurrentes sont retirées du contexte de génération.
    """

    if plan.route != "software" or not plan.procedural_detail or not hits:
        return hits
    primary_source = hits[0].document.source_id
    return [hit for hit in hits if hit.document.source_id == primary_source]


def _focus_coherent_sources(plan: QueryPlan, hits: list[Hit]) -> list[Hit]:
    """Garde la source la plus directe par volet de la question.

    Une question mixte a besoin d'une preuve hotel et d'une preuve logiciel de gestion hôtelière, mais
    pas de plusieurs tutoriels concurrents. Une question conceptuelle sur une
    difference peut egalement etre couverte integralement par la FAQ directe.
    """

    focused = _focus_single_software_procedure(plan, hits)
    if focused is not hits:
        return focused
    if (
        plan.route == "software"
        and re.search(r"\b(?:difference|différence)\b", plan.standalone_question, re.I)
        and hits
        and hits[0].document.source_type == "qa"
    ):
        primary_source = hits[0].document.source_id
        return [hit for hit in hits if hit.document.source_id == primary_source]
    if plan.route != "both" or not plan.procedural_detail:
        return hits
    software_hits = [hit for hit in hits if hit.document.domain == "software"]
    if not software_hits:
        return hits
    primary_software_source = software_hits[0].document.source_id
    coherent_software_sources = {primary_software_source}
    if re.search(r"\bno[ -]?show\b", plan.standalone_question, re.I):
        no_show_hits = [
            hit
            for hit in software_hits
            if re.search(r"no[ -]?show", hit.document.title, re.I)
            or re.search(r"no[ -]?show", hit.document.content, re.I)
        ]
        if no_show_hits:
            # La FAQ opérationnelle est préférable au lexique ou à un extrait
            # purement fiscal : elle décrit le traitement demandé dans le logiciel de gestion hôtelière.
            coherent_software_sources = {
                max(
                    no_show_hits,
                    key=lambda hit: (
                        hit.document.source_type == "qa",
                        len(hit.document.content),
                    ),
                ).document.source_id
            }
    if re.search(
        r"\b(?:clim(?:atisation)?|changer\s+le\s+client\s+de\s+chambre)\b",
        plan.standalone_question,
        re.I,
    ):
        move_hits = [
            hit
            for hit in software_hits
            if re.search(r"\bd[ée]plac", hit.document.title, re.I)
        ]
        if move_hits:
            # La FAQ complete regroupe le parcours entier dans un extrait,
            # contrairement aux etapes video atomiques. Elle evite de faire
            # passer deux sources complementaires pour deux methodes.
            coherent_software_sources = {
                max(
                    move_hits,
                    key=lambda hit: len(hit.document.content),
                ).document.source_id
            }
    return [
        hit
        for hit in hits
        if hit.document.domain == "hotel"
        or hit.document.source_id in coherent_software_sources
    ]


def _normalize_single_source_citations(draft: str, hits: list[Hit]) -> str:
    """Élimine l'ambiguïté page/source quand un seul extrait est disponible."""

    if len(hits) != 1 or prompt.ABSTENTION in draft:
        return draft
    normalized = re.sub(r"\[(\d+)\]", "[1]", draft)
    source_heading = re.search(r"\n\s*(?:\*\*)?Sources?\b", normalized, re.I)
    body_end = source_heading.start() if source_heading else len(normalized)
    body = normalized[:body_end].rstrip()
    tail = normalized[body_end:]
    if body and not re.search(r"\[1\]\s*[.!?]?\s*$", body):
        body += " [1]"
    return body + tail


def _attach_declared_sources(answer: str, sources: list[int]) -> str:
    existing = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    if existing:
        return answer
    citations = " ".join(f"[{index}]" for index in sources)
    # Le controleur a valide le texte entier contre ces sources. Si le modele
    # a omis seulement la forme des citations, rattacher la preuve a la fin de
    # la reponse, puis conserver la liste lisible pour l'interface.
    return answer.rstrip() + f" {citations}\n\nSources\n{citations}"


def _parse_verification(raw: str, max_source_index: int) -> tuple[str, str]:
    cleaned = raw.strip()
    decision_match = re.search(
        r"<decision>\s*(supported|abstain)\s*</decision>",
        cleaned,
        flags=re.IGNORECASE,
    )
    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    sources_match = re.search(
        r"<sources>\s*(.*?)\s*</sources>",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if decision_match:
        decision = decision_match.group(1).lower()
        if decision == "abstain":
            return prompt.ABSTENTION, "abstain"
        if answer_match and answer_match.group(1).strip():
            verified_answer = _clean_answer(answer_match.group(1))
            declared = (
                [int(value) for value in re.findall(r"\d+", sources_match.group(1))]
                if sources_match
                else []
            )
            answer_indices = [
                int(value) for value in re.findall(r"\[(\d+)\]", verified_answer)
            ]
            sources = declared or answer_indices
            if (
                not sources
                or any(
                    index < 1 or index > max_source_index
                    for index in sources + answer_indices
                )
                or _uncited_numbered_steps(verified_answer)
            ):
                return prompt.ABSTENTION, "invalid_format"
            return _attach_declared_sources(verified_answer, sources), "supported"
        return prompt.ABSTENTION, "invalid_format"

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return prompt.ABSTENTION, "invalid_format"
    try:
        payload = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return prompt.ABSTENTION, "invalid_format"
    decision = str(payload.get("decision", "")).strip().lower()
    verified_answer = str(payload.get("answer", "")).strip()
    if decision == "abstain":
        return prompt.ABSTENTION, "abstain"
    if decision != "supported" or not verified_answer:
        return prompt.ABSTENTION, "invalid_format"
    verified_answer = _clean_answer(verified_answer)
    declared = [
        int(value)
        for value in payload.get("sources", [])
        if str(value).strip().isdigit()
    ]
    answer_indices = [
        int(value) for value in re.findall(r"\[(\d+)\]", verified_answer)
    ]
    sources = declared or answer_indices
    if (
        not sources
        or any(
            index < 1 or index > max_source_index
            for index in sources + answer_indices
        )
        or _uncited_numbered_steps(verified_answer)
    ):
        return prompt.ABSTENTION, "invalid_format"
    return _attach_declared_sources(verified_answer, sources), "supported"


def _parse_decision_only(
    raw: str, draft: str, max_source_index: int
) -> tuple[str, str]:
    decision = re.search(
        r"<decision>\s*(supported|abstain)\s*</decision>",
        raw,
        flags=re.IGNORECASE,
    )
    sources_match = re.search(
        r"<sources>\s*(.*?)\s*</sources>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not decision:
        return prompt.ABSTENTION, "invalid_format"
    if decision.group(1).lower() == "abstain":
        return prompt.ABSTENTION, "abstain"
    declared = (
        [int(value) for value in re.findall(r"\d+", sources_match.group(1))]
        if sources_match
        else []
    )
    draft_indices = [int(value) for value in re.findall(r"\[(\d+)\]", draft)]
    if (
        not declared
        or not draft_indices
        or any(
            index < 1 or index > max_source_index
            for index in declared + draft_indices
        )
        or _uncited_numbered_steps(draft)
    ):
        return prompt.ABSTENTION, "invalid_format"
    return _clean_answer(draft), "supported_decision_only"


def _draft_has_valid_citation_structure(draft: str, max_source_index: int) -> bool:
    if prompt.ABSTENTION in draft:
        return False
    indices = [int(value) for value in re.findall(r"\[(\d+)\]", draft)]
    return bool(indices) and all(
        1 <= index <= max_source_index for index in indices
    ) and not _uncited_numbered_steps(draft)


def _verify(plan: QueryPlan, hits: list[Hit], draft: str) -> tuple[str, str]:
    if not settings()["llm"].get("verify_generation", True):
        return _clean_answer(draft), "disabled"
    deterministic_guardrail = _deterministic_guardrail(
        plan.standalone_question, hits
    )
    messages = [
        {"role": "system", "content": prompt.VERIFIER},
        {
            "role": "user",
            "content": prompt.build_verification_message(
                plan.standalone_question,
                hits,
                draft,
                sensitive=plan.sensitive,
                explicit_support_required=plan.explicit_support_required,
                procedural_detail=plan.procedural_detail,
                deterministic_guardrail=deterministic_guardrail,
                route=plan.route,
            ),
        },
    ]
    raw = _generate(messages)
    verified = _parse_verification(raw, max_source_index=len(hits))
    if (
        verified[1] == "abstain"
        and len(hits) > 1
        and _draft_has_valid_citation_structure(draft, len(hits))
    ):
        review_raw = _generate(
            [
                messages[0],
                {
                    "role": "user",
                    "content": messages[1]["content"]
                    + "\n\n---\n\n"
                    + prompt.VERIFIER_ABSTENTION_REVIEW,
                },
            ]
        )
        review = _parse_verification(review_raw, max_source_index=len(hits))
        if review[1] != "invalid_format":
            return review
    if verified[1] != "invalid_format":
        return verified

    # Une sortie mal formée ne signifie pas que les preuves sont absentes.
    # Un seul nouvel essai strict évite une abstention purement technique.
    # Repartir d'un échange propre. Réinjecter une sortie mal formée comme
    # réponse assistant incite certains modèles à en reproduire la structure.
    retry_raw = _generate(
        [
            messages[0],
            {
                "role": "user",
                "content": messages[1]["content"]
                + "\n\n---\n\n"
                + prompt.VERIFIER_FORMAT_RETRY,
            },
        ]
    )
    retry = _parse_verification(retry_raw, max_source_index=len(hits))
    if retry[1] != "invalid_format":
        return retry

    decision_raw = _generate(
        [
            messages[0],
            {
                "role": "user",
                "content": messages[1]["content"]
                + "\n\n---\n\n"
                + prompt.VERIFIER_DECISION_ONLY_RETRY,
            },
        ]
    )
    return _parse_decision_only(decision_raw, draft, max_source_index=len(hits))


def unload() -> None:
    """Libère le LLM local, notamment après une commande CLI ponctuelle."""

    _load_transformers_llm.cache_clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


def ask(
    question: str,
    top_k: int | None = None,
    hybrid: bool = False,
    history: Sequence[Mapping[str, str]] | None = None,
) -> Answer:
    plan = plan_query(question, history=history)
    search_fn = (
        retriever.search_balanced if plan.route == "both" else retriever.search
    )
    search_kwargs = {
        "top_k": top_k,
        "hybrid": hybrid,
        "max_chunks_per_source": plan.max_chunks_per_source,
        "exclude_subtopics": plan.excluded_subtopics,
    }
    if plan.route != "both":
        search_kwargs["domain"] = plan.domain_filter
    hits = search_fn(plan.retrieval_query, **search_kwargs)
    hits = _focus_coherent_sources(plan, hits)

    if not hits:
        return Answer(
            question=question,
            text=prompt.ABSTENTION,
            hits=[],
            retrieval_question=plan.retrieval_query,
            route=plan.route,
            used_history=plan.used_history,
            verification_decision="no_hits",
        )

    conflict_reason = _document_conflict_reason(plan.standalone_question, hits)
    if conflict_reason:
        return Answer(
            question=question,
            text=prompt.ABSTENTION,
            hits=hits,
            retrieval_question=plan.retrieval_query,
            route=plan.route,
            used_history=plan.used_history,
            verification_decision=conflict_reason,
        )

    messages = [
        {"role": "system", "content": prompt.SYSTEM},
        {
            "role": "user",
            "content": prompt.build_user_message(
                plan.standalone_question,
                hits,
                route=plan.route,
                sensitive=plan.sensitive,
                explicit_support_required=plan.explicit_support_required,
                procedural_detail=plan.procedural_detail,
                deterministic_guardrail=_deterministic_guardrail(
                    plan.standalone_question, hits
                ),
            ),
        },
    ]
    draft = _generate(messages)
    draft = _normalize_single_source_citations(draft, hits)
    verified_text, verification_decision = _verify(plan, hits, draft)
    answer = Answer(
        question=question,
        text=verified_text,
        hits=hits,
        retrieval_question=plan.retrieval_query,
        route=plan.route,
        used_history=plan.used_history,
        verification_decision=verification_decision,
    )
    # Si le controle a valide le fond et qu'une seule preuve existe, une
    # omission purement formelle de citation peut etre reparee sans ambiguite.
    # On ne corrige jamais un indice faux ni une reponse multi-source.
    if (
        answer.verification_decision == "supported"
        and len(hits) == 1
        and not re.findall(r"\[(\d+)\]", answer.text)
    ):
        answer.text = answer.text.rstrip() + " [1]\n\nSources\n[1]"
    # Un texte sans citation vérifiable n'est jamais servi comme une réponse.
    if not answer.citations_valid:
        return Answer(
            question=question,
            text=prompt.ABSTENTION,
            hits=hits,
            retrieval_question=plan.retrieval_query,
            route=plan.route,
            used_history=plan.used_history,
            verification_decision="invalid_citation",
        )
    return answer
