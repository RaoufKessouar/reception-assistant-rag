"""Prépare une question avant le retrieval, sans appel LLM supplémentaire."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

Route = Literal["software", "hotel", "both"]


@dataclass(frozen=True)
class QueryPlan:
    original_question: str
    standalone_question: str
    retrieval_query: str
    route: Route
    domain_filter: Literal["software", "hotel"] | None
    used_history: bool
    sensitive: bool
    explicit_support_required: bool
    procedural_detail: bool
    max_chunks_per_source: int | None
    excluded_subtopics: tuple[str, ...]
    aliases: tuple[str, ...]


_CONTEXT_REFERENCES = re.compile(
    r"\b(?:ces?\s+modific\w*|ces?\s+changements?|cette\s+operation|"
    r"cela|ca|le\s+faire|la\s+faire|les\s+faire|faire\s+ca|faire\s+cela|"
    r"comme\s+avant|comme\s+indique|pareil|et\s+ensuite)\b"
)

_HOTEL_SOFTWARE_TERMS = (
    "generic_connector",
    "logiciel de gestion hôtelière",
    "software",
    "logiciel",
    "bouton",
    "menu",
    "section",
    "onglet",
    "ecran",
    "cliquer",
    "formulaire",
    "planning",
    "software",
    "button",
    "screen",
    "tab",
    "hotel management software",
)

_HOTEL_TERMS = (
    "hotel",
    "reception",
    "client",
    "chambre",
    "sejour",
    "caution",
    "responsable",
    "bruit",
    "climatisation",
    "visiteur",
    "passant",
    "agress*",
    "ivresse",
    "urgence",
    "police",
)

_POLICY_TERMS = (
    "accept*",
    "autoris*",
    "a le droit",
    "avons nous le droit",
    "peut*",
    "pouv*",
    "est ce que on peut",
    "combien",
    "tarif",
    "politique",
    "condition",
    "droit",
    "autorisation",
)

_SENSITIVE_TERMS = (
    "paiement",
    "payer",
    "carte*",
    "debit",
    "preautorisation",
    "remboursement",
    "facturer",
    "facturation",
    "caution",
    "garantie",
    "agress*",
    "danger",
    "urgence",
    "police",
    "securite",
)

_EXPLICIT_SUPPORT_TERMS = (
    "droit",
    "autoris*",
    "accept*",
    "peut*",
    "pouv*",
    "est ce que on peut",
    "avant l arrivee",
    "apres l arrivee",
    "avant le check in",
    "sans accord",
    "carte jointe",
    "carte enregistree",
)

_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bcheck[ -]?in\b"),
        "effectuer un check-in reservation",
    ),
    (
        re.compile(r"\bcheck[ -]?out\b"),
        "effectuer un check-out reservation facturation integralite",
    ),
    (
        re.compile(r"\bliens?\s+(?:de\s+)?paiement\b"),
        "creer un lien securise paiement 3D Secure type de lien email client",
    ),
    (
        re.compile(
            r"\b(?:ajouter|ajout\w*|enregistrer|facturer)\s+(?:des?|le|la|un|une)?\s*"
            r"(?:extras?|prestations?\s+complementaires?|supplements?)\b"
        ),
        "bouton bleu ajout de prestation complementaire choisir date extra "
        "quantite ajouter les extras selectionnes",
    ),
    (
        re.compile(r"\b(?:cb|cartes?\s+bleues?)\b"),
        "carte bancaire paiement par carte moyen de paiement accepte "
        "accepter le paiement par carte ou en especes",
    ),
    (
        re.compile(r"\b(?:clim(?:atisation)?|air\s+conditionne)\b"),
        "climatisation panne technique changer le client de chambre "
        "deplacer une reservation d'une chambre a une autre en cours de "
        "sejour planning attribution de chambre",
    ),
    (
        re.compile(r"\bcheques?\s+vacances?\b"),
        "cheque-vacances ANCV moyen de paiement",
    ),
    (
        re.compile(r"\b(?:passant|personne\s+de\s+la\s+rue)\b"),
        "personne exterieure visiteur exterieur incident de securite",
    ),
    (
        re.compile(r"\b(?:ivre|bourre|ivresse)\b"),
        "ivresse comportement agressif risque pour la securite",
    ),
    (
        re.compile(r"\brajouter\b"),
        "ajouter",
    ),
)


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_value.lower()).strip()


def _contains_any(value: str, terms: Sequence[str]) -> bool:
    """Cherche des mots entiers; ``*`` autorise uniquement un suffixe lexical."""

    for term in terms:
        is_prefix = term.endswith("*")
        literal = re.escape(term[:-1] if is_prefix else term)
        end = "" if is_prefix else r"(?!\w)"
        if re.search(rf"(?<!\w){literal}{end}", value):
            return True
    return False


def _last_user_question(history: Sequence[Mapping[str, str]] | None) -> str | None:
    for turn in reversed(history or []):
        if turn.get("role") == "user" and turn.get("content", "").strip():
            return turn["content"].strip()
    return None


def _route(folded: str, *, uses_history: bool = False) -> Route:
    has_software = _contains_any(folded, _HOTEL_SOFTWARE_TERMS)
    has_hotel = _contains_any(folded, _HOTEL_TERMS)
    asks_policy = _contains_any(folded, _POLICY_TERMS)
    asks_combined_workflow = bool(re.search(r"\b(?:et|puis)\s+comment\b", folded))

    if asks_policy and asks_combined_workflow:
        return "both"
    if has_software and has_hotel and (
        asks_policy or uses_history or asks_combined_workflow
    ):
        return "both"
    if has_software and asks_policy:
        return "both"
    if has_software:
        return "software"
    if has_hotel or asks_policy:
        return "hotel"
    return "both"


def plan_query(
    question: str,
    history: Sequence[Mapping[str, str]] | None = None,
) -> QueryPlan:
    """Construit une question autonome, enrichie et routée."""

    original = re.sub(r"\s+", " ", question).strip()
    folded_original = _fold(original)
    previous = _last_user_question(history)
    uses_history = bool(previous and _CONTEXT_REFERENCES.search(folded_original))
    standalone = (
        f"Demande précédente : {previous}\nDemande actuelle : {original}"
        if uses_history
        else original
    )
    folded = _fold(standalone)

    aliases = tuple(
        expansion for pattern, expansion in _ALIASES if pattern.search(folded)
    )
    retrieval_query = standalone
    if aliases:
        retrieval_query += "\nTermes équivalents utiles : " + "; ".join(aliases)

    route = _route(folded, uses_history=uses_history)
    sensitive = _contains_any(folded, _SENSITIVE_TERMS)
    explicit_support_required = sensitive and _contains_any(
        folded, _EXPLICIT_SUPPORT_TERMS
    )
    asks_how = bool(re.search(r"\bcomment\b", folded)) and not bool(
        re.search(r"\b(?:difference|qu est ce)\b", folded)
    )
    procedural_detail = route in {"software", "both"} and (
        asks_how
        or any(
            term in folded
            for term in (
                "bouton",
                "section",
                "onglet",
                "menu",
                "etape",
                "guider",
            )
        )
    )
    external_incident = bool(
        re.search(r"\b(?:passant|personne\s+exterieure|personne\s+de\s+la\s+rue)\b", folded)
    )
    mentions_noise = bool(re.search(r"\b(?:bruit|nuisance\s+sonore)\b", folded))
    excluded_subtopics = (
        ("noise",) if external_incident and not mentions_noise else ()
    )
    return QueryPlan(
        original_question=original,
        standalone_question=standalone,
        retrieval_query=retrieval_query,
        route=route,
        domain_filter=None if route == "both" else route,
        used_history=uses_history,
        sensitive=sensitive,
        explicit_support_required=explicit_support_required,
        procedural_detail=procedural_detail,
        # Une procédure multi-action peut nécessiter plusieurs étapes voisines
        # du même tutoriel. Le top-k limite déjà le contexte total.
        max_chunks_per_source=5 if procedural_detail else None,
        excluded_subtopics=excluded_subtopics,
        aliases=aliases,
    )
