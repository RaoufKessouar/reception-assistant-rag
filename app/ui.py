"""Interface Streamlit de référence pour Reception Assistant.

Lancement serveur : bash scripts/run_ui_server.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import path as project_path
from src.config import settings
from src.generation import answer as gen


def _timestamp(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remaining:02d}"


def _evidence_caption(result: gen.Answer) -> str:
    cited = result.cited_hits()
    if result.abstained:
        return "Preuve : aucune réponse servie — abstention de sécurité"
    if not cited or not result.citations_valid:
        return "Preuve : réponse bloquée — citation absente ou invalide"
    threshold = settings()["qdrant"].get("min_score")
    score = max(hit.score for hit in cited)
    suffix = f" · seuil dense {threshold:.2f}" if threshold is not None else ""
    return (
        f"Preuve : {len(cited)} source(s) vérifiée(s) citée(s)"
        f" · meilleur score {score:.3f}{suffix} · routage {result.route.upper()}"
    )


def _first_existing_path(values: list[str]) -> Path | None:
    for value in values:
        candidate = project_path(value)
        if candidate.exists():
            return candidate
    return None


def _render_source(hit) -> None:
    doc = hit.document
    icon = {"video": "🎥", "internal_sop": "📋", "qa": "💬"}.get(
        doc.source_type, "📄"
    )
    timestamp = _timestamp(doc.timestamp_start)
    time_label = f" · {timestamp}" if timestamp else ""
    label = f"{icon} {doc.citation()}{time_label} · score {hit.score:.3f}"
    with st.expander(label):
        st.caption(
            f"Domaine : {doc.domain} · Autorité : {doc.authority} · "
            f"Type : {doc.source_type}"
        )
        st.markdown(doc.content)

        screenshot_paths = []
        if doc.screenshot:
            screenshot_paths.append(doc.screenshot)
        screenshot_paths.extend(
            value for value in doc.screenshots if value not in screenshot_paths
        )
        screenshot = _first_existing_path(screenshot_paths)
        if screenshot:
            st.image(str(screenshot), caption=f"Preuve visuelle · {timestamp or 'capture'}")

        video = project_path(doc.source_path) if doc.source_path else None
        if video and video.exists() and doc.timestamp_start is not None:
            st.video(str(video), start_time=int(doc.timestamp_start))


def _render_answer(result: gen.Answer) -> None:
    if result.abstained:
        st.warning(result.text)
    else:
        st.markdown(result.text)
    st.caption(_evidence_caption(result))

    cited = result.cited_hits()
    if cited:
        st.markdown("#### Sources")
        for hit in cited:
            _render_source(hit)


st.set_page_config(page_title="Reception Assistant", page_icon="🛎️", layout="wide")
st.title("🛎️ Reception Assistant")
st.caption(
    "Répond uniquement à partir des procédures vérifiées. "
    "Si les preuves sont insuffisantes, il s’abstient."
)

with st.sidebar:
    st.subheader("Recherche")
    top_k = st.slider("Sources candidates", 3, 10, 5)
    st.caption("Mode actif : BGE-M3 dense · diversification par source")
    st.info(
        "La recherche hybride a été testée puis désactivée : elle réduisait "
        "le Recall@5 et le MRR."
    )
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            _render_answer(message["answer"])

question = st.chat_input("Posez une question logiciel de gestion hôtelière ou réception hôtelière…")
if question:
    history = []
    for message in st.session_state.messages:
        content = (
            message["content"]
            if message["role"] == "user"
            else message["answer"].text
        )
        history.append({"role": message["role"], "content": content})
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des preuves et vérification de la réponse…"):
            result = gen.ask(
                question,
                top_k=top_k,
                hybrid=False,
                history=history,
            )
        _render_answer(result)
    st.session_state.messages.append({"role": "assistant", "answer": result})
