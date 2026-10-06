"""Point d'entrée de l'interface Reception Assistant.

    streamlit run interface_custom/app.py --server.address 127.0.0.1 --server.port 8501

À lancer depuis la racine du projet. Ce fichier ne touche jamais à src/, config/,
knowledge/ ni data/ : il appelle uniquement src.generation.answer.ask() via
interface_custom.rag_bridge (adapteur corrigé, préserver tel quel).
"""

from __future__ import annotations

import html
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from interface_custom import config, styles  # noqa: E402
from interface_custom.rag_bridge import EngineError, ask as rag_ask  # noqa: E402

P = config.palette()

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(styles.FONT_LINK, unsafe_allow_html=True)
st.markdown(styles.build_css(P), unsafe_allow_html=True)


# ─────────────────────────── état ───────────────────────────
def _init_state() -> None:
    st.session_state.setdefault("messages", [])      # {role, content, sources, elapsed, at}
    st.session_state.setdefault("page", "question")
    st.session_state.setdefault("top_k", config.TOP_K_DEFAULT)
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("log", [])           # {at, question, n_sources}


_init_state()


# ─────────────────────── mise en forme ──────────────────────
def to_html(text: str) -> str:
    """Markdown minimal → HTML, sans dépendance externe."""
    esc = html.escape(text or "")
    esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc, flags=re.S)
    esc = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", esc, flags=re.S)
    esc = re.sub(r"`([^`]+?)`", r"<code>\1</code>", esc)

    out, buf, mode = [], [], None

    def flush() -> None:
        nonlocal buf, mode
        if not buf:
            return
        if mode == "ol":
            out.append("<ol>" + "".join(f"<li>{b}</li>" for b in buf) + "</ol>")
        elif mode == "ul":
            out.append("<ul>" + "".join(f"<li>{b}</li>" for b in buf) + "</ul>")
        else:
            out.append("<p>" + "<br>".join(buf) + "</p>")
        buf, mode = [], None

    for line in esc.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        ordered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        bulleted = re.match(r"^[-•–]\s+(.*)$", stripped)
        if ordered:
            if mode != "ol":
                flush()
                mode = "ol"
            buf.append(ordered.group(1))
        elif bulleted:
            if mode != "ul":
                flush()
                mode = "ul"
            buf.append(bulleted.group(1))
        else:
            if mode in ("ol", "ul"):
                flush()
            mode = "p"
            buf.append(stripped)
    flush()
    return "".join(out)


def fr_score(v: float | None) -> str:
    if v is None:
        return ""
    return f"Pertinence {v:.2f}".replace(".", ",")


def media_path(value: str | None) -> Path | None:
    """Dossier de captures : le chemin peut traverser un lien symbolique
    (data → /data), sa résolution est donc autorisée hors de la racine strictement."""

    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    allowed_roots = (ROOT.resolve(), (ROOT / "data").resolve())
    if not any(
        candidate == base or candidate.is_relative_to(base) for base in allowed_roots
    ):
        return None
    return candidate if candidate.is_file() else None


# ─────────────────────────── rail ───────────────────────────
NAV = [("question", "Nouvelle question"), ("registre", "Registre"), ("apropos", "À propos")]

with st.sidebar:
    st.markdown(
        f'<div class="g-brand"><div class="g-mark">{html.escape(config.APP_TITLE.upper())}</div>'
        f'<div class="g-sub">{html.escape(config.APP_SUBTITLE)}</div></div>'
        '<div class="g-panel-rule"></div>',
        unsafe_allow_html=True,
    )
    for key, label in NAV:
        if st.button(
            label,
            key=f"nav_{key}",
            type="primary" if st.session_state.page == key else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div class="g-panel-rule" style="margin-top:20px"></div>'
                '<div class="g-panel-kicker">Sources consultées</div>', unsafe_allow_html=True)
    st.session_state.top_k = st.slider(
        "top_k", config.TOP_K_MIN, config.TOP_K_MAX, st.session_state.top_k,
        label_visibility="collapsed",
    )

    if st.session_state.messages:
        st.markdown('<div class="g-panel-kicker">Session</div>', unsafe_allow_html=True)
        for m in [m for m in st.session_state.messages if m["role"] == "user"][-4:]:
            st.markdown(
                f'<div class="g-panel-note" style="margin-bottom:8px">'
                f'{html.escape(m["content"][:64])}{"…" if len(m["content"]) > 64 else ""}</div>',
                unsafe_allow_html=True,
            )
        if st.button("Effacer la session", key="clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        '<div class="g-panel-rule" style="margin-top:20px"></div>'
        f'<div class="g-panel-note">{len(st.session_state.log)} consultations · '
        f'{datetime.now():%d/%m/%Y}</div>',
        unsafe_allow_html=True,
    )


# ────────────────────── envoi d'une question ─────────────────────
def submit(question: str) -> None:
    question = (question or "").strip()
    if not question:
        return
    st.session_state.messages.append(
        {"role": "user", "content": question, "at": datetime.now()}
    )
    started = time.perf_counter()
    try:
        with st.spinner("Consultation du registre…"):
            reply = rag_ask(
                question,
                top_k=st.session_state.top_k,
                messages=st.session_state.messages[:-1],
                history_max_turns=config.HISTORY_MAX_TURNS,
            )
        elapsed = time.perf_counter() - started
        sources = [
            s for s in reply.sources
            if s.score is None or s.score >= config.SCORE_DISPLAY_FLOOR
        ]
        st.session_state.messages.append({
            "role": "assistant", "content": reply.text, "sources": sources,
            "elapsed": elapsed, "at": datetime.now(),
        })
        st.session_state.log.append({
            "at": datetime.now(), "question": question, "n_sources": len(sources),
        })
    except EngineError as exc:
        st.session_state.messages.append({
            "role": "assistant", "content": "", "error": str(exc),
            "sources": [], "elapsed": time.perf_counter() - started, "at": datetime.now(),
        })


if st.session_state.pending:
    q = st.session_state.pending
    st.session_state.pending = None
    submit(q)


# ─────────────────────────── écrans ──────────────────────────
def header(title: str, meta: str = "") -> None:
    st.markdown(
        f'<div class="g-head"><h1>{html.escape(title)}</h1>'
        f'<div class="g-meta">{html.escape(meta)}</div></div>',
        unsafe_allow_html=True,
    )


def render_sources(sources: list) -> None:
    st.markdown('<div class="g-src-kicker">Sources</div>', unsafe_allow_html=True)
    if not sources:
        st.markdown(
            '<div class="g-src-note">Le moteur n’a renvoyé aucune source pour cette réponse.</div>',
            unsafe_allow_html=True,
        )
        return
    for s in sources:
        num = s.citation_index
        detail = f'<div class="g-d">{html.escape(s.detail)}</div>' if s.detail else ""
        score = f'<div class="g-s">{fr_score(s.score)}</div>' if s.score is not None else ""
        snippet = f'<div class="g-x">{html.escape(s.snippet)}…</div>' if s.snippet else ""
        st.markdown(
            f'<div class="g-src"><span class="g-n">{num}</span>'
            f'<div class="g-t">{html.escape(s.title)}</div>{detail}{score}{snippet}</div>',
            unsafe_allow_html=True,
        )
        screenshot = next(
            (path for value in s.images if (path := media_path(value))),
            None,
        )
        if screenshot:
            caption = f"Source [{num}]"
            if s.timestamp_start is not None:
                mm, ss = divmod(int(s.timestamp_start), 60)
                caption += f" · {mm:02d}:{ss:02d}"
            st.image(str(screenshot), caption=caption, use_container_width=True)


def screen_question() -> None:
    msgs = st.session_state.messages
    if msgs:
        last_user = next(
            (m["content"] for m in reversed(msgs) if m["role"] == "user"),
            "Consultation",
        )
        last = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
        meta = ""
        if last and not last.get("error"):
            n = len(last.get("sources", []))
            secs = f'{last.get("elapsed", 0):.1f}'.replace(".", ",")
            meta = f'{n} source{"s" if n > 1 else ""} · {secs} s'
        header(last_user[:72] + ("…" if len(last_user) > 72 else ""), meta)
    else:
        header("Nouvelle question")

    body, aside = st.columns([1, 0.38], gap="large")

    with body:
        if not msgs:
            st.markdown(
                '<div class="g-empty"><div class="g-ask">Que puis-je consulter pour vous&nbsp;?</div>'
                '<div class="g-lede">Chaque réponse cite les documents dont elle provient. '
                'À défaut de source, l’assistant le dit.</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="g-sugg">', unsafe_allow_html=True)
            cols = st.columns(len(config.SUGGESTIONS), gap="large")
            for col, (heading, items) in zip(cols, config.SUGGESTIONS.items()):
                with col:
                    st.markdown(
                        f'<div class="g-col-kicker">{html.escape(heading)}</div>',
                        unsafe_allow_html=True,
                    )
                    for j, item in enumerate(items):
                        if st.button(item, key=f"sg_{heading}_{j}", use_container_width=True):
                            st.session_state.pending = item
                            st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            for m in msgs:
                if m["role"] == "user":
                    st.markdown(
                        f'<div class="g-turn"><div class="g-q">{html.escape(m["content"])}</div></div>',
                        unsafe_allow_html=True,
                    )
                elif m.get("error"):
                    st.markdown(
                        f'<div class="g-turn"><div class="g-err">Le moteur n’a pas répondu. '
                        f'{html.escape(m["error"])}</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="g-a-label"><span>Réponse de l’assistant</span><i></i></div>'
                        f'<div class="g-a">{to_html(m["content"])}</div>',
                        unsafe_allow_html=True,
                    )

        with st.form("gustave_ask", clear_on_submit=True):
            field, bell = st.columns([1, 0.075], gap="small")
            with field:
                text = st.text_area(
                    "question",
                    height=78,
                    placeholder="Posez votre question, puis sonnez…" if not msgs
                    else "Préciser la question, puis sonnez…",
                    label_visibility="collapsed",
                )
            with bell:
                sent = st.form_submit_button(" ", use_container_width=False)
        if sent:
            st.session_state.pending = text
            st.rerun()

        st.markdown(
            f'<div class="g-foot"><div>{html.escape(config.FOOTER_LEFT)}</div>'
            f'<div>{html.escape(config.FOOTER_RIGHT)}</div></div>',
            unsafe_allow_html=True,
        )

    with aside:
        last = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
        if last and not last.get("error"):
            render_sources(last.get("sources", []))


def screen_registre() -> None:
    header("Registre", f"{len(st.session_state.log)} consultations")
    if not st.session_state.log:
        st.markdown(
            '<div class="g-src-note" style="border:0;padding-top:24px">'
            'Aucune consultation dans cette session.</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        '<div class="g-log-head"><div>Heure</div><div>Question</div><div>Sources</div></div>',
        unsafe_allow_html=True,
    )
    for entry in reversed(st.session_state.log):
        st.markdown(
            f'<div class="g-log"><div class="g-time">{entry["at"]:%H:%M}</div>'
            f'<div>{html.escape(entry["question"])}</div>'
            f'<div class="g-cnt">{entry["n_sources"] or "—"}</div></div>',
            unsafe_allow_html=True,
        )


def screen_apropos() -> None:
    header("À propos")
    st.markdown(
        '<div class="g-a" style="max-width:720px;padding-top:20px">'
        "<p>Interface de consultation du moteur RAG existant. Les réponses sont produites "
        "par <code>src.generation.answer.ask()</code> ; cette couche n’effectue ni "
        "reformulation, ni routage, ni post-traitement du contenu.</p>"
        f"<p>Nombre de passages transmis à chaque appel&nbsp;: "
        f"<strong>{st.session_state.top_k}</strong>. Historique transmis&nbsp;: "
        f"<strong>{config.HISTORY_MAX_TURNS} tours</strong> au plus.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


{"question": screen_question, "registre": screen_registre, "apropos": screen_apropos}[
    st.session_state.page
]()
