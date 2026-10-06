"""Réglages de l'interface. Aucun paramètre du moteur RAG ici."""

# Palette active : "papier" (2a), "laque" (2b) ou "prune" (2c)
PALETTE = "papier"

APP_TITLE = "Reception Assistant"
APP_SUBTITLE = "Assistant de réception hôtelière"
FOOTER_LEFT = "Reception Assistant"
FOOTER_RIGHT = "Version d’évaluation"

# Nombre de passages transmis à ask(top_k=…). Réglable dans le rail.
TOP_K_DEFAULT = 5
TOP_K_MIN = 1
TOP_K_MAX = 12

# Historique transmis à ask(history=…) : nombre de tours (logique = paires).
HISTORY_MAX_TURNS = 6

# Seuil cosmétique sous lequel un passage n'est plus affiché (le moteur reçoit toujours top_k).
SCORE_DISPLAY_FLOOR = 0.0

# Suggestions de l'écran d'accueil. Purement d'affichage.
SUGGESTIONS = {
    "Réception": [
        "Quelle est la procédure d’enregistrement après 23 h ?",
        "Quelles sont les conditions d’accueil des animaux ?",
        "Dans quels cas un surclassement est-il accordé ?",
    ],
    "Direction": [
        "Quel est le tarif Deluxe en haute saison ?",
        "Quelles pénalités pour une annulation de groupe à J-7 ?",
        "Comment facturer un no-show garanti ?",
    ],
    "Exploitation": [
        "Qui prévenir en cas de panne d’ascenseur la nuit ?",
        "Quels sont les horaires du room service le dimanche ?",
        "Quelles obligations pour le registre HACCP ?",
    ],
}

PALETTES = {
    # 2a — papier et laiton
    "papier": {
        "bg": "#f3f2f2",
        "surface": "#f9f8f8",
        "panel": "#eae9e9",
        "panel_text": "#444141",
        "panel_muted": "#7d7979",
        "panel_accent": "#7d5411",
        "panel_rule": "rgba(32,31,29,.16)",
        "panel_border": "rgba(32,31,29,.16)",
        "text": "#201f1d",
        "text_soft": "#605d5d",
        "text_faint": "#9b9797",
        "rule": "rgba(32,31,29,.16)",
        "rule_soft": "rgba(32,31,29,.09)",
        "accent": "#b68235",
        "accent_deep": "#7d5411",
        "field_border": "#605d5d",
        "field_bg": "#ffffff",
        "bubble_bg": "transparent",
        "bubble_text": "#7d5411",
        "bubble_border": "#b68235",
        "tint": "rgba(182,130,53,.10)",
        "bell_hi": "#dcc9a4",
        "bell_dome": "#b68235",
        "bell_base": "#8a6520",
        "bell_shadow": "rgba(32,31,29,.14)",
    },
    # 2b — laque rouge, or et blanc
    "laque": {
        "bg": "#fdfbf7",
        "surface": "#ffffff",
        "panel": "#7d1015",
        "panel_text": "#f4e6d4",
        "panel_muted": "#e0b5b0",
        "panel_accent": "#d9b45f",
        "panel_rule": "rgba(217,180,95,.45)",
        "panel_border": "#7d1015",
        "text": "#241c1c",
        "text_soft": "#6b5b58",
        "text_faint": "#a8968f",
        "rule": "#e6d3a8",
        "rule_soft": "rgba(125,16,21,.10)",
        "accent": "#7d1015",
        "accent_deep": "#8a6520",
        "field_border": "#b8a58f",
        "field_bg": "#ffffff",
        "bubble_bg": "#7d1015",
        "bubble_text": "#f8f0df",
        "bubble_border": "#7d1015",
        "tint": "rgba(217,180,95,.14)",
        "bell_hi": "#f0dba6",
        "bell_dome": "#d9b45f",
        "bell_base": "#c39a4e",
        "bell_shadow": "rgba(125,16,21,.16)",
    },
    # 2c — violet et or
    "prune": {
        "bg": "#f5f2f6",
        "surface": "#fbfafc",
        "panel": "#3b2a52",
        "panel_text": "#e6dcee",
        "panel_muted": "#b9a8c9",
        "panel_accent": "#cfa95e",
        "panel_rule": "rgba(207,169,94,.40)",
        "panel_border": "#3b2a52",
        "text": "#241f2b",
        "text_soft": "#605a6b",
        "text_faint": "#9a91a6",
        "rule": "rgba(59,42,82,.20)",
        "rule_soft": "rgba(59,42,82,.10)",
        "accent": "#3b2a52",
        "accent_deep": "#7d5411",
        "field_border": "#9a91a6",
        "field_bg": "#ffffff",
        "bubble_bg": "#3b2a52",
        "bubble_text": "#f0eaf5",
        "bubble_border": "#3b2a52",
        "tint": "rgba(92,63,133,.10)",
        "bell_hi": "#ecd6a4",
        "bell_dome": "#cfa95e",
        "bell_base": "#b68235",
        "bell_shadow": "rgba(59,42,82,.16)",
    },
}


def palette() -> dict:
    return PALETTES.get(PALETTE, PALETTES["papier"])
