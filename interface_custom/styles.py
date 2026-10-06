"""Feuille de style locale : aucun appel à une police ou un service externe."""

FONT_LINK = ""

HEAD = "'Cormorant Garamond', Georgia, 'Times New Roman', serif"
BODY = "'Lora', Georgia, 'Times New Roman', serif"


def build_css(p: dict) -> str:
    return f"""
<style>
:root {{
  --g-bg: {p['bg']};
  --g-surface: {p['surface']};
  --g-text: {p['text']};
  --g-text-soft: {p['text_soft']};
  --g-text-faint: {p['text_faint']};
  --g-rule: {p['rule']};
  --g-rule-soft: {p['rule_soft']};
  --g-accent: {p['accent']};
  --g-accent-deep: {p['accent_deep']};
  --g-head: {HEAD};
  --g-body: {BODY};
}}

/* ── chrome Streamlit ── */
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
[data-testid="stAppViewContainer"] {{ background: var(--g-bg); }}
[data-testid="stAppViewBlockContainer"], .block-container {{
  padding: 1.25rem 2.25rem 1rem !important; max-width: none !important;
}}
html, body, .stApp, [data-testid="stAppViewContainer"] * {{
  font-family: var(--g-body); color: var(--g-text);
}}
::selection {{ background: {p['tint']}; }}
a, a:visited {{ color: var(--g-accent-deep); text-underline-offset: 3px; }}
a:hover {{ color: var(--g-accent); }}
*:focus-visible {{ outline: 2px solid var(--g-accent) !important; outline-offset: 2px; }}

/* ── rail de gauche ── */
section[data-testid="stSidebar"] {{
  background: {p['panel']} !important;
  border-right: 1px solid {p['panel_border']};
  width: 262px !important; min-width: 262px !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding: 26px 22px; }}
section[data-testid="stSidebar"] * {{ color: {p['panel_text']}; }}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg {{
  stroke: {p['panel_muted']};
}}
.g-brand {{ text-align: center; padding-bottom: 20px; }}
.g-brand .g-mark {{
  font-family: var(--g-head); font-size: 30px; letter-spacing: .2em;
  color: {p['panel_text']}; line-height: 1.1;
}}
.g-brand .g-sub {{
  font-size: 9.5px; letter-spacing: .2em; text-transform: uppercase;
  color: {p['panel_accent']}; margin-top: 5px;
}}
.g-panel-rule {{ height: 1px; background: {p['panel_rule']}; margin: 0 0 18px; }}
.g-panel-kicker {{
  font-size: 10px; letter-spacing: .16em; text-transform: uppercase;
  color: {p['panel_muted']}; margin: 20px 0 10px;
}}
.g-panel-note {{ font-size: 11px; line-height: 1.55; color: {p['panel_muted']}; }}

/* navigation du rail : boutons Streamlit dépouillés */
section[data-testid="stSidebar"] .stButton > button {{
  width: 100%; text-align: left; justify-content: flex-start;
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  padding: 8px 0 !important; border-radius: 0 !important;
  font-family: var(--g-head) !important; font-size: 17px !important; font-weight: 400 !important;
  letter-spacing: .04em; color: {p['panel_text']} !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{ color: {p['panel_accent']} !important; }}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  color: {p['panel_accent']} !important;
}}
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {{ color: {p['panel_muted']}; }}
section[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {{
  background: {p['panel_accent']} !important;
}}

/* ── en-tête d'écran ── */
.g-head {{
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 4px 0 16px; border-bottom: 1px solid var(--g-rule); margin-bottom: 4px;
}}
.g-head h1 {{
  font-family: var(--g-head); font-size: 24px; font-weight: 600;
  margin: 0; color: var(--g-text); line-height: 1.2;
}}
.g-head .g-meta {{
  font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  color: {p['accent_deep']}; font-variant-numeric: tabular-nums;
}}
.g-foot {{
  display: flex; justify-content: space-between; padding: 12px 0 0;
  border-top: 1px solid var(--g-rule); margin-top: 18px;
  font-size: 11px; letter-spacing: .16em; text-transform: uppercase; color: var(--g-text-faint);
}}

/* ── état vide ── */
.g-empty {{ text-align: center; padding: 34px 0 6px; }}
.g-empty .g-ask {{
  font-family: var(--g-head); font-size: 38px; font-style: italic; font-weight: 400;
  line-height: 1.15; color: var(--g-text);
}}
.g-empty .g-lede {{
  font-size: 13.5px; color: var(--g-text-soft); margin: 8px auto 0; max-width: 560px;
  text-wrap: pretty;
}}
.g-col-kicker {{
  font-size: 10px; letter-spacing: .16em; text-transform: uppercase;
  color: var(--g-text-faint); padding-bottom: 9px; border-bottom: 1px solid var(--g-rule);
  margin-bottom: 4px;
}}
.g-sugg .stButton > button,
[class*="st-key-sg_"] .stButton > button {{
  width: 100%; text-align: left; justify-content: flex-start;
  background: transparent !important; border: 0 !important;
  border-bottom: 1px solid var(--g-rule-soft) !important; border-radius: 0 !important;
  box-shadow: none !important; padding: 11px 0 !important;
  font-family: var(--g-body) !important; font-size: 13.5px !important; font-weight: 400 !important;
  color: var(--g-text) !important; line-height: 1.4;
}}
.g-sugg .stButton > button:hover,
[class*="st-key-sg_"] .stButton > button:hover {{
  color: var(--g-accent-deep) !important; background: {p['tint']} !important;
}}

/* ── fil de conversation ── */
.g-turn {{ margin: 22px 0 0; }}
.g-q {{
  margin-left: auto; max-width: 80%;
  background: {p['bubble_bg']}; color: {p['bubble_text']};
  border: 1px solid {p['bubble_border']}; border-radius: 4px; padding: 14px 18px;
  font-family: var(--g-head); font-size: 19px; font-style: italic; line-height: 1.35;
}}
.g-a-label {{ display: flex; align-items: center; gap: 12px; margin: 22px 0 14px; }}
.g-a-label span {{
  font-size: 10px; letter-spacing: .2em; text-transform: uppercase; color: var(--g-text-faint);
}}
.g-a-label i {{ flex: 1; height: 1px; background: var(--g-rule); }}
.g-a {{ font-size: 15px; line-height: 1.62; text-align: justify; hyphens: auto; }}
.g-a p {{ margin: 0 0 13px; }}
.g-a ol, .g-a ul {{ margin: 0 0 13px; padding-left: 22px; text-align: left; }}
.g-a li {{ margin: 0 0 7px; }}
.g-a code {{
  font-size: 13px; background: {p['tint']}; padding: 1px 5px; border-radius: 3px;
}}
.g-a strong {{ font-weight: 600; }}
.g-abstain {{
  border-left: 3px solid var(--g-accent); background: {p['tint']};
  padding: 12px 16px; text-align: left;
}}
.g-err {{
  border: 1px solid var(--g-accent); border-radius: 4px; background: {p['tint']};
  padding: 14px 16px; font-size: 13.5px; color: var(--g-accent-deep); line-height: 1.5;
}}

/* ── panneau de sources ── */
.g-src-kicker {{
  font-size: 10px; letter-spacing: .18em; text-transform: uppercase;
  color: var(--g-text-faint); margin-bottom: 12px;
}}
.g-src {{
  border: 1px solid var(--g-rule); border-radius: 4px; padding: 13px 15px;
  margin-bottom: 12px; background: var(--g-surface);
}}
.g-src .g-n {{
  color: var(--g-accent-deep); font-size: 12px; font-variant-numeric: tabular-nums;
}}
.g-src .g-t {{ font-family: var(--g-head); font-size: 17px; font-weight: 600; line-height: 1.25; }}
.g-src .g-d {{ font-size: 12px; color: var(--g-text-soft); margin-top: 3px; }}
.g-src .g-s {{
  font-size: 11px; color: var(--g-text-faint); margin-top: 8px;
  font-variant-numeric: tabular-nums;
}}
.g-src .g-x {{
  font-size: 12px; color: var(--g-text-soft); line-height: 1.5; margin-top: 9px;
  padding-top: 9px; border-top: 1px solid var(--g-rule-soft);
}}
.g-src-note {{
  font-size: 11.5px; color: var(--g-text-faint); line-height: 1.5;
  padding-top: 12px; border-top: 1px solid var(--g-rule);
}}

/* ── registre ── */
.g-log {{
  display: grid; grid-template-columns: 84px 1fr 96px; align-items: center;
  font-size: 14px; padding: 15px 0; border-bottom: 1px solid var(--g-rule-soft);
}}
.g-log .g-time {{ font-variant-numeric: tabular-nums; color: var(--g-text-faint); }}
.g-log .g-cnt {{
  text-align: right; font-variant-numeric: tabular-nums; color: var(--g-accent-deep);
}}
.g-log-head {{
  display: grid; grid-template-columns: 84px 1fr 96px; padding: 14px 0;
  border-bottom: 1px solid var(--g-rule); font-size: 10px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--g-text-faint);
}}
.g-log-head div:last-child, .g-log .g-cnt {{ text-align: right; }}

/* ── zone de saisie + sonnette ── */
div[data-testid="stForm"] {{
  border: 0 !important; padding: 14px 0 0 !important;
  border-top: 1px solid var(--g-rule) !important; margin-top: 16px;
}}
div[data-testid="stForm"] textarea {{
  background: {p['field_bg']} !important; border: 1px solid {p['field_border']} !important;
  border-radius: 4px !important; color: var(--g-text) !important;
  font-family: var(--g-body) !important; font-size: 14.5px !important; line-height: 1.5;
  box-shadow: none !important; padding: 12px 16px !important;
}}
div[data-testid="stForm"] textarea::placeholder {{ color: var(--g-text-faint) !important; }}
div[data-testid="stForm"] textarea:focus {{ border-color: var(--g-accent) !important; }}

/* la sonnette de comptoir remplace le bouton d'envoi */
div[data-testid="stForm"] button[kind="primaryFormSubmit"],
div[data-testid="stForm"] button[kind="secondaryFormSubmit"],
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {{
  position: relative; width: 64px !important; height: 58px !important; min-height: 58px !important;
  padding: 0 !important; border: 0 !important; box-shadow: none !important;
  background-color: transparent !important; color: transparent !important;
  background-image:
    radial-gradient(circle at 50% 13px, {p['bell_hi']} 0 2.2px, transparent 2.6px),
    linear-gradient({p['bell_dome']}, {p['bell_dome']});
  background-size: 100% 100%, 2px 6px;
  background-position: 0 0, 50% 15px;
  background-repeat: no-repeat;
  transition: filter .15s ease, transform .1s ease;
}}
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button::before {{
  content: ""; position: absolute; left: 50%; top: 21px; transform: translateX(-50%);
  width: 32px; height: 17px; border-radius: 16px 16px 4px 4px;
  background: linear-gradient(155deg, {p['bell_hi']} 0%, {p['bell_dome']} 52%, {p['bell_base']} 100%);
}}
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button::after {{
  content: ""; position: absolute; left: 50%; top: 41px; transform: translateX(-50%);
  width: 40px; height: 5px; border-radius: 2px;
  background: linear-gradient(180deg, {p['bell_dome']}, {p['bell_base']});
  box-shadow: 0 7px 0 -3px {p['bell_shadow']};
}}
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {{
  filter: brightness(1.08);
}}
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:active {{
  transform: translateY(1px);
}}

/* divers */
[data-testid="stSpinner"] p {{
  font-family: var(--g-head) !important; font-size: 16px; font-style: italic;
  color: var(--g-text-soft) !important;
}}
hr {{ border: 0; border-top: 1px solid var(--g-rule); margin: 18px 0; }}
</style>
"""
