"""Paquet HTML autonome pour une validation humaine rapide et stratifiée."""

from __future__ import annotations

import hashlib
import html
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .....config import path
from .....schema import ProceduralStep
from .common import atomic_write_json, read_json

DEFAULT_QUOTAS = {"high": 24, "medium": 28, "low": 8}


def _asset_path(value: str, project_root: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else project_root / candidate


def _load_steps(videos_root: Path) -> dict[tuple[str, int], ProceduralStep]:
    steps: dict[tuple[str, int], ProceduralStep] = {}
    for file in sorted(videos_root.glob("*/steps.json")):
        payload = read_json(file)
        rows = payload.get("steps", []) if isinstance(payload, dict) else payload
        for row in rows:
            step = ProceduralStep(**row)
            steps[(step.video_id, step.step_number)] = step
    return steps


def _suggestion_rows(file: Path) -> list[dict[str, Any]]:
    if not file.is_file():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_suggestions(file: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = _suggestion_rows(file)
    return {
        (str(row["video_id"]), int(row["step_number"])): row
        for row in rows
        if row.get("status") == "proposal_only"
    }


def _diverse_sample(
    rows: list[dict[str, Any]], quota: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Maximise d'abord la diversité des vidéos, puis complète le quota."""

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[str(row["video_id"])].append(row)
    video_ids = sorted(by_video)
    rng.shuffle(video_ids)
    for items in by_video.values():
        items.sort(key=lambda item: int(item["step_number"]))
        rng.shuffle(items)

    selected: list[dict[str, Any]] = []
    while len(selected) < quota:
        progressed = False
        for video_id in video_ids:
            if by_video[video_id] and len(selected) < quota:
                selected.append(by_video[video_id].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != quota:
        raise ValueError(
            f"Échantillon insuffisant : {len(selected)}/{quota} éléments disponibles"
        )
    return selected


def select_sample(
    report: dict[str, Any],
    suggestions: dict[tuple[str, int], dict[str, Any]],
    *,
    quotas: dict[str, int] | None = None,
    seed: int = 20260825,
) -> list[dict[str, Any]]:
    quotas = quotas or DEFAULT_QUOTAS
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    findings = [
        row
        for row in report.get("step_findings", [])
        if row.get("status") == "review_required"
    ]
    for risk in ("high", "medium", "low"):
        eligible = []
        for row in findings:
            if row.get("risk") != risk:
                continue
            key = (str(row["video_id"]), int(row["step_number"]))
            if risk != "low" and key not in suggestions:
                continue
            eligible.append(row)
        selected.extend(_diverse_sample(eligible, quotas[risk], rng))
    rng.shuffle(selected)
    return selected


def _proposal_hash(proposal: dict[str, Any]) -> str:
    payload = json.dumps(
        proposal, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
            image.save(destination, "JPEG", quality=88, optimize=True)
    except (OSError, ValueError):
        shutil.copy2(source, destination.with_suffix(source.suffix.lower()))


def _evidence(
    proposal: dict[str, Any],
    *,
    item_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> list[dict[str, str]]:
    sources = [
        ("Avant", proposal.get("screenshot_before")),
        *(
            (f"Support {index}", value)
            for index, value in enumerate(
                proposal.get("supporting_screenshots") or [], 1
            )
        ),
        ("Après", proposal.get("screenshot_after")),
    ]
    evidence = []
    seen: set[str] = set()
    for index, (role, value) in enumerate(sources, 1):
        if not value or str(value) in seen:
            continue
        seen.add(str(value))
        source = _asset_path(str(value), project_root)
        if not source.is_file():
            raise FileNotFoundError(f"Capture absente : {source}")
        destination = item_dir / f"{index:02d}.jpg"
        _copy_image(source, destination)
        if not destination.exists():
            destination = destination.with_suffix(source.suffix.lower())
        evidence.append(
            {
                "role": role,
                "file": destination.relative_to(output_dir).as_posix(),
                "source": str(value),
            }
        )
    return evidence


def _text(value: Any) -> str:
    return html.escape(str(value or "—"))


def _render_html(
    batch_id: str,
    items: list[dict[str, Any]],
    *,
    review_type: str = "independent_stratified_sample",
) -> str:
    if not items:
        raise ValueError("Le paquet de revue ne peut pas être vide")
    lot_count = max(int(item["lot"]) for item in items)
    total = len(items)
    cards = []
    for item in items:
        images = "".join(
            f'<figure><a href="{html.escape(image["file"])}" target="_blank">'
            f'<img loading="lazy" src="{html.escape(image["file"])}"></a>'
            f'<figcaption>{_text(image["role"])}</figcaption></figure>'
            for image in item["evidence"]
        )
        proposal = item["proposal"]
        cards.append(
            f'''<article class="card" data-key="{_text(item['key'])}" data-lot="{item['lot']}">
<header><b>{item['order']}. {_text(item['video_title'])}</b>
<span class="risk {item['risk']}">{item['risk']}</span></header>
<p><strong>Étape {item['step_number']} :</strong> {_text(proposal.get('instruction'))}</p>
<dl><dt>Action</dt><dd>{_text(proposal.get('action_type'))} — {_text(proposal.get('action_target'))}</dd>
<dt>Emplacement</dt><dd>{_text(proposal.get('action_location'))}</dd>
<dt>Écran / résultat</dt><dd>{_text(proposal.get('screen_before'))} → {_text(proposal.get('screen_after'))}</dd>
<dt>Description</dt><dd>{_text(proposal.get('visual_description'))}</dd></dl>
<div class="gallery">{images}</div>
<div class="decision"><button class="ok" onclick="decide('{_text(item['key'])}','correct')">✓ Correct</button>
<button class="bad" onclick="decide('{_text(item['key'])}','incorrect')">✗ À corriger</button>
<input placeholder="Correction ou remarque (si nécessaire)" oninput="note('{_text(item['key'])}',this.value)"></div>
</article>'''
        )
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    lot_sizes = {
        lot: sum(int(item["lot"]) == lot for item in items)
        for lot in range(1, lot_count + 1)
    }
    if review_type == "exception_individual_review":
        heading = f"Validation individuelle — {total} exceptions"
    else:
        heading = f"Validation rapide — {lot_count} lots, {total} contrôles"
    return f'''<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Revue vidéo — {batch_id}</title>
<style>
body{{font:16px system-ui;margin:0;background:#eef1f5;color:#18202a}} .top{{position:sticky;top:0;z-index:3;background:#18202a;color:white;padding:12px 4%;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
.top button,.decision button{{border:0;border-radius:7px;padding:10px 14px;font-weight:700;cursor:pointer}} .top button{{background:#ffd166}} #progress{{margin-left:auto}}
main{{max-width:1500px;margin:auto;padding:22px}} .lot-title{{margin:32px 0 12px;padding:12px;background:#dbe8ff;border-radius:8px;display:flex;justify-content:space-between}}
.card{{background:white;border:3px solid transparent;border-radius:12px;padding:16px;margin:14px 0;box-shadow:0 2px 8px #0002}} .card.correct{{border-color:#2b9348}} .card.incorrect{{border-color:#d90429}}
.card header{{display:flex;justify-content:space-between;gap:15px;font-size:19px}} .risk{{color:white;border-radius:999px;padding:3px 10px;font-size:13px}} .high{{background:#d90429}} .medium{{background:#e07a00}} .low{{background:#2b9348}}
dl{{display:grid;grid-template-columns:140px 1fr;gap:5px 12px}} dt{{font-weight:700}} dd{{margin:0}} .gallery{{display:flex;gap:10px;overflow-x:auto;background:#111;padding:10px;border-radius:8px}}
figure{{margin:0;min-width:min(620px,78vw)}} img{{display:block;width:100%;max-height:520px;object-fit:contain;background:#222}} figcaption{{color:white;text-align:center;padding:5px}}
.decision{{display:grid;grid-template-columns:150px 150px 1fr;gap:10px;margin-top:12px}} .ok{{background:#80ed99}} .bad{{background:#ff8fa3}} input{{padding:9px;border:1px solid #9aa4b2;border-radius:7px}}
@media(max-width:700px){{.decision{{grid-template-columns:1fr 1fr}}.decision input{{grid-column:1/3}}dl{{grid-template-columns:1fr}}}}
</style></head><body><div class="top"><b>{heading}</b>
<button onclick="validateLot()">✓ Valider le lot affiché</button><button onclick="exportResults()">Exporter les résultats</button><span id="progress"></span></div><main>
{''.join(f'<section id="lot-{lot}"><h2 class="lot-title">Lot {lot}/{lot_count} — {lot_sizes[lot]} élément(s) <button onclick="validateLot({lot})">Tout ce lot est correct</button></h2>' + ''.join(card for card,item in zip(cards,items) if item['lot']==lot) + '</section>' for lot in range(1,lot_count + 1))}
</main><script>const ITEMS={payload}; const KEY='reception-rag-review-{batch_id}'; let state=JSON.parse(localStorage.getItem(KEY)||'{{}}');
function save(){{localStorage.setItem(KEY,JSON.stringify(state));paint()}} function decide(k,d){{state[k]=state[k]||{{}};state[k].decision=d;save()}} function note(k,v){{state[k]=state[k]||{{}};state[k].note=v;localStorage.setItem(KEY,JSON.stringify(state))}}
function paint(){{document.querySelectorAll('.card').forEach(c=>{{c.classList.remove('correct','incorrect');if(state[c.dataset.key]?.decision)c.classList.add(state[c.dataset.key].decision);const i=c.querySelector('input');if(document.activeElement!==i)i.value=state[c.dataset.key]?.note||''}});const n=ITEMS.filter(x=>state[x.key]?.decision).length;document.getElementById('progress').textContent=n+'/{total} contrôlés'}}
function validateLot(lot){{if(!lot){{const y=scrollY+120;const s=[...document.querySelectorAll('section')].find(x=>x.offsetTop<=y&&x.offsetTop+x.offsetHeight>y);lot=Number((s?.id||'lot-1').split('-')[1])}}ITEMS.filter(x=>x.lot===lot).forEach(x=>{{state[x.key]=state[x.key]||{{}};if(!state[x.key].decision)state[x.key].decision='correct'}});save()}}
function exportResults(){{const missing=ITEMS.filter(x=>!state[x.key]?.decision);if(missing.length){{alert('Il reste '+missing.length+' élément(s) non contrôlé(s).');return}}const result={{schema_version:1,batch_id:'{batch_id}',review_type:'{review_type}',reviewer:'reviewer',reviewed_at:new Date().toISOString(),items:ITEMS.map(x=>({{video_id:x.video_id,step_number:x.step_number,risk:x.risk,proposal_hash:x.proposal_hash,decision:state[x.key].decision,note:state[x.key].note||null}}))}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{{type:'application/json'}}));a.download='{batch_id}-results.json';a.click();URL.revokeObjectURL(a.href)}} paint();</script></body></html>'''


def build(
    *,
    audit_file: Path | None = None,
    suggestions_file: Path | None = None,
    videos_root: Path | None = None,
    output_dir: Path | None = None,
    project_root: Path | None = None,
    quotas: dict[str, int] | None = None,
    seed: int = 20260825,
) -> tuple[Path, list[dict[str, Any]]]:
    """Crée 60 contrôles indépendants en 6 lots de 10, sans valider les sources."""

    project_root = (project_root or path("")).resolve()
    audit_file = audit_file or path("data/interim/video_quality_audit.json")
    suggestions_file = suggestions_file or path(
        "data/interim/video_calibration_suggestions.jsonl"
    )
    videos_root = videos_root or path("data/interim/videos")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"video-review-{timestamp}-s{seed}"
    output_dir = output_dir or path(f"data/exports/{batch_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = read_json(audit_file)
    suggestions = _load_suggestions(suggestions_file)
    steps = _load_steps(videos_root)
    sample = select_sample(report, suggestions, quotas=quotas, seed=seed)
    items: list[dict[str, Any]] = []
    for order, finding in enumerate(sample, 1):
        key_tuple = (str(finding["video_id"]), int(finding["step_number"]))
        step = steps[key_tuple]
        suggestion = suggestions.get(key_tuple)
        proposal = (
            dict(suggestion["proposal"])
            if suggestion
            else step.model_dump(mode="json")
        )
        item_dir = output_dir / "assets" / f"item-{order:02d}"
        evidence = _evidence(
            proposal,
            item_dir=item_dir,
            output_dir=output_dir,
            project_root=project_root,
        )
        items.append(
            {
                "order": order,
                "lot": (order - 1) // 10 + 1,
                "key": f"{key_tuple[0]}:{key_tuple[1]}",
                "video_id": key_tuple[0],
                "video_title": step.video_title,
                "step_number": key_tuple[1],
                "risk": str(finding["risk"]),
                "proposal": proposal,
                "proposal_hash": _proposal_hash(proposal),
                "evidence": evidence,
            }
        )

    manifest = {
        "schema_version": 1,
        "batch_id": batch_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sample_method": "stratified_by_risk_then_diversified_by_video",
        "quotas": quotas or DEFAULT_QUOTAS,
        "items": items,
    }
    atomic_write_json(output_dir / "review_manifest.json", manifest)
    (output_dir / "index.html").write_text(
        _render_html(batch_id, items), encoding="utf-8"
    )
    return output_dir, items


def build_exception_packet(
    *,
    audit_file: Path | None = None,
    suggestions_file: Path | None = None,
    videos_root: Path | None = None,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Crée une revue individuelle des étapes dont la calibration a échoué."""

    project_root = (project_root or path("")).resolve()
    audit_file = audit_file or path("data/interim/video_quality_audit.json")
    suggestions_file = suggestions_file or path(
        "data/interim/video_calibration_suggestions.jsonl"
    )
    videos_root = videos_root or path("data/interim/videos")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"video-exception-review-{timestamp}"
    output_dir = output_dir or path(f"data/exports/{batch_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    findings = {
        (str(row["video_id"]), int(row["step_number"])): row
        for row in read_json(audit_file).get("step_findings", [])
    }
    errors = sorted(
        (row for row in _suggestion_rows(suggestions_file) if row.get("status") == "error"),
        key=lambda row: (str(row["video_id"]), int(row["step_number"])),
    )
    steps = _load_steps(videos_root)
    items: list[dict[str, Any]] = []
    for order, error in enumerate(errors, 1):
        key_tuple = (str(error["video_id"]), int(error["step_number"]))
        step = steps[key_tuple]
        proposal = step.model_dump(mode="json")
        item_dir = output_dir / "assets" / f"item-{order:02d}"
        evidence = _evidence(
            proposal,
            item_dir=item_dir,
            output_dir=output_dir,
            project_root=project_root,
        )
        finding = findings.get(key_tuple, {})
        items.append(
            {
                "order": order,
                "lot": 1,
                "key": f"{key_tuple[0]}:{key_tuple[1]}",
                "video_id": key_tuple[0],
                "video_title": step.video_title,
                "step_number": key_tuple[1],
                "risk": str(finding.get("risk", "high")),
                "proposal": proposal,
                "proposal_hash": _proposal_hash(proposal),
                "evidence": evidence,
                "technical_error": error.get("error"),
            }
        )

    manifest = {
        "schema_version": 1,
        "batch_id": batch_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_method": "individual_review_of_calibration_errors",
        "items": items,
    }
    atomic_write_json(output_dir / "review_manifest.json", manifest)
    (output_dir / "index.html").write_text(
        _render_html(
            batch_id,
            items,
            review_type="exception_individual_review",
        ),
        encoding="utf-8",
    )
    return output_dir, items
