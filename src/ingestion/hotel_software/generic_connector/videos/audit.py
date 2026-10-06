"""Audit automatique des Procedural Chunks video avant revue humaine."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .....config import path
from .....schema import ProceduralStep
from .common import atomic_write_json

TARGET_ACTIONS = {"click", "type", "select", "hover"}
RISK_WEIGHT = {"critical": 100, "high": 10, "medium": 3, "low": 0}
NO_ISSUE_FLAGS = {
    "aucune information contradictoire ou manquante",
    "aucune information manquante ou contradictoire",
}


def _read_json(file: Path) -> Any:
    return json.loads(file.read_text(encoding="utf-8"))


def _asset_path(value: str, project_root: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else project_root / candidate


def _manifest_entries(manifest_file: Path) -> dict[str, dict[str, Any]]:
    if not manifest_file.exists():
        return {}
    payload = _read_json(manifest_file)
    rows = payload.get("videos", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{manifest_file}: champ 'videos' invalide")
    return {
        str(row["video_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("video_id")
    }


def _category(entry: dict[str, Any]) -> str:
    relative = str(entry.get("relative_to_video_root") or "")
    return Path(relative).parts[0] if Path(relative).parts else "sans_categorie"


def _normalized_flag(value: str) -> str:
    value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _flag_category(flag: str) -> str:
    """Regroupe les formulations libres du LLM en classes auditables."""

    value = _normalized_flag(flag)
    if value in NO_ISSUE_FLAGS:
        return "no_issue"
    if any(
        marker in value
        for marker in (
            "conflict",
            "contradic",
            "discrepancy",
            "ne correspond pas",
            "aucune correspondance",
            "sans correspondance",
        )
    ):
        return "audio_visual_conflict"
    if "unsupported action target" in value:
        return "unsupported_action_target"
    if "unsupported action type" in value:
        return "unsupported_action_type"
    if any(
        marker in value
        for marker in (
            "missing exact ui label",
            "libelle exact",
            "libelle visible",
            "action target inconnu",
            "unclear action target",
            "information manquante action target",
        )
    ):
        return "missing_ui_label"
    if any(
        marker in value
        for marker in (
            "capture",
            "screenshot",
            "preuve visuelle",
            "description visuelle",
            "visual description",
            "ecran",
            "frame",
            "image",
        )
    ) and any(
        marker in value
        for marker in (
            "aucun",
            "aucune",
            "absence",
            "missing",
            "manquant",
            "non fourni",
            "pas visible",
            "inexist",
            "null",
        )
    ):
        return "missing_visual_evidence"
    if any(
        marker in value
        for marker in (
            "uncertain",
            "incertain",
            "incertitude",
            "confident false",
            "hypothese",
            "deduit",
            "inferred",
        )
    ):
        return "visual_uncertainty"
    if any(
        marker in value
        for marker in (
            "aucun",
            "aucune",
            "absence",
            "missing",
            "manquant",
            "incomplet",
            "non specifie",
            "non visible",
        )
    ):
        return "missing_information"
    return "other_quality_warning"


def _risk_level(
    issues: list[str], step: ProceduralStep, flag_categories: set[str]
) -> str:
    if any(issue.startswith(("missing_asset:", "invalid_")) for issue in issues):
        return "critical"
    if step.status in {"verified", "deprecated"}:
        return "low"
    if any(
        issue in {"no_screenshot", "missing_action_target", "timestamp_outside_video"}
        for issue in issues
    ) or flag_categories & {"audio_visual_conflict", "unsupported_action_target"}:
        return "high"
    if not step.confident or flag_categories - {"no_issue"} or issues:
        return "medium"
    return "low"


def _step_findings(
    step: ProceduralStep,
    *,
    project_root: Path,
    duration: float | None,
) -> tuple[list[str], list[str], str, list[str]]:
    issues: list[str] = []
    assets = [
        value
        for value in [
            step.screenshot_before,
            *step.supporting_screenshots,
            step.screenshot_after,
        ]
        if value
    ]
    missing_assets = [
        value
        for value in assets
        if not _asset_path(value, project_root).is_file()
    ]
    issues.extend(f"missing_asset:{value}" for value in missing_assets)
    if not assets:
        issues.append("no_screenshot")
    if step.action_type in TARGET_ACTIONS and not step.action_target:
        issues.append("missing_action_target")
    if not step.instruction.strip():
        issues.append("empty_instruction")
    if "\ufffd" in step.instruction or "\ufffd" in step.spoken_instruction:
        issues.append("invalid_replacement_character")
    if duration is not None and step.timestamp_end > duration + 1.0:
        issues.append("timestamp_outside_video")
    issues.extend(f"quality_flag:{flag}" for flag in step.quality_flags)
    categories = sorted({_flag_category(flag) for flag in step.quality_flags})
    return (
        sorted(set(issues)),
        missing_assets,
        _risk_level(issues, step, set(categories)),
        categories,
    )


def _recommended_sample(videos: list[dict[str, Any]], limit: int = 16) -> list[str]:
    videos = [
        video
        for video in videos
        if video["statuses"].get("review_required", 0) > 0
    ]
    selected: list[str] = []

    def add(video_id: str) -> None:
        if video_id not in selected:
            selected.append(video_id)

    for video in videos:
        if video["critical_steps"] or video["missing_assets"]:
            add(video["video_id"])

    categories = sorted({video["category"] for video in videos})
    for category in categories:
        candidates = [video for video in videos if video["category"] == category]
        highest = max(
            candidates,
            key=lambda item: (item["risk_score"], item["steps"], item["video_id"]),
        )
        add(highest["video_id"])

    if videos:
        add(max(videos, key=lambda item: item["duration"])["video_id"])
        add(min(videos, key=lambda item: item["duration"])["video_id"])
        add(max(videos, key=lambda item: item["steps"])["video_id"])

    for video in sorted(
        videos,
        key=lambda item: (item["risk_score"], item["steps"], item["video_id"]),
        reverse=True,
    ):
        if len(selected) >= limit:
            break
        add(video["video_id"])
    return selected


def _recommended_steps(
    video_ids: list[str], findings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Une etape prioritaire par video, plutôt que relire la video entière."""

    rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    selected: list[dict[str, Any]] = []
    for video_id in video_ids:
        candidates = [
            finding
            for finding in findings
            if finding["video_id"] == video_id
            and finding.get("status") == "review_required"
        ]
        if not candidates:
            continue
        item = max(
            candidates,
            key=lambda finding: (
                rank[finding["risk"]],
                len(finding.get("quality_flag_categories", [])),
                -finding["step_number"],
            ),
        )
        selected.append(
            {
                key: item[key]
                for key in (
                    "video_id",
                    "video_title",
                    "step_number",
                    "timestamp_start",
                    "timestamp_end",
                    "risk",
                    "quality_flag_categories",
                )
            }
        )
    return selected


def build_report(
    *,
    videos_root: Path | None = None,
    manifest_file: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    project_root = (project_root or path("")).resolve()
    videos_root = videos_root or path("data/interim/videos")
    manifest_file = manifest_file or path("data/interim/video_manifest.json")
    manifest = _manifest_entries(manifest_file)
    status_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    flag_category_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    videos: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    processed_ids: set[str] = set()
    invalid_steps = 0

    for steps_file in sorted(videos_root.glob("*/steps.json")):
        video_id = steps_file.parent.name
        processed_ids.add(video_id)
        entry = manifest.get(video_id, {})
        duration = (
            float(entry["duration"])
            if isinstance(entry.get("duration"), (int, float))
            else None
        )
        payload = _read_json(steps_file)
        rows = payload.get("steps", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"{steps_file}: champ 'steps' invalide")

        parsed: list[ProceduralStep] = []
        video_invalid = 0
        for row_index, row in enumerate(rows, 1):
            try:
                parsed.append(ProceduralStep(**row))
            except Exception as exc:
                invalid_steps += 1
                video_invalid += 1
                findings.append(
                    {
                        "video_id": video_id,
                        "step_number": row_index,
                        "risk": "critical",
                        "issues": [f"invalid_schema:{exc}"],
                        "quality_flags": [],
                    }
                )

        sequence_issues: list[str] = []
        numbers = [step.step_number for step in parsed]
        expected_numbers = list(range(1, len(parsed) + 1))
        if numbers != expected_numbers:
            sequence_issues.append("non_contiguous_step_numbers")
        timestamps = [step.timestamp_start for step in parsed]
        if timestamps != sorted(timestamps):
            sequence_issues.append("non_monotonic_timestamps")
        if any(step.total_steps != len(parsed) for step in parsed):
            sequence_issues.append("inconsistent_total_steps")

        per_risk: Counter[str] = Counter()
        missing_asset_count = 0
        for step in parsed:
            status_counts[step.status] += 1
            flag_counts.update(step.quality_flags)
            issues, missing_assets, risk, flag_categories = _step_findings(
                step, project_root=project_root, duration=duration
            )
            flag_category_counts.update(flag_categories)
            per_risk[risk] += 1
            risk_counts[risk] += 1
            missing_asset_count += len(missing_assets)
            if issues or risk != "low" or step.status == "review_required":
                findings.append(
                    {
                        "video_id": video_id,
                        "video_title": step.video_title,
                        "step_number": step.step_number,
                        "timestamp_start": step.timestamp_start,
                        "timestamp_end": step.timestamp_end,
                        "status": step.status,
                        "confident": step.confident,
                        "risk": risk,
                        "issues": issues,
                        "quality_flags": step.quality_flags,
                        "quality_flag_categories": flag_categories,
                    }
                )

        risk_score = (
            sum(per_risk[level] * weight for level, weight in RISK_WEIGHT.items())
            + missing_asset_count * 20
            + video_invalid * 100
            + len(sequence_issues) * 25
        )
        videos.append(
            {
                "video_id": video_id,
                "video_title": (
                    parsed[0].video_title
                    if parsed
                    else entry.get("relative_to_video_root")
                ),
                "category": _category(entry),
                "duration": duration or 0.0,
                "steps": len(parsed),
                "statuses": dict(Counter(step.status for step in parsed)),
                "confident_steps": sum(step.confident for step in parsed),
                "flagged_steps": sum(bool(step.quality_flags) for step in parsed),
                "critical_steps": per_risk["critical"] + video_invalid,
                "high_risk_steps": per_risk["high"],
                "medium_risk_steps": per_risk["medium"],
                "low_risk_steps": per_risk["low"],
                "missing_assets": missing_asset_count,
                "sequence_issues": sequence_issues,
                "risk_score": risk_score,
            }
        )

    videos.sort(key=lambda item: item["video_id"])
    recommendations = _recommended_sample(videos)
    recommended_steps = _recommended_steps(recommendations, findings)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "manifest_videos": len(manifest),
            "step_files": len(videos),
            "total_steps": sum(video["steps"] for video in videos),
            "invalid_steps": invalid_steps,
            "status_counts": dict(status_counts),
            "risk_counts": dict(risk_counts),
            "flagged_steps": sum(video["flagged_steps"] for video in videos),
            "confident_steps": sum(video["confident_steps"] for video in videos),
            "missing_assets": sum(video["missing_assets"] for video in videos),
            "unprocessed_manifest_videos": sorted(set(manifest) - processed_ids),
            "orphan_step_videos": sorted(processed_ids - set(manifest)),
        },
        "quality_flags": dict(flag_counts.most_common()),
        "quality_flag_categories": dict(flag_category_counts.most_common()),
        "recommended_review_videos": recommendations,
        "recommended_review_steps": recommended_steps,
        "videos": videos,
        "step_findings": findings,
    }


def write_report(output: Path | None = None) -> tuple[Path, dict[str, Any]]:
    output = output or path("data/interim/video_quality_audit.json")
    report = build_report()
    atomic_write_json(output, report)
    return output, report
