import json

from PIL import Image

from src.ingestion.hotel_software.generic_connector.videos.batch_review import (
    build,
    build_exception_packet,
    select_sample,
)
from src.schema import ProceduralStep


def test_sample_is_stratified_and_reproducible():
    findings = []
    suggestions = {}
    for risk, count in (("high", 4), ("medium", 4), ("low", 3)):
        for index in range(count):
            row = {
                "video_id": f"{risk}-{index}",
                "step_number": 1,
                "status": "review_required",
                "risk": risk,
            }
            findings.append(row)
            if risk != "low":
                suggestions[(row["video_id"], 1)] = {"status": "proposal_only"}
    report = {"step_findings": findings}
    quotas = {"high": 2, "medium": 2, "low": 1}

    first = select_sample(report, suggestions, quotas=quotas, seed=7)
    second = select_sample(report, suggestions, quotas=quotas, seed=7)

    assert first == second
    assert [row["risk"] for row in first].count("high") == 2
    assert [row["risk"] for row in first].count("medium") == 2
    assert [row["risk"] for row in first].count("low") == 1


def test_build_creates_offline_html_packet(tmp_path):
    videos_root = tmp_path / "videos"
    frame = tmp_path / "frame.png"
    Image.new("RGB", (80, 50), "white").save(frame)
    findings = []
    suggestions = []
    quotas = {"high": 1, "medium": 1, "low": 1}
    for risk in quotas:
        video_id = f"video-{risk}"
        folder = videos_root / video_id
        folder.mkdir(parents=True)
        step = ProceduralStep(
            procedure="test",
            step_number=1,
            total_steps=1,
            timestamp_start=0,
            timestamp_end=1,
            spoken_instruction="Cliquez.",
            instruction="Cliquez sur Enregistrer.",
            action_type="click",
            action_target="Enregistrer",
            screenshot_before=frame.as_posix(),
            video_id=video_id,
            video_title=video_id,
        )
        (folder / "steps.json").write_text(
            json.dumps({"steps": [step.model_dump(mode="json")]}),
            encoding="utf-8",
        )
        findings.append(
            {
                "video_id": video_id,
                "step_number": 1,
                "status": "review_required",
                "risk": risk,
            }
        )
        if risk != "low":
            suggestions.append(
                {
                    "video_id": video_id,
                    "step_number": 1,
                    "status": "proposal_only",
                    "proposal": step.model_dump(mode="json"),
                }
            )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"step_findings": findings}), encoding="utf-8")
    suggestions_file = tmp_path / "suggestions.jsonl"
    suggestions_file.write_text(
        "\n".join(json.dumps(row) for row in suggestions), encoding="utf-8"
    )
    output = tmp_path / "packet"

    packet, items = build(
        audit_file=audit,
        suggestions_file=suggestions_file,
        videos_root=videos_root,
        output_dir=output,
        project_root=tmp_path,
        quotas=quotas,
        seed=3,
    )

    assert packet == output
    assert len(items) == 3
    assert (output / "index.html").is_file()
    assert (output / "review_manifest.json").is_file()
    assert len(list((output / "assets").rglob("*.jpg"))) == 3


def test_exception_packet_uses_dynamic_total(tmp_path):
    videos_root = tmp_path / "videos"
    frame = tmp_path / "frame.png"
    Image.new("RGB", (80, 50), "white").save(frame)
    findings = []
    suggestions = []
    for index in range(3):
        video_id = f"video-{index}"
        folder = videos_root / video_id
        folder.mkdir(parents=True)
        step = ProceduralStep(
            procedure="test",
            step_number=1,
            timestamp_start=0,
            timestamp_end=1,
            spoken_instruction="Cliquez.",
            instruction="Cliquez.",
            screenshot_before=frame.as_posix(),
            video_id=video_id,
            video_title=video_id,
        )
        (folder / "steps.json").write_text(
            json.dumps({"video_id": video_id, "steps": [step.model_dump(mode="json")]}),
            encoding="utf-8",
        )
        findings.append(
            {
                "video_id": video_id,
                "step_number": 1,
                "status": "review_required",
                "risk": "high",
            }
        )
        suggestions.append(
            {
                "video_id": video_id,
                "step_number": 1,
                "status": "error",
                "error": "invalid json",
            }
        )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"step_findings": findings}), encoding="utf-8")
    suggestions_file = tmp_path / "suggestions.jsonl"
    suggestions_file.write_text(
        "\n".join(json.dumps(row) for row in suggestions), encoding="utf-8"
    )
    output = tmp_path / "exceptions"

    packet, items = build_exception_packet(
        audit_file=audit,
        suggestions_file=suggestions_file,
        videos_root=videos_root,
        output_dir=output,
        project_root=tmp_path,
    )

    html = (packet / "index.html").read_text(encoding="utf-8")
    assert len(items) == 3
    assert "3 exceptions" in html
    assert "n+'/3 contrôlés'" in html
    assert "exception_individual_review" in html
