import json

from src.ingestion.hotel_software.generic_connector.videos.audit import build_report
from src.schema import ProceduralStep


def _step(video_id: str, screenshot: str) -> ProceduralStep:
    return ProceduralStep(
        procedure="test_procedure",
        step_number=1,
        total_steps=1,
        source_segment_index=0,
        timestamp_start=1.0,
        timestamp_end=2.0,
        spoken_instruction="Cliquez sur Enregistrer.",
        instruction="Cliquez sur le bouton Enregistrer.",
        action_type="click",
        action_target="Enregistrer",
        screenshot_before=screenshot,
        screenshot_after=screenshot,
        confident=True,
        video_id=video_id,
        video_title="Video test",
    )


def test_video_audit_checks_assets_and_completeness(tmp_path):
    videos_root = tmp_path / "interim" / "videos"
    video_dir = videos_root / "video-test"
    video_dir.mkdir(parents=True)
    screenshot = tmp_path / "frame.png"
    screenshot.write_bytes(b"png")
    step = _step("video-test", screenshot.as_posix())
    (video_dir / "steps.json").write_text(
        json.dumps({"steps": [step.model_dump(mode="json")]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "interim" / "video_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video_id": "video-test",
                        "relative_to_video_root": "Planning/Test.mp4",
                        "duration": 10.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        videos_root=videos_root,
        manifest_file=manifest,
        project_root=tmp_path,
    )

    assert report["summary"]["manifest_videos"] == 1
    assert report["summary"]["step_files"] == 1
    assert report["summary"]["total_steps"] == 1
    assert report["summary"]["missing_assets"] == 0
    assert report["summary"]["invalid_steps"] == 0
    assert report["summary"]["risk_counts"] == {"low": 1}
    assert report["recommended_review_videos"] == ["video-test"]
    assert report["recommended_review_steps"][0]["step_number"] == 1


def test_video_audit_flags_missing_target_and_asset(tmp_path):
    videos_root = tmp_path / "videos"
    video_dir = videos_root / "video-risk"
    video_dir.mkdir(parents=True)
    step = _step("video-risk", "missing.png").model_copy(
        update={
            "action_target": None,
            "confident": False,
            "quality_flags": ["missing_exact_ui_label"],
        }
    )
    (video_dir / "steps.json").write_text(
        json.dumps({"steps": [step.model_dump(mode="json")]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video_id": "video-risk",
                        "relative_to_video_root": "Reservations/Risk.mp4",
                        "duration": 10.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        videos_root=videos_root,
        manifest_file=manifest,
        project_root=tmp_path,
    )

    assert report["summary"]["missing_assets"] == 2
    assert report["summary"]["risk_counts"] == {"critical": 1}
    finding = report["step_findings"][0]
    assert "missing_action_target" in finding["issues"]
    assert "quality_flag:missing_exact_ui_label" in finding["issues"]


def test_video_audit_separates_warning_levels_and_skips_reviewed_video(tmp_path):
    videos_root = tmp_path / "videos"
    for video_id, flag, status in [
        ("video-conflict", "audio_visual_location_conflict", "review_required"),
        ("video-uncertain", "uncertain_before_visual", "review_required"),
        ("video-reviewed", "audio_visual_location_conflict", "verified"),
    ]:
        video_dir = videos_root / video_id
        video_dir.mkdir(parents=True)
        screenshot = tmp_path / f"{video_id}.png"
        screenshot.write_bytes(b"png")
        values = _step(video_id, screenshot.as_posix()).model_dump()
        values.update(confident=False, quality_flags=[flag], status=status)
        if status == "verified":
            values.update(reviewed_by="reviewer", reviewed_at="2026-08-25T00:00:00Z")
        step = ProceduralStep(**values)
        (video_dir / "steps.json").write_text(
            json.dumps({"steps": [step.model_dump(mode="json")]}), encoding="utf-8"
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video_id": video_id,
                        "relative_to_video_root": f"Planning/{video_id}.mp4",
                        "duration": 10.0,
                    }
                    for video_id in (
                        "video-conflict",
                        "video-uncertain",
                        "video-reviewed",
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        videos_root=videos_root,
        manifest_file=manifest,
        project_root=tmp_path,
    )

    assert report["summary"]["risk_counts"] == {
        "high": 1,
        "medium": 1,
        "low": 1,
    }
    assert report["quality_flag_categories"] == {
        "audio_visual_conflict": 2,
        "visual_uncertainty": 1,
    }
    assert "video-reviewed" not in report["recommended_review_videos"]
