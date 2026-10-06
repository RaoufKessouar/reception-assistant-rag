import json

from src.ingestion.hotel_software.generic_connector.videos.review_packet import build
from src.schema import ProceduralStep


def test_video_review_packet_copies_only_recommended_assets(tmp_path):
    videos_root = tmp_path / "videos"
    video_dir = videos_root / "video-test"
    video_dir.mkdir(parents=True)
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    step = ProceduralStep(
        procedure="test",
        step_number=1,
        total_steps=1,
        timestamp_start=1.0,
        timestamp_end=2.0,
        spoken_instruction="Cliquez ici.",
        instruction="Cliquez sur Enregistrer.",
        action_type="click",
        action_target="Enregistrer",
        screenshot_before=before.as_posix(),
        screenshot_after=after.as_posix(),
        video_id="video-test",
        video_title="Vidéo test",
    )
    (video_dir / "steps.json").write_text(
        json.dumps({"steps": [step.model_dump(mode="json")]}), encoding="utf-8"
    )
    audit_file = tmp_path / "audit.json"
    audit_file.write_text(
        json.dumps(
            {
                "recommended_review_steps": [
                    {
                        "video_id": "video-test",
                        "step_number": 1,
                        "quality_flag_categories": ["visual_uncertainty"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "packet"

    packet_dir, items = build(
        audit_file=audit_file,
        output_dir=output,
        videos_root=videos_root,
        project_root=tmp_path,
    )

    assert packet_dir == output
    assert len(items) == 1
    assert len(items[0]["screenshots"]) == 2
    assert (output / items[0]["screenshots"][0]["file"]).read_bytes() == b"before"
    assert (output / items[0]["screenshots"][1]["file"]).read_bytes() == b"after"
    assert (output / "review_items.json").is_file()
    assert (output / "README.md").is_file()
