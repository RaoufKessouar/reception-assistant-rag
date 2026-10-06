import json

from src.ingestion.hotel_software.generic_connector.videos.review import apply_decisions
from src.schema import ProceduralStep


def test_apply_video_review_decision_updates_and_verifies_step(tmp_path, monkeypatch):
    from src.ingestion.hotel_software.generic_connector.videos import review

    video_dir = tmp_path / "video-test"
    video_dir.mkdir()
    step = ProceduralStep(
        procedure="test",
        step_number=1,
        total_steps=1,
        timestamp_start=1,
        timestamp_end=2,
        spoken_instruction="Cliquez ici.",
        instruction="Cliquez ici.",
        video_id="video-test",
    )
    steps_file = video_dir / "steps.json"
    steps_file.write_text(
        json.dumps({"steps": [step.model_dump(mode="json")]}), encoding="utf-8"
    )
    monkeypatch.setattr(review, "load", lambda _video_id: [step])
    monkeypatch.setattr(review, "output_file", lambda _video_id: steps_file)
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "video_id": "video-test",
                "step_number": 1,
                "status": "verified",
                "reviewed_by": "reviewer",
                "changes": {
                    "instruction": "Cliquez sur Enregistrer.",
                    "action_type": "click",
                    "action_target": "Enregistrer",
                },
            }
        ),
        encoding="utf-8",
    )

    applied = apply_decisions(decisions)
    saved = json.loads(steps_file.read_text(encoding="utf-8"))["steps"][0]

    assert applied == [
        {"video_id": "video-test", "step_number": 1, "status": "verified"}
    ]
    assert saved["instruction"] == "Cliquez sur Enregistrer."
    assert saved["action_target"] == "Enregistrer"
    assert saved["status"] == "verified"
    assert saved["reviewed_by"] == "reviewer"
