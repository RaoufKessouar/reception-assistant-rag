import json

from src.ingestion.hotel_software.generic_connector.videos.calibration import (
    CalibrationDraft,
    _normalise_draft,
    _per_frame_pixel_budget,
    evaluate_reference_sample,
    propose_remaining,
    propose_verified_corpus,
    select_references,
    write_reference_set,
)
from src.schema import ProceduralStep


def _step(
    video_id: str,
    *,
    step_number: int = 1,
    status: str = "review_required",
) -> ProceduralStep:
    values = {
        "procedure": "paiement",
        "step_number": step_number,
        "total_steps": 1,
        "timestamp_start": 1.0,
        "timestamp_end": 2.0,
        "spoken_instruction": "Cliquez sur Oui pour rembourser.",
        "instruction": "Dans le détail du paiement, sélectionnez Oui.",
        "action_type": "click",
        "action_target": "Oui",
        "action_location": "dans la section Remboursement",
        "screenshot_before": "frame-1.png",
        "screenshot_after": None,
        "video_id": video_id,
        "video_title": "Paiement automatisé",
        "status": status,
    }
    if status == "verified":
        values.update(
            reviewed_by="reviewer",
            reviewed_at="2026-08-25T00:00:00Z",
        )
    return ProceduralStep(**values)


def test_reference_set_contains_only_verified_steps(tmp_path):
    videos_root = tmp_path / "videos"
    for step in [_step("verified", status="verified"), _step("draft")]:
        folder = videos_root / step.video_id
        folder.mkdir(parents=True)
        (folder / "steps.json").write_text(
            json.dumps({"steps": [step.model_dump(mode="json")]}),
            encoding="utf-8",
        )

    output, references = write_reference_set(
        output=tmp_path / "references.json", videos_root=videos_root
    )

    assert output.exists()
    assert len(references) == 1
    assert references[0]["video_id"] == "verified"


def test_select_references_excludes_current_step_and_prefers_similar_text():
    current = _step("current")
    similar = _step("similar", status="verified")
    unrelated = _step("unrelated", status="verified").model_copy(
        update={
            "procedure": "planning",
            "spoken_instruction": "Ouvrez le planning des chambres.",
            "instruction": "Ouvrez le planning.",
        }
    )
    selected = select_references(current, [current, unrelated, similar], limit=1)
    assert selected == [similar]


def test_normalise_draft_removes_invalid_frame_indices():
    draft = CalibrationDraft(
        instruction="Cliquez sur Oui.",
        action_type="click",
        action_target="Oui",
        relevant_frame_indices=[2, 2, 9],
        result_frame_index=9,
        confident=True,
    )

    normalised = _normalise_draft(draft, frame_count=2)

    assert normalised.relevant_frame_indices == [2]
    assert normalised.result_frame_index is None
    assert normalised.confident is True


def test_pixel_budget_is_shared_between_multiple_frames():
    assert _per_frame_pixel_budget(
        2,
        min_pixels=200_704,
        max_pixels=2_207_744,
        total_pixels=4_415_488,
    ) == 2_207_744
    assert _per_frame_pixel_budget(
        6,
        min_pixels=200_704,
        max_pixels=2_207_744,
        total_pixels=4_415_488,
    ) == 735_914


def test_evaluation_measures_gain_without_modifying_gold(tmp_path):
    videos_root = tmp_path / "videos"
    folder = videos_root / "video-test"
    folder.mkdir(parents=True)
    gold = _step("video-test", status="verified")
    steps_file = folder / "steps.json"
    steps_file.write_text(
        json.dumps({"steps": [gold.model_dump(mode="json")]}), encoding="utf-8"
    )
    packet = tmp_path / "review_items.json"
    packet.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "video_id": "video-test",
                        "video_title": "Paiement automatisé",
                        "procedure": "paiement",
                        "step_number": 1,
                        "total_steps": 1,
                        "timestamp_start": 1.0,
                        "timestamp_end": 2.0,
                        "spoken_instruction": "Cliquez sur Oui pour rembourser.",
                        "instruction": "Cliquez ici.",
                        "action_type": "none",
                        "action_target": None,
                        "action_location": None,
                        "screen_before": None,
                        "screen_after": "Remboursement effectué",
                        "visual_description": None,
                        "quality_flags": ["missing_exact_ui_label"],
                        "screenshots": [
                            {
                                "role": "avant",
                                "source": "frame-1.png",
                                "file": "01-avant.png",
                            },
                            {
                                "role": "apres",
                                "source": "frame-2.png",
                                "file": "02-apres.png",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_proposer(_step, _references):
        return CalibrationDraft(
            instruction=gold.instruction,
            action_type=gold.action_type,
            action_target=gold.action_target,
            action_location=gold.action_location,
            relevant_frame_indices=[1],
            result_frame_index=None,
            confident=True,
        )

    output = tmp_path / "evaluation.json"
    _, report = evaluate_reference_sample(
        packet_file=packet,
        output=output,
        videos_root=videos_root,
        force=True,
        proposer=fake_proposer,
    )

    assert report["summary"]["completed"] == 1
    assert report["summary"]["errors"] == 0
    assert report["summary"]["metrics"]["action_type_exact"] == {
        "baseline": 0.0,
        "proposal": 1.0,
        "delta": 1.0,
    }
    assert report["summary"]["metrics"]["screenshot_after_exact"] == {
        "baseline": 0.0,
        "proposal": 0.0,
        "delta": 0.0,
    }
    assert report["summary"]["metrics"]["after_state_presence_exact"] == {
        "baseline": 0.0,
        "proposal": 0.0,
        "delta": 0.0,
    }
    proposal = report["items"][0]["proposal"]
    assert proposal["instruction"] == "Cliquez ici."
    assert proposal["action_type"] == "click"
    assert proposal["screenshot_after"] == "frame-2.png"
    saved = json.loads(steps_file.read_text(encoding="utf-8"))["steps"][0]
    assert saved["status"] == "verified"
    assert saved["instruction"] == gold.instruction


def test_proposals_preserve_other_risks_and_retry_errors(tmp_path):
    videos_root = tmp_path / "videos"
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"png")
    for video_id in ("high-video", "medium-video"):
        folder = videos_root / video_id
        folder.mkdir(parents=True)
        step = _step(video_id).model_copy(
            update={"screenshot_before": frame.as_posix()}
        )
        (folder / "steps.json").write_text(
            json.dumps({"steps": [step.model_dump(mode="json")]}),
            encoding="utf-8",
        )
    audit_file = tmp_path / "audit.json"
    audit_file.write_text(
        json.dumps(
            {
                "step_findings": [
                    {
                        "video_id": "high-video",
                        "step_number": 1,
                        "status": "review_required",
                        "risk": "high",
                    },
                    {
                        "video_id": "medium-video",
                        "step_number": 1,
                        "status": "review_required",
                        "risk": "medium",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "suggestions.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "video_id": "high-video",
                        "step_number": 1,
                        "status": "error",
                        "error": "temporary",
                    }
                ),
                json.dumps(
                    {
                        "video_id": "unrelated-video",
                        "step_number": 3,
                        "status": "proposal_only",
                        "proposal": {},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    def fake_proposer(step, _references):
        return CalibrationDraft(
            instruction=step.instruction,
            action_type="click",
            action_target="Oui",
            relevant_frame_indices=[1],
            confident=True,
        )

    _, high = propose_remaining(
        audit_file=audit_file,
        output=output,
        videos_root=videos_root,
        risk="high",
        proposer=fake_proposer,
    )
    _, medium = propose_remaining(
        audit_file=audit_file,
        output=output,
        videos_root=videos_root,
        risk="medium",
        proposer=fake_proposer,
    )

    saved = {
        (row["video_id"], row["step_number"]): row
        for row in map(json.loads, output.read_text(encoding="utf-8").splitlines())
    }
    assert high[0]["status"] == "proposal_only"
    assert medium[0]["status"] == "proposal_only"
    assert saved[("high-video", 1)]["status"] == "proposal_only"
    assert saved[("medium-video", 1)]["status"] == "proposal_only"
    assert saved[("unrelated-video", 3)]["status"] == "proposal_only"


def test_full_vlm_pass_never_modifies_verified_source(tmp_path):
    videos_root = tmp_path / "videos"
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"png")
    gold = _step("verified-video", status="verified").model_copy(
        update={"screenshot_before": frame.as_posix()}
    )
    folder = videos_root / gold.video_id
    folder.mkdir(parents=True)
    steps_file = folder / "steps.json"
    steps_file.write_text(
        json.dumps({"steps": [gold.model_dump(mode="json")]}), encoding="utf-8"
    )

    def fake_proposer(_step, _references):
        return CalibrationDraft(
            instruction="Instruction candidate",
            action_type="click",
            action_target="Oui",
            relevant_frame_indices=[1],
            confident=True,
        )

    output = tmp_path / "full.jsonl"
    _, suggestions = propose_verified_corpus(
        output=output,
        videos_root=videos_root,
        proposer=fake_proposer,
    )

    assert suggestions[0]["status"] == "proposal_only"
    assert "instruction" in suggestions[0]["changed_fields"]
    saved = json.loads(steps_file.read_text(encoding="utf-8"))["steps"][0]
    assert saved["status"] == "verified"
    assert saved["instruction"] == gold.instruction
