import json

from src.ingestion.hotel_software.generic_connector.videos import batch_validation
from src.ingestion.hotel_software.generic_connector.videos.batch_review import _proposal_hash
from src.schema import ProceduralStep


def _write_step(root, step):
    folder = root / step.video_id
    folder.mkdir(parents=True)
    (folder / "steps.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "video_id": step.video_id,
                "steps": [step.model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )


def test_sampled_batch_is_traced_and_preserves_rejected_evidence(tmp_path, monkeypatch):
    videos_root = tmp_path / "videos"
    before = (tmp_path / "before.png").as_posix()
    after = (tmp_path / "after.png").as_posix()
    high = ProceduralStep(
        procedure="test",
        step_number=1,
        timestamp_start=0,
        timestamp_end=1,
        spoken_instruction="Cliquez.",
        instruction="Cliquez.",
        visual_description="Description originale.",
        screenshot_before=before,
        screenshot_after=after,
        video_id="video-high",
    )
    low = high.model_copy(
        update={
            "video_id": "video-low",
            "screenshot_after": None,
        }
    )
    exception = high.model_copy(update={"video_id": "video-exception"})
    for step in (high, low, exception):
        _write_step(videos_root, step)

    proposal = high.model_dump(mode="json")
    proposal["visual_description"] = "Description proposée mais rejetée."
    suggestions = tmp_path / "suggestions.jsonl"
    suggestions.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "video_id": "video-high",
                        "step_number": 1,
                        "status": "proposal_only",
                        "model": "test-model",
                        "proposal": proposal,
                    }
                ),
                json.dumps(
                    {
                        "video_id": "video-exception",
                        "step_number": 1,
                        "status": "error",
                        "error": "invalid json",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    batch_id = "video-review-test"
    manifest_items = [
        {
            "video_id": "video-high",
            "step_number": 1,
            "risk": "high",
            "proposal_hash": _proposal_hash(proposal),
        },
        {
            "video_id": "video-low",
            "step_number": 1,
            "risk": "low",
            "proposal_hash": _proposal_hash(low.model_dump(mode="json")),
        },
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "batch_id": batch_id, "items": manifest_items}
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": batch_id,
                "review_type": "independent_stratified_sample",
                "reviewer": "reviewer",
                "reviewed_at": "2026-08-26T15:44:51Z",
                "items": [
                    {**manifest_items[0], "decision": "incorrect", "note": "ancienne capture"},
                    {**manifest_items[1], "decision": "correct", "note": None},
                ],
            }
        ),
        encoding="utf-8",
    )
    resolutions = tmp_path / "resolutions.json"
    resolutions.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": batch_id,
                "resolutions": [
                    {
                        "video_id": "video-high",
                        "step_number": 1,
                        "severity": "minor",
                        "resolution": "prefer_original_primary_evidence",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(batch_validation, "path", lambda value: tmp_path / value)

    preview = batch_validation.apply_sampled_batch(
        results_file=results,
        manifest_file=manifest,
        resolutions_file=resolutions,
        suggestions_file=suggestions,
        videos_root=videos_root,
        apply=False,
    )
    assert preview["population"] == {
        "eligible": 2,
        "calibrated": 1,
        "kept_original": 1,
        "pending_exceptions": 1,
    }

    applied = batch_validation.apply_sampled_batch(
        results_file=results,
        manifest_file=manifest,
        resolutions_file=resolutions,
        suggestions_file=suggestions,
        videos_root=videos_root,
        apply=True,
    )
    high_saved = json.loads(
        (videos_root / "video-high" / "steps.json").read_text(encoding="utf-8")
    )["steps"][0]
    low_saved = json.loads(
        (videos_root / "video-low" / "steps.json").read_text(encoding="utf-8")
    )["steps"][0]
    exception_saved = json.loads(
        (videos_root / "video-exception" / "steps.json").read_text(encoding="utf-8")
    )["steps"][0]

    assert applied["applied"] is True
    assert high_saved["status"] == low_saved["status"] == "verified"
    assert exception_saved["status"] == "review_required"
    assert high_saved["visual_description"] == "Description originale."
    assert high_saved["screenshot_after"] is None
    assert high_saved["supporting_screenshots"] == [after]
    assert high_saved["validation_method"] == "batch_sampled"
    assert high_saved["validation_batch_id"] == batch_id
    assert high_saved["validation_model"] == "test-model"
    assert low_saved["validation_model"] is None
    assert (
        tmp_path
        / "data/interim/video_batch_validation_backups"
        / batch_id
        / "video-high"
        / "steps.json"
    ).is_file()


def test_review_result_hash_must_match_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    results = tmp_path / "results.json"
    item = {
        "video_id": "video",
        "step_number": 1,
        "risk": "high",
        "proposal_hash": "expected",
    }
    manifest.write_text(
        json.dumps({"schema_version": 1, "batch_id": "batch", "items": [item]}),
        encoding="utf-8",
    )
    results.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": "batch",
                "review_type": "independent_stratified_sample",
                "reviewer": "reviewer",
                "reviewed_at": "2026-08-26T00:00:00Z",
                "items": [{**item, "proposal_hash": "tampered", "decision": "correct"}],
            }
        ),
        encoding="utf-8",
    )

    try:
        batch_validation.validate_results(
            results,
            manifest,
            expected_review_type="independent_stratified_sample",
        )
    except ValueError as error:
        assert "modifiée après génération" in str(error)
    else:
        raise AssertionError("Une empreinte modifiée doit être rejetée")
