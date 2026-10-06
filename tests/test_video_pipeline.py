import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import settings
from src.ingestion.hotel_software.generic_connector.videos import (
    common,
    exporter,
    pipeline,
    review,
    s2_keyframes,
    s3_describe,
    s4_assemble,
)
from src.ingestion.hotel_software.generic_connector.videos.s1_transcribe import _asr_options
from src.ingestion.hotel_software.generic_connector.videos.s4_assemble import _groups
from src.schema import ProceduralStep, StepDraft, TranscriptSegment


def _step(**updates):
    values = {
        "procedure": "modifier-reservation",
        "step_number": 1,
        "timestamp_start": 10.0,
        "timestamp_end": 12.0,
        "spoken_instruction": "Cliquez ici.",
        "instruction": "Cliquez sur Modifier.",
        "video_id": "video-1",
    }
    values.update(updates)
    return ProceduralStep(**values)


def test_video_discovery_is_recursive_and_ids_do_not_collide(tmp_path):
    first = tmp_path / "A" / "demo.mp4"
    second = tmp_path / "B" / "demo.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    videos = common.discover_videos(tmp_path)
    assert videos == [first.resolve(), second.resolve()]
    assert common.video_id_for(first, tmp_path) != common.video_id_for(second, tmp_path)


def test_generated_step_is_never_verified_automatically():
    step = _step()
    assert step.status == "review_required"
    assert step.to_document().status == "review_required"


def test_verified_step_requires_human_audit_trail():
    with pytest.raises(ValidationError):
        _step(status="verified")
    step = _step(
        status="verified",
        reviewed_by="reception-manager",
        reviewed_at=datetime.now(timezone.utc),
    )
    assert step.to_document().status == "verified"


def test_scene_frames_are_kept_with_before_after_group():
    descriptions = [
        {"segment": 2, "moment": "before", "text": "Cliquez", "analysis": {}},
        {
            "segment": 2,
            "moment": "scene",
            "text": "Cliquez",
            "analysis": {"screen_name": "menu"},
        },
        {"segment": 2, "moment": "after", "text": "Cliquez", "analysis": {}},
    ]
    groups = _groups(descriptions)
    assert len(groups) == 1
    assert groups[0]["scenes"][0]["analysis"]["screen_name"] == "menu"


def test_server_b_requires_isolated_stages():
    with pytest.raises(ValueError, match="une etape GPU par invocation"):
        pipeline._validate_stages(1, 4)


def test_server_b_uses_integrated_qwen3_transformers_llm():
    assert settings()["server"] == "server-b"
    assert settings()["hw"]["llm_backend"] == "transformers"
    assert settings()["hw"]["llm_model"] == "Qwen/Qwen3-14B"
    assert settings()["hw"]["llm_device"] == "cuda:1"


def test_asr_domain_vocabulary_is_forwarded():
    config = {
        "asr": {
            "initial_prompt": "Tutoriel de reservation avec des arrhes.",
            "hotwords": "planning reservation arrhes",
        }
    }
    assert _asr_options(config) == config["asr"]


def test_keyframe_capture_rewinds_when_video_end_is_not_decodable(
    tmp_path, monkeypatch
):
    destination = tmp_path / "frame.png"
    attempts = []

    def fake_run(command, **_kwargs):
        attempts.append(float(command[command.index("-ss") + 1]))
        if len(attempts) == 2:
            destination.write_bytes(b"png")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr(s2_keyframes, "require_executable", lambda _name: "ffmpeg")
    monkeypatch.setattr(s2_keyframes.subprocess, "run", fake_run)

    captured_at = s2_keyframes._grab(
        tmp_path / "video.mp4", 105.02, destination, force=True
    )

    assert attempts == [105.02, 104.77]
    assert captured_at == pytest.approx(104.77)
    assert destination.read_bytes() == b"png"


def test_semantic_cues_capture_multiple_ui_details(tmp_path, monkeypatch):
    segment = TranscriptSegment(
        start=10.0,
        end=14.0,
        text="Le statut, l'occupation et les prestations.",
        words=[
            {"text": "statut", "start": 10.0, "end": 10.4},
            {"text": "occupation", "start": 11.0, "end": 11.5},
            {"text": "prestations", "start": 12.0, "end": 12.6},
        ],
    )
    config = {
        "video": {
            "capture_all_segments": True,
            "semantic_frame_cues": ["statut", "occupation", "prestation"],
            "semantic_frame_min_matches": 3,
            "semantic_frame_max_per_segment": 6,
            "semantic_frame_delay": 0.15,
        }
    }
    monkeypatch.setattr(s2_keyframes, "settings", lambda: config)
    monkeypatch.setattr(
        s2_keyframes,
        "_frame_record",
        lambda **values: values,
    )

    frames = s2_keyframes.from_semantic_cues(
        tmp_path / "video.mp4", "video-1", [segment], 20.0, force=True
    )

    assert [frame["suffix"].split("-")[1] for frame in frames] == [
        "statut",
        "occupation",
        "prestation",
    ]


def test_stale_keyframes_are_removed_after_manifest_write(tmp_path, monkeypatch):
    directory = tmp_path / "video-1" / "keyframes"
    directory.mkdir(parents=True)
    kept = directory / "kept.png"
    stale = directory / "stale.png"
    kept.write_bytes(b"kept")
    stale.write_bytes(b"stale")
    monkeypatch.setattr(
        s2_keyframes, "work_dir", lambda _video_id: tmp_path / "video-1"
    )

    s2_keyframes._remove_unreferenced_frames(
        "video-1", [{"file": "data/interim/videos/video-1/keyframes/kept.png"}]
    )

    assert kept.exists()
    assert not stale.exists()


def test_vlm_prompt_requires_the_exact_screen_analysis_shape():
    rendered = s3_describe._prompt_for(
        {"text": "Cliquez sur Modifier.", "moment": "before"}
    )

    assert '"ui_element"' in rendered
    assert '"confident"' in rendered
    assert "exactement ces huit cles" in rendered
    assert "ne retourne jamais de liste" in rendered
    assert "expected_result est toujours null" in rendered


def test_vlm_semantic_guard_rejects_non_visual_claims():
    frame = {"moment": "after", "text": "Cliquez sur la reservation."}
    analysis = {
        "screen_name": "Dossier",
        "ui_element": "le bouton Modifier",
        "action_type": "click",
        "expected_result": "Modification de la reservation",
        "visual_description": "Cliquez sur la date pour modifier la reservation.",
        "confident": True,
    }

    guarded = s3_describe._semantic_guard(frame, analysis)

    assert guarded["confident"] is False
    assert guarded["expected_result"] is None
    assert guarded["visual_description"] is None
    assert "libelle exact" in guarded["uncertainty_reason"]


def test_vlm_semantic_guard_discards_unrelated_date_target():
    frame = {
        "moment": "before",
        "text": "Cliquez sur la reservation depuis le planning.",
    }
    analysis = {
        "screen_name": "PLANNING",
        "ui_element": "22",
        "location": "cellule du calendrier",
        "action_type": "click",
        "visual_description": "Le planning est visible.",
        "confident": True,
    }

    guarded = s3_describe._semantic_guard(frame, analysis)

    assert guarded["ui_element"] is None
    assert guarded["location"] is None
    assert guarded["confident"] is False
    assert "sans rapport" in guarded["uncertainty_reason"]


def test_vlm_semantic_guard_discards_unrelated_visible_label():
    frame = {
        "moment": "before",
        "text": "Ajoutez une prestation complementaire.",
    }
    analysis = {
        "screen_name": "Dossier",
        "ui_element": "Chambre no: 25",
        "location": "a droite",
        "action_type": "click",
        "visual_description": "Le dossier est visible.",
        "confident": True,
    }

    guarded = s3_describe._semantic_guard(frame, analysis)

    assert guarded["ui_element"] is None
    assert guarded["location"] is None
    assert guarded["confident"] is False
    assert "sans correspondance" in guarded["uncertainty_reason"]


def test_untrusted_visual_does_not_reach_assembler_as_exact_target():
    analysis = s4_assemble._analysis(
        {
            "analysis": {
                "screen_name": "Dossier",
                "ui_element": "Simple",
                "location": "a droite",
                "expected_result": "Modification",
                "confident": False,
            }
        }
    )

    assert analysis["screen_name"] is None
    assert analysis["ui_element"] is None
    assert analysis["location"] is None
    assert analysis["expected_result"] is None


def test_assembler_removes_claims_without_trusted_evidence():
    group = {
        "text": "Les reservations peuvent etre modifiees selon les droits.",
        "before": {
            "analysis": {
                "screen_name": "Dossier",
                "ui_element": "Historique",
                "confident": False,
            }
        },
        "after": {
            "analysis": {
                "screen_name": "Historique",
                "ui_element": "Historique",
                "confident": False,
            }
        },
        "scenes": [],
    }
    draft = StepDraft(
        instruction="Cliquez sur Historique.",
        action_type="click",
        action_target="Historique",
        action_location="en haut a droite",
        before_state="Dossier ouvert",
        after_state="Historique affiche",
        visual_description="Le bouton Historique est visible.",
        confident=True,
    )

    guarded = s4_assemble._guard_draft(group, draft)

    assert guarded.instruction == group["text"]
    assert guarded.action_type == "none"
    assert guarded.action_target is None
    assert guarded.action_location is None
    assert guarded.before_state is None
    assert guarded.after_state is None
    assert guarded.visual_description is None
    assert guarded.confident is False
    assert "unsupported_action_target" in guarded.quality_flags
    assert "unsupported_action_type" in guarded.quality_flags


def test_quality_flag_always_prevents_automatic_confidence():
    assert s4_assemble._final_confidence(True, True, []) is True
    assert s4_assemble._final_confidence(True, True, ["manual_check"]) is False
    assert s4_assemble._final_confidence(True, False, []) is False


def test_assembler_uses_visual_location_when_audio_direction_conflicts():
    trusted = {
        "analysis": {
            "screen_name": "Dossier",
            "ui_element": None,
            "location": "a droite de la reservation",
            "action_type": "none",
            "expected_result": None,
            "visual_description": "Quatre icones sont visibles a droite.",
            "confident": True,
            "uncertainty_reason": None,
        }
    }
    group = {
        "text": "Tout a gauche se trouvent des boutons.",
        "before": trusted,
        "after": trusted,
        "scenes": [],
    }
    draft = StepDraft(
        instruction="Reperez les boutons.",
        action_type="none",
        action_location="tout a gauche",
        confident=True,
    )

    guarded = s4_assemble._guard_draft(group, draft)

    assert guarded.action_location == "a droite de la reservation"
    assert "audio_visual_location_conflict" in guarded.quality_flags


def test_assembler_fills_location_from_visual_when_audio_omits_it():
    trusted = {
        "analysis": {
            "screen_name": "Dossier",
            "ui_element": None,
            "location": "sous la colonne Prestations",
            "action_type": "none",
            "expected_result": None,
            "visual_description": "Une icone est visible.",
            "confident": True,
            "uncertainty_reason": None,
        }
    }
    group = {
        "text": "Vous pouvez utiliser ce bouton.",
        "before": trusted,
        "after": trusted,
        "scenes": [],
    }
    draft = StepDraft(
        instruction="Utilisez ce bouton.",
        action_type="none",
        action_location=None,
        confident=True,
    )

    guarded = s4_assemble._guard_draft(group, draft)

    assert guarded.action_location == "sous la colonne Prestations"


def test_review_requires_explicit_override_for_quality_flags(tmp_path, monkeypatch):
    step = _step(quality_flags=["missing_exact_ui_label"])
    monkeypatch.setattr(review, "load", lambda video_id: [step])
    monkeypatch.setattr(review, "output_file", lambda video_id: tmp_path / "steps.json")

    with pytest.raises(ValueError, match="alertes"):
        review.update_status("video-1", "verified", reviewer="manager")

    updated = review.update_status(
        "video-1",
        "verified",
        reviewer="manager",
        allow_quality_flags=True,
    )
    assert updated[0].status == "verified"
    assert updated[0].reviewed_by == "manager"


def test_export_contains_only_human_verified_steps(tmp_path, monkeypatch):
    root = tmp_path
    screenshot = root / "data/interim/videos/video-1/keyframes/before.png"
    screenshot.parent.mkdir(parents=True)
    screenshot.write_bytes(b"png")
    verified = _step(
        step_number=3,
        status="verified",
        reviewed_by="manager",
        reviewed_at=datetime.now(timezone.utc),
        screenshot_before="data/interim/videos/video-1/keyframes/before.png",
    )
    pending = _step(step_number=2)
    second_verified = _step(
        step_number=7,
        timestamp_start=20.0,
        timestamp_end=22.0,
        status="verified",
        reviewed_by="manager",
        reviewed_at=datetime.now(timezone.utc),
    )
    steps_file = root / "data/interim/videos/video-1/steps.json"
    steps_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "video_id": "video-1",
                "steps": [
                    verified.model_dump(mode="json"),
                    pending.model_dump(mode="json"),
                    second_verified.model_dump(mode="json"),
                ],
            }
        ),
        encoding="utf-8",
    )

    fake_path = lambda value: root / value
    fake_artifact = lambda value: (
        Path(value) if Path(value).is_absolute() else root / value
    )
    fake_relative = lambda value: value.resolve().relative_to(root.resolve()).as_posix()
    monkeypatch.setattr(exporter, "path", fake_path)
    monkeypatch.setattr(exporter, "artifact_path", fake_artifact)
    monkeypatch.setattr(exporter, "project_relative", fake_relative)

    output, documents, report = exporter.export_verified()
    assert output.exists()
    assert len(documents) == 2
    assert all(document.status == "verified" for document in documents)
    assert [document.breadcrumb for document in documents] == [
        "modifier-reservation > etape 1",
        "modifier-reservation > etape 2",
    ]
    assert report["documents"] == 2
    assert report["source_statuses"] == {
        "verified": 2,
        "review_required": 1,
        "deprecated": 0,
    }
    assert (root / documents[0].screenshot).exists()
