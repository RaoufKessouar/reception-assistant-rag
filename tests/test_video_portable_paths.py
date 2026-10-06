from src.ingestion.hotel_software.generic_connector.videos import common, exporter


def test_data_mount_is_exported_with_portable_path(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    data_root = tmp_path / "fast-data"
    file = data_root / "processed" / "frame.png"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"png")
    project_root.mkdir()

    def fake_path(value):
        if value == "":
            return project_root
        if value == "data":
            return data_root
        return project_root / value

    monkeypatch.setattr(common, "path", fake_path)
    monkeypatch.setattr(exporter, "path", fake_path)

    assert common.project_relative(file) == "data/processed/frame.png"
    assert exporter._archive_name(file, project_root) == "data/processed/frame.png"
