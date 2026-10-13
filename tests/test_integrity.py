import zipfile

import pytest

from music_project.data.download import safe_extract
from music_project.io import write_jsonl


def test_manifest_cannot_be_silently_rewritten(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_jsonl(path, [{"split": "train"}], immutable=True)
    write_jsonl(path, [{"split": "train"}], immutable=True)
    with pytest.raises(FileExistsError):
        write_jsonl(path, [{"split": "test"}], immutable=True)


def test_zip_paths_cannot_escape_destination(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escaped.txt", "test")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        safe_extract(archive, tmp_path / "destination")
    assert not (tmp_path / "escaped.txt").exists()
