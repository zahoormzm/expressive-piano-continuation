import pytest

from music_project.io import write_jsonl
from music_project.tokens.prepare import prepare


def test_tokenization_refuses_retained_cross_split_compositions(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "usable": True,
                "split": "train",
                "composition_key": ["composer", "piece"],
                "sha256": "source-one",
            },
            {
                "usable": True,
                "split": "test",
                "composition_key": ["composer", "piece"],
                "sha256": "source-two",
            },
        ],
    )
    with pytest.raises(ValueError, match="crosses splits"):
        prepare(tmp_path, manifest, tmp_path / "tokens", None, 0)
