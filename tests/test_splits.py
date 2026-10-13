import json

from music_project.data import splits
from music_project.io import write_jsonl


def test_quarantine_keeps_original_assignments_and_removes_cross_split_groups(
    tmp_path, monkeypatch
):
    assignments = [
        ("train", "ambiguous"),
        ("validation", "ambiguous"),
        ("train", "unique-train"),
        ("validation", "unique-val"),
        ("test", "unique-test"),
    ]
    rows = [
        {
            "split": split,
            "composition_key": ["composer", composition],
            "sha256": str(index),
            "parse_status": "ok",
            "usable": True,
            "filtering_reasons": [],
            "relative_path": f"{index}.mid",
        }
        for index, (split, composition) in enumerate(assignments)
    ]
    source, output = tmp_path / "original.jsonl", tmp_path / "clean.jsonl"
    write_jsonl(source, rows)
    original_bytes = source.read_bytes()
    monkeypatch.setattr(splits, "EXPECTED_SPLITS", {"train": 2, "validation": 2, "test": 1})
    report = splits.prepare_splits(source, output, tmp_path / "report.json")
    cleaned = [json.loads(line) for line in output.read_text().splitlines()]
    assert source.read_bytes() == original_bytes
    assert [row["split"] for row in cleaned] == [row["split"] for row in rows]
    assert [row["usable"] for row in cleaned] == [False, False, True, True, True]
    assert report["retained_record_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["retained_metadata_overlap_groups"] == 0
