"""Deterministic, atomic output and content identities."""

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path: Path, content: bytes, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to change immutable artifact: {path}")
        return
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path: Path, content: object, *, immutable: bool = False) -> None:
    write_bytes(
        path,
        (json.dumps(content, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
        immutable=immutable,
    )


def write_jsonl(path: Path, rows: list[dict], *, immutable: bool = False) -> None:
    content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    write_bytes(path, content.encode("utf-8"), immutable=immutable)
