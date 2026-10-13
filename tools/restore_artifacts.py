"""Verify and restore downloaded private-release ZIPs into this repository."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("downloads"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "submission/artifacts-manifest.json").read_text())
    for asset in manifest["assets"]:
        path = args.downloads / asset["name"]
        if (
            not path.is_file()
            or path.stat().st_size != asset["bytes"]
            or digest(path) != asset["sha256"]
        ):
            raise ValueError("Missing or changed release asset: " + asset["name"])
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                destination = (root / member.filename).resolve()
                if not destination.is_relative_to(root) or member.filename.startswith(("/", "\\")):
                    raise ValueError("Unsafe archive member")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if destination.exists():
                    saved = hashlib.sha256()
                    with archive.open(member) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            saved.update(block)
                    if digest(destination) != saved.hexdigest():
                        raise ValueError(
                            "Refusing to replace a changed local file: " + member.filename
                        )
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
        print("Restored " + asset["name"])
    print("Run scripts/setup_assets.py for the separately pinned upstream assets.")


if __name__ == "__main__":
    main()
