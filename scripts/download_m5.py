"""Explicitly download and checksum the public M5 benchmark files."""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

EXPECTED_FILES = ("sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv")


def _extract_archive(archive: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive) as downloaded:
        for member in downloaded.infolist():
            target = (output_root / member.filename).resolve()
            if target != output_root and output_root not in target.parents:
                raise ValueError(f"archive member escapes output directory: {member.filename}")
        downloaded.extractall(output_root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_m5(output_dir: Path) -> Path:
    """Run the official Kaggle CLI, unzip under ``output_dir``, and write checksums."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            "m5-forecasting-accuracy",
            "-p",
            str(output_dir),
        ],
        check=True,
    )
    archive = output_dir / "m5-forecasting-accuracy.zip"
    if not archive.exists():
        raise FileNotFoundError(f"Kaggle CLI did not create {archive}")
    _extract_archive(archive, output_dir)
    checksums = {}
    for filename in EXPECTED_FILES:
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"M5 archive did not contain {filename}")
        checksums[filename] = _sha256(path)
    (output_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    download_m5(arguments.output_dir)
