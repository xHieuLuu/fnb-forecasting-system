"""Fixed-file CSV persistence for validated data bundles."""

import json
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd

from fnb_forecast.contracts import DataBundle, DataProvenance

from .validation import validate_bundle

TABLES = (
    "daily_sales",
    "item_master",
    "calendar",
    "weather",
    "recipes",
    "ingredient_master",
    "inventory_lots",
    "scheduled_receipts",
)


def canonical_bundle_checksum(bundle: DataBundle) -> str:
    """Return a SHA-256 checksum for canonically serialized bundle tables."""
    digest = sha256()
    for name in TABLES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
        csv_bytes = getattr(bundle, name).to_csv(
            index=False, lineterminator="\n", date_format="%Y-%m-%d"
        ).encode("utf-8")
        digest.update(csv_bytes)
    return digest.hexdigest()


def load_csv_bundle(root: Path, provenance: DataProvenance) -> DataBundle:
    """Load the eight required UTF-8 CSV tables and validate their contents."""
    root = Path(root)
    tables = {name: pd.read_csv(root / f"{name}.csv", encoding="utf-8") for name in TABLES}
    return validate_bundle(DataBundle(**tables, provenance=provenance))


def save_bundle(bundle: DataBundle, root: Path) -> None:
    """Save every table as UTF-8 CSV plus its provenance metadata."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        getattr(bundle, name).to_csv(
            root / f"{name}.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            date_format="%Y-%m-%d",
        )
    provenance = asdict(bundle.provenance)
    generated_at = provenance["generated_at"]
    if isinstance(generated_at, datetime):
        provenance["generated_at"] = generated_at.isoformat()
    (root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
