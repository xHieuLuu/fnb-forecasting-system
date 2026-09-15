import json

from fnb_forecast.data.io import load_csv_bundle, save_bundle


def test_bundle_round_trip_preserves_provenance(valid_bundle, tmp_path) -> None:
    save_bundle(valid_bundle, tmp_path)

    loaded = load_csv_bundle(tmp_path, valid_bundle.provenance)

    assert loaded.provenance == valid_bundle.provenance
    assert loaded.daily_sales.equals(valid_bundle.daily_sales)


def test_save_bundle_writes_provenance_json(valid_bundle, tmp_path) -> None:
    save_bundle(valid_bundle, tmp_path)

    saved = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))

    assert saved["name"] == "fixture"
    assert saved["is_synthetic"] is False
