"""Check the published FEM table and its held-out position splits."""
import csv
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/generated/fem_dataset.csv"
META = ROOT / "data/generated/fem_dataset.json"


def test_fem_dataset_contract():
    metadata = json.loads(META.read_text())
    with CSV.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert sha256(CSV.read_bytes()).hexdigest() == metadata["csv_sha256"]
    assert sha256((ROOT / "src/mass_changes.py").read_bytes()).hexdigest() == metadata["model_sha256"]
    assert metadata["dataset_type"] == "synthetic_fem"
    assert metadata["experimental_data_included"] is False
    assert len(rows) == metadata["rows_total"] == 1176

    expected_pairs = {(p, m) for p in metadata["positions_cm"] for m in metadata["masses_g"]}
    actual_pairs = {(int(row["position_cm"]), float(row["mass_g"])) for row in rows}
    assert actual_pairs == expected_pairs
    assert len(actual_pairs) == len(rows)

    split_positions = {split: set(positions) for split, positions in metadata["split_positions_cm"].items()}
    assert set.union(*split_positions.values()) == set(metadata["positions_cm"])
    assert sum(map(len, split_positions.values())) == len(metadata["positions_cm"])
    for split, positions in split_positions.items():
        selected = [row for row in rows if row["split"] == split]
        assert len(selected) == metadata["rows_by_split"][split]
        assert {int(row["position_cm"]) for row in selected} == positions

    reference = metadata["reference_frequencies_hz"]
    tolerance = metadata["numerical_monotonicity_tolerance_hz"]
    for row in rows:
        assert int(row["node"]) == int(row["position_cm"])
        frequencies = [float(row[f"f{i}_hz"]) for i in range(1, 7)]
        assert all(0 < a < b for a, b in zip(frequencies, frequencies[1:]))
        assert all(value <= base + tolerance for value, base in zip(frequencies, reference))
