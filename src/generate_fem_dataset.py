"""Generate the Hannover cantilever FEM dataset for a future MLP.

Only synthetic labels from src/mass_changes.py are written. No measured modal
frequencies enter the training table.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from mass_changes import MassChanges  # noqa: E402

POSITIONS_CM = tuple(range(10, 121, 2))
MASSES_G = (1.0,) + tuple(2.5 * i for i in range(1, 21))
VALIDATION_POSITIONS = frozenset({30, 50, 70, 90, 110})
TEST_POSITIONS = frozenset({40, 60, 80, 100})
EDGE_TEST_POSITIONS = frozenset({120})
COLUMNS = ["position_cm", "node", "mass_g", "split"] + [f"f{i}_hz" for i in range(1, 7)]
DEFAULT_MAT = PROJECT / "data/extracted/Code_Veröffentlichung/Data/BeamData/LiveBeamData.mat"
DEFAULT_OUTPUT = PROJECT / "data/generated/fem_dataset.csv"


def split_for_position(position_cm: int) -> str:
    if position_cm in VALIDATION_POSITIONS:
        return "val"
    if position_cm in TEST_POSITIONS:
        return "test"
    if position_cm in EDGE_TEST_POSITIONS:
        return "test_edge"
    return "train"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(mat_path: Path, output_path: Path) -> dict:
    model = MassChanges.from_mat(mat_path)
    if len(model.rho_l) != 120 or not np.isclose(model.length_m, 1.20):
        raise ValueError("The position_cm = node mapping requires the 120-element, 1.20 m Hannover model")

    reference = model.frequencies()
    rows = []
    max_positive_delta_hz = 0.0
    counts = {"train": 0, "val": 0, "test": 0, "test_edge": 0}
    for position_cm in POSITIONS_CM:
        split = split_for_position(position_cm)
        for mass_g in MASSES_G:
            frequencies = model.frequencies(node=position_cm, mass_kg=mass_g / 1000)
            if not (np.all(np.isfinite(frequencies)) and np.all(np.diff(frequencies) > 0)):
                raise ValueError(f"Invalid frequencies at {position_cm} cm, {mass_g} g")
            max_positive_delta_hz = max(max_positive_delta_hz, float(np.max(frequencies - reference)))
            if np.any(frequencies > reference + 1e-5):
                raise ValueError(f"Added mass raised a frequency beyond numerical tolerance at {position_cm} cm, {mass_g} g")
            rows.append([position_cm, position_cm, f"{mass_g:g}", split,
                         *[f"{frequency:.10f}" for frequency in frequencies]])
            counts[split] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    split_positions = {split: [p for p in POSITIONS_CM if split_for_position(p) == split]
                       for split in counts}
    metadata = {
        "dataset_type": "synthetic_fem",
        "source": "Hannover MassChanges.m translated by src/mass_changes.py",
        "input_columns": ["position_cm", "mass_g"],
        "target_columns": [f"f{i}_hz" for i in range(1, 7)],
        "auxiliary_columns": ["node", "split"],
        "position_mapping": "Following the source MATLAB convention, the centimetre position label is passed as the model node index.",
        "positions_cm": list(POSITIONS_CM),
        "masses_g": list(MASSES_G),
        "split_positions_cm": split_positions,
        "rows_by_split": counts,
        "rows_total": len(rows),
        "reference_frequencies_hz": [float(value) for value in reference],
        "max_positive_delta_hz": max_positive_delta_hz,
        "numerical_monotonicity_tolerance_hz": 1e-5,
        "mat_sha256": file_sha256(mat_path),
        "model_sha256": file_sha256(PROJECT / "src/mass_changes.py"),
        "csv_sha256": file_sha256(output_path),
        "experimental_data_included": False,
        "notes": "test holds measured interior positions; test_edge holds the 120 cm boundary. All splits share the same mass values.",
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT,
                        help="Official Hannover LiveBeamData.mat")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="CSV output; metadata is written beside it as JSON")
    args = parser.parse_args()
    metadata = generate(args.mat, args.output)
    print(f"Wrote {metadata['rows_total']} FEM cases to {args.output}")
    print("Rows by split:", metadata["rows_by_split"])


if __name__ == "__main__":
    main()
