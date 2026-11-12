"""Build the dependency-free dashboard data file from tracked experiment results."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.project.resolve()
    out = args.output or root / 'docs/data.js'
    base = root / 'data/generated/doe_comparison'
    manifest = json.loads((base / 'manifest.json').read_text())
    source = root / 'data/generated/fem_dataset.csv'
    assert sha256(source.read_bytes()).hexdigest() == manifest['csv_sha256']
    assert manifest['models_compared'] == ['MLP']

    point_rows = read_csv(base / 'design_points.csv')
    test_rows = read_csv(base / 'test_metrics.csv')
    experimental_rows = read_csv(base / 'experimental_metrics.csv')
    assert len(point_rows) == 2400
    assert len(test_rows) == 234
    assert len(experimental_rows) == 39
    assert all(row['model'] == 'MLP' for row in test_rows + experimental_rows)

    data = {
        'sourceHash': manifest['csv_sha256'],
        'budgets': manifest['budgets'],
        'seeds': manifest['seeds'],
        'stats': {'simulatedCases': 1176, 'measuredCases': 34,
                  'predictedModes': 6, 'measuredModes': [2, 3, 4]},
        'protocol': {'validationPositions': manifest['validation_positions_cm'],
                     'testPositions': manifest['test_positions_cm'],
                     'edgePositions': manifest['edge_test_position_cm'],
                     'grid': manifest['grid'], 'lhd': manifest['lhd']},
        'points': [{
            'design': row['design'], 'budget': int(row['budget']),
            'seed': int(row['design_seed']), 'position': float(row['position_cm']),
            'mass': float(row['mass_g'])
        } for row in point_rows],
        'tests': [{
            'design': row['design'], 'budget': int(row['budget']),
            'seed': int(row['seed']), 'split': row['split'],
            'mode': int(row['mode']), 'maeHz': float(row['mae_hz'])
        } for row in test_rows],
        'experimental': [{
            'design': row['design'], 'budget': int(row['budget']),
            'seed': int(row['seed']), 'mode': int(row['mode']),
            'surrogateFemPp': float(row['surrogate_fem_mae_pp']),
            'femExperimentPp': float(row['fem_experiment_mae_pp']),
            'surrogateExperimentPp': float(row['surrogate_experiment_mae_pp'])
        } for row in experimental_rows],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('window.STRUCTURAL_DATA = ' + json.dumps(data, separators=(',', ':')) + ';\n')
    print(out, out.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
