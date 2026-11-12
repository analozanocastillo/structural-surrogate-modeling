"""Compare matched-budget grid and Latin hypercube FEM designs.

The original FEM dataset and model are read only. Experimental measurements are
used solely after training, never to choose epochs or design points.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import qmc
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def grid_points(source: pd.DataFrame, positions: list[int], masses: list[float], budget: int) -> pd.DataFrame:
    n_positions = budget // 10
    chosen_positions = np.asarray(positions)[np.rint(np.linspace(0, len(positions) - 1, n_positions)).astype(int)]
    chosen_masses = np.asarray(masses)[np.rint(np.linspace(0, len(masses) - 1, 10)).astype(int)]
    assert len(np.unique(chosen_positions)) == n_positions
    assert len(np.unique(chosen_masses)) == 10
    columns = ['position_cm', 'mass_g'] + [f'f{i}_hz' for i in range(1, 7)]
    points = source.loc[source.position_cm.isin(chosen_positions) & source.mass_g.isin(chosen_masses), columns].copy()
    assert len(points) == budget
    return points.sort_values(['position_cm', 'mass_g']).reset_index(drop=True)


def lhd_points(fem, positions: list[int], budget: int, design_seed: int, targets: list[str]) -> pd.DataFrame:
    # Four common corner anchors keep boundary coverage comparable with the grid.
    sampler = qmc.LatinHypercube(d=2, scramble=True, rng=np.random.default_rng(design_seed))
    u = sampler.random(n=budget - 4)
    allowed = np.asarray(positions)
    nodes = allowed[np.floor(u[:, 0] * len(allowed)).astype(int)]
    grams = 1.0 + 49.0 * u[:, 1]
    pairs = [(int(node), float(mass)) for node, mass in zip(nodes, grams)]
    pairs += [(int(node), float(mass)) for node in (allowed[0], allowed[-1]) for mass in (1.0, 50.0)]
    assert len(pairs) == budget and len(set(pairs)) == budget
    values = np.array([fem.frequencies(node, mass / 1000) for node, mass in pairs])
    points = pd.DataFrame(pairs, columns=['position_cm', 'mass_g'])
    points[targets] = values
    return points.sort_values(['position_cm', 'mass_g']).reset_index(drop=True)


def train_mlp(points: pd.DataFrame, validation: pd.DataFrame, features: list[str],
              targets: list[str], seed: int) -> tuple[nn.Module, dict, int, float]:
    x = points[features].to_numpy(dtype=np.float32)
    y = points[targets].to_numpy(dtype=np.float32)
    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(0), y.std(0)
    assert np.all(x_std > 0) and np.all(y_std > 0)
    scaler = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    train_x = torch.from_numpy((x - x_mean) / x_std)
    train_y = torch.from_numpy((y - y_mean) / y_std)
    val_x = torch.from_numpy((validation[features].to_numpy(dtype=np.float32) - x_mean) / x_std)
    val_y = torch.from_numpy((validation[targets].to_numpy(dtype=np.float32) - y_mean) / y_std)

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 6))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_loss, best_epoch, best_state = float('inf'), 0, None
    for epoch in range(1, 1501):
        net.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(net(xb), yb)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(net(val_x), val_y).item()
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, best_state = val_loss, epoch, deepcopy(net.state_dict())
        if epoch - best_epoch >= 120:
            break
    net.load_state_dict(best_state)
    net.eval()
    return net, scaler, best_epoch, best_loss


def predict_mlp(net: nn.Module, scaler: dict, x: np.ndarray) -> np.ndarray:
    x_scaled = (x.astype(np.float32) - scaler['x_mean']) / scaler['x_std']
    with torch.no_grad():
        result = net(torch.from_numpy(x_scaled)).numpy()
    return result * scaler['y_std'] + scaler['y_mean']


def experimental_cases(root: Path, fem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = root / 'data/extracted/Code_Veröffentlichung/Data/ModalProperties/LiveBeamModalData.mat'
    modal = loadmat(path)
    cases = modal['mfMassCase']
    added = np.unique(cases[cases[:, 1] > 0], axis=0)
    assert len(added) == 34
    measured_shift = np.asarray([
        (100 * (modal['mfFreqAna'][np.all(cases == case, axis=1)] /
                modal['mfFreqRef'][np.all(cases == case, axis=1)] - 1)).mean(axis=0)
        for case in added])
    fem_ref = fem.frequencies()[1:4]
    fem_hz = np.asarray([fem.frequencies(int(node), grams / 1000)[1:4]
                         for node, grams in added])
    return added.astype(np.float32), measured_shift, fem_ref, fem_hz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.project.resolve()
    out = args.output or root / 'data/generated/doe_comparison'
    out.mkdir(parents=True, exist_ok=True)
    dataset = root / 'data/generated/fem_dataset.csv'
    metadata = json.loads(dataset.with_suffix('.json').read_text())
    assert digest(dataset) == metadata['csv_sha256']
    assert digest(root / 'src/mass_changes.py') == metadata['model_sha256']
    beam = root / 'data/extracted/Code_Veröffentlichung/Data/BeamData/LiveBeamData.mat'
    assert digest(beam) == metadata['mat_sha256']
    sys.path.insert(0, str(root / 'src'))
    from mass_changes import MassChanges
    fem = MassChanges.from_mat(beam)
    source = pd.read_csv(dataset)
    features, targets = metadata['input_columns'], metadata['target_columns']
    split = {name: source.loc[source.split.eq(name)].copy()
             for name in ('train', 'val', 'test', 'test_edge')}
    positions = metadata['split_positions_cm']['train']
    masses = metadata['masses_g']
    exp_x, exp_measured_shift, fem_ref, exp_fem_hz = experimental_cases(root, fem)
    exp_fem_shift = 100 * (exp_fem_hz / fem_ref - 1)
    torch.set_num_threads(2)

    seeds = (11, 22, 33)
    budgets = (200, 400)
    design_rows, test_rows, exp_rows, run_rows = [], [], [], []

    def evaluate(design: str, budget: int, seed: int, model_name: str, predict, epoch: int | None):
        for name in ('val', 'test', 'test_edge'):
            frame = split[name]
            actual = frame[targets].to_numpy(dtype=np.float32)
            estimate = predict(frame[features].to_numpy(dtype=np.float32))
            for mode, error in enumerate(np.abs(estimate - actual).mean(0), start=1):
                test_rows.append(dict(design=design, budget=budget, seed=seed, model=model_name,
                                      split=name, mode=mode, mae_hz=float(error), best_epoch=epoch))
        exp_pred = predict(exp_x)[:, 1:4]
        exp_pred_shift = 100 * (exp_pred / fem_ref - 1)
        for j, mode in enumerate((2, 3, 4)):
            exp_rows.append(dict(design=design, budget=budget, seed=seed, model=model_name,
                                 mode=mode,
                                 surrogate_fem_mae_pp=float(np.abs(exp_pred_shift[:, j] - exp_fem_shift[:, j]).mean()),
                                 fem_experiment_mae_pp=float(np.abs(exp_fem_shift[:, j] - exp_measured_shift[:, j]).mean()),
                                 surrogate_experiment_mae_pp=float(np.abs(exp_pred_shift[:, j] - exp_measured_shift[:, j]).mean())))

    print('Training original 966-case grid baseline', flush=True)
    start = time.monotonic()
    net, scaler, epoch, val_loss = train_mlp(split['train'], split['val'], features, targets, 42)
    run_rows.append(dict(design='original_grid', budget=966, seed=42, model='MLP',
                         best_epoch=epoch, val_mse_scaled=val_loss, seconds=time.monotonic()-start))
    evaluate('original_grid', 966, 42, 'MLP', lambda x: predict_mlp(net, scaler, x), epoch)

    for budget in budgets:
        grid = grid_points(split['train'], positions, masses, budget)
        grid['design'], grid['budget'], grid['design_seed'] = 'grid', budget, 0
        design_rows.append(grid)
        for seed in seeds:
            lhd = lhd_points(fem, positions, budget, seed, targets)
            lhd['design'], lhd['budget'], lhd['design_seed'] = 'lhd', budget, seed
            design_rows.append(lhd)
            for design, points in (('grid', grid), ('lhd', lhd)):
                assert len(points) == budget
                assert not set(points.position_cm).intersection(
                    metadata['split_positions_cm']['val'] +
                    metadata['split_positions_cm']['test'] +
                    metadata['split_positions_cm']['test_edge'])
                start = time.monotonic()
                net, scaler, epoch, val_loss = train_mlp(points, split['val'], features, targets, seed)
                seconds = time.monotonic() - start
                run_rows.append(dict(design=design, budget=budget, seed=seed, model='MLP',
                                     best_epoch=epoch, val_mse_scaled=val_loss, seconds=seconds))
                evaluate(design, budget, seed, 'MLP', lambda x: predict_mlp(net, scaler, x), epoch)
                print(f'{design} {budget} seed {seed}: epoch {epoch}, {seconds:.1f} s', flush=True)

    pd.concat(design_rows, ignore_index=True).to_csv(out / 'design_points.csv', index=False)
    pd.DataFrame(test_rows).to_csv(out / 'test_metrics.csv', index=False)
    pd.DataFrame(exp_rows).to_csv(out / 'experimental_metrics.csv', index=False)
    pd.DataFrame(run_rows).to_csv(out / 'training_runs.csv', index=False)
    manifest = {
        'purpose': 'Matched-budget grid versus LHD on the unchanged Hannover FEM',
        'models_compared': ['MLP'],
        'csv_sha256': metadata['csv_sha256'], 'mat_sha256': metadata['mat_sha256'],
        'model_sha256': metadata['model_sha256'],
        'budgets': budgets, 'seeds': seeds,
        'grid': '20 or 40 evenly spaced allowed nodes x 10 evenly spaced original mass levels; includes four corners',
        'lhd': 'SciPy LatinHypercube d=2, scramble=True, rng=seed, budget-4 points plus four corners; positions quantized to allowed integer FEM nodes, masses continuous 1-50 g',
        'validation_positions_cm': metadata['split_positions_cm']['val'],
        'test_positions_cm': metadata['split_positions_cm']['test'],
        'edge_test_position_cm': metadata['split_positions_cm']['test_edge'],
        'mlp': '2-32-32-6 tanh, Adam 0.003, batch 128, max 1500 epochs, patience 120, train-only scaling, validation checkpoint',
        'experimental_data_role': 'evaluation only; 34 added-mass cases, paired shift estimates averaged per case',
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                     'scipy': __import__('scipy').__version__, 'torch': torch.__version__,
                     'pandas': pd.__version__},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Saved', out, flush=True)


if __name__ == '__main__':
    main()
