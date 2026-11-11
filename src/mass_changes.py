"""Port faithful to Hannover's MassChanges.m (Euler–Bernoulli modal FEM).

The original MATLAB code is CC BY 4.0 and credits Ragnitz et al.
This independent Python translation preserves its parameters, fixed point masses,
first-six-node clamping and generalized eigenvalue formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.linalg import eigh


@dataclass
class MassChanges:
    rho_l: np.ndarray  # [kg/m], one entry per finite element
    fixture_nodes: int = 6
    length_m: float = 1.20
    thickness_m: float = 0.003
    density_kg_m3: float = 7850.0
    young_pa: float = 2.05e11

    # Original MATLAB: [original (1-based) node number, point mass (kg)]
    FIXED_MASSES = ((73, 0.005), (95, 0.005), (117, 0.005),
                    (120, 0.005), (22, 0.0206))

    def __post_init__(self):
        self.rho_l = np.asarray(self.rho_l, dtype=float).reshape(-1)
        if len(self.rho_l) <= self.fixture_nodes or np.any(self.rho_l <= 0):
            raise ValueError('rho_l must contain > fixture_nodes positive values')

    @classmethod
    def from_mat(cls, path: str | Path) -> 'MassChanges':
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'{path} not found: copy LiveBeamData.mat from the Hannover dataset')
        data = loadmat(path)
        if 'rhoLs' not in data:
            raise KeyError(f"{path}: missing 'rhoLs' variable")
        # MATLAB also loads EAs, but the original calculate() never uses it.
        return cls(rho_l=data['rhoLs'])

    def assemble_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (K, M), clamped identically to original MATLAB."""
        n = len(self.rho_l)
        le = self.length_m / n
        width = self.rho_l / (self.density_kg_m3 * self.thickness_m)
        inertia = width * self.thickness_m**3 / 12.0
        ei = self.young_pa * inertia
        ke_unit = np.array([
            [12, 6*le, -12, 6*le],
            [6*le, 4*le**2, -6*le, 2*le**2],
            [-12, -6*le, 12, -6*le],
            [6*le, 2*le**2, -6*le, 4*le**2]
        ]) / le**3
        me_unit = np.array([
            [156, 22*le, 54, -13*le],
            [22*le, 4*le**2, 13*le, -3*le**2],
            [54, 13*le, 156, -22*le],
            [-13*le, -3*le**2, -22*le, 4*le**2]
        ]) * (le / 420)
        ndof = (n+1)*2
        k = np.zeros((ndof, ndof))
        m = np.zeros((ndof, ndof))
        for i in range(n):
            ix = slice(2*i, 2*i+4)
            k[ix, ix] += ke_unit * ei[i]
            m[ix, ix] += me_unit * self.rho_l[i]
        start = self.fixture_nodes * 2
        return k[start:, start:], m[start:, start:]

    def _add_point_mass(self, m: np.ndarray, original_node: int, kg: float) -> None:
        """Original MATLAB node numbering is 1-based; only translation DOF gains mass."""
        node = int(np.floor(original_node + .5))  # MATLAB round for positive nodes
        index = 2 * (node - self.fixture_nodes - 1)  # Python zero-based DOF index
        if index < 0 or index >= m.shape[0]:
            raise ValueError(f'original_node {node} out of free-DOF range')
        if kg < 0:
            raise ValueError('Additional mass must be non-negative')
        m[index, index] += kg

    def frequencies(self, node: int = 7, mass_kg: float = 0.0, n_modes: int = 6) -> np.ndarray:
        """First n natural frequencies [Hz] including original fixed and optional point masses."""
        k, m = self.assemble_matrices()
        for fixed_node, fixed_mass in self.FIXED_MASSES:
            self._add_point_mass(m, fixed_node, fixed_mass)
        self._add_point_mass(m, node, mass_kg)
        if not 1 <= n_modes <= k.shape[0]:
            raise ValueError('Invalid n_modes')
        # Dense symmetric generalized solver is robust for the ~230 DOFs of this case.
        eigenvalues = eigh(k, m, eigvals_only=True, subset_by_index=[0, n_modes-1])
        if np.any(eigenvalues <= 0):
            raise ValueError('Non-positive eigenvalues: check FEM and boundary conditions')
        return np.sqrt(eigenvalues) / (2*np.pi)

    def objective(self, node: int, mass_kg: float, measured_reference: np.ndarray,
                  measured_modified: np.ndarray, mode_indices: tuple[int, ...] = (1,2,3)) -> float:
        """Original relative-frequency-shift mismatch; mode_indices are 1-based."""
        mode_ix = np.asarray(mode_indices, dtype=int) - 1
        if len(mode_ix) == 0 or np.any(mode_ix < 0) or np.any(mode_ix >= 6):
            raise ValueError('mode_indices must be between 1 and 6')
        exp_ref = np.asarray(measured_reference, dtype=float).reshape(-1)
        exp_mod = np.asarray(measured_modified, dtype=float).reshape(-1)
        if exp_ref.size != mode_ix.size or exp_mod.size != mode_ix.size:
            raise ValueError('Measured frequency arrays must match selected modes')
        sim_ref = self.frequencies()[mode_ix]
        sim_mod = self.frequencies(node, mass_kg)[mode_ix]
        return float(np.linalg.norm((sim_mod-sim_ref)/sim_ref - (exp_mod-exp_ref)/exp_ref))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hannover cantilever beam modal solver')
    parser.add_argument('--mat', required=True, help='Path to LiveBeamData.mat')
    parser.add_argument('--node', type=int, default=7, help='Original MATLAB 1-based node index')
    parser.add_argument('--mass-g', type=float, default=0.0, help='Variable added mass in grams')
    args = parser.parse_args()
    model = MassChanges.from_mat(args.mat)
    print('Reference frequencies [Hz]:', np.round(model.frequencies(), 5))
    print('Modified frequencies [Hz]:', np.round(model.frequencies(args.node, args.mass_g/1000), 5))
