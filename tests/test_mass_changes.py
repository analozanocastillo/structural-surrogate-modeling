import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from mass_changes import MassChanges


def test_modal_solver_without_hannover_file():
    # Synthetic material distribution, solely a numerical smoke test.
    model = MassChanges(rho_l=np.full(120, 0.20))
    baseline = model.frequencies()
    assert baseline.shape == (6,)
    assert np.all(np.isfinite(baseline))
    assert np.all(np.diff(baseline) > 0)
    np.testing.assert_allclose(baseline, model.frequencies(7, 0))
    modified = model.frequencies(80, 0.01)
    assert np.all(modified <= baseline + 1e-7)
    assert np.any(modified < baseline - 1e-5)
