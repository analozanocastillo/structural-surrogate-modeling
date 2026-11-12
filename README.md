# Structural Surrogate Modeling

This project develops a workflow for machine learning surrogates of structural simulations. Its current case study examines how an added point mass changes the natural frequencies of a cantilever beam. It translates the finite element model (FEM) supplied with the [Leibniz Universität Hannover cantilever beam dataset](https://data.uni-hannover.de/en/dataset/cantilever-beam-dataset-for-structural-change-localisation) from MATLAB to Python, checks numerical agreement, and compares FEM predictions with measured modal frequencies.

The surrogate maps **mass and position** to natural frequencies. The repository includes a reproducible FEM training dataset and an executed PyTorch MLP notebook. The numerical FEM remains the reference for evaluating the surrogate; measured frequencies provide a separate physical benchmark.

**Explore the results without installing Python:** download the repository and open [`docs/index.html`](docs/index.html) in a browser. The interactive page contains the sampling designs, MLP errors, and measurement comparison. It is a static site with no external JavaScript dependencies; the same `docs/` folder can be served by GitHub Pages after publication.

## Research questions and workflow

1. **MATLAB to Python:** Does the Python implementation reproduce the original FEM matrices and modal frequencies for the same added-mass configurations?
2. **FEM to experiment:** How closely does the numerical model describe the measured frequency shifts when mass and position change?
3. **FEM to MLP:** How accurately does a PyTorch MLP reproduce the FEM response on positions excluded from training, and how does the training design affect that accuracy?

The original Hannover archive provides the MATLAB model and experimental data. `src/mass_changes.py` computes the FEM response; `notebooks/hannover_validacion_fem.ipynb` performs the validation and produces the figures. Experimental measurements are used for **evaluation**, not presented as synthetic FEM training data.

## Data and model

The [official dataset](https://doi.org/10.25835/dr0bu9oq) contains vibration measurements, identified modal properties, and a numerical beam model. The dataset authors describe measurements for 35 structural states. This repository's analysis uses the **34 added-mass configurations** found in the modal data, with reference measurements for the unmodified beam. Repeated modal estimates for a configuration are averaged before comparison, so each added-mass configuration contributes one comparison point.

The Python model follows Hannover's `MassChanges.m`: an Euler–Bernoulli beam, assembled stiffness and mass matrices, clamped boundary conditions, fixed point masses from the original model, and one variable point mass. It solves a generalized eigenvalue problem and returns the first **six natural frequencies in Hz** for a MATLAB-style one-based node index and an added mass in kilograms. Position in centimeters in the experimental analysis is mapped to the corresponding model node as in the source study.

The experimental modal file identifies three frequencies, compared here with **FEM modes 2, 3, and 4**. Absolute frequencies and changes relative to the reference state are distinct quantities; the notebook explains and plots both. The measured response is the empirical benchmark, while the FEM remains an approximation of the physical system.

## Current results

| Validation | Result |
|---|---:|
| Python vs MATLAB: six configurations × six frequencies | Maximum absolute difference: `1.65 × 10⁻⁶ Hz`; maximum relative difference: `5.26 × 10⁻⁷` |
| Assembled stiffness matrix | Relative difference: `1.5 × 10⁻¹⁶` |
| Assembled mass matrix | Maximum difference: `0` |
| FEM vs measurement: mode 2 frequency shift | Mean absolute error: `0.093` percentage points |
| FEM vs measurement: mode 3 frequency shift | Mean absolute error: `0.217` percentage points |
| FEM vs measurement: mode 4 frequency shift | Mean absolute error: `0.256` percentage points |
| MLP vs FEM: interior held-out positions, six-mode mean MAE | `0.0220 Hz` |
| MLP vs FEM: 120 cm edge, six-mode mean MAE | `0.3967 Hz` |

These are results from the executed [validation notebook](notebooks/hannover_validacion_fem.ipynb), not claims of exact physical agreement. The largest shift errors occur at **100 cm and 50 g**. Python–MATLAB agreement checks the translation; FEM–experiment differences measure limitations of the numerical model and the comparison procedure.

The MLP errors come from the executed [training notebook](notebooks/mlp_fem_training.ipynb). They measure agreement with the **Python FEM labels**, not with experiments. The 120 cm position is outside the largest training position (118 cm), and its error is substantially higher than on the interior test positions.

### MLP, FEM, and measurements on identical cases

The MLP notebook also evaluates the trained model on the **34 measured added-mass configurations**, without using measurements for training or model selection. For each configuration, repeated paired experimental frequency shifts are averaged as in the validation notebook. The table reports mean absolute error (MAE) of the **relative frequency shift**, in percentage points (pp), for the same configurations and FEM modes 2–4:

| FEM mode | MLP vs FEM (pp) | FEM vs experiment (pp) | MLP vs experiment (pp) |
|---:|---:|---:|---:|
| 2 | 0.095 | 0.093 | 0.153 |
| 3 | 0.105 | 0.217 | 0.306 |
| 4 | 0.101 | 0.256 | 0.324 |

For every individual case, the **signed** MLP-to-experiment error equals the signed MLP-to-FEM error plus the signed FEM-to-experiment error. Their MAEs do not add because errors can partly cancel. Training on FEM outputs alone provides no information from which to learn a correction for the FEM's physical discrepancy; accidental cancellation remains possible in individual cases. The 120 cm cases, outside the MLP training position range, contribute to the MLP-to-FEM errors above. The notebook includes a chart of these three comparisons and skips this section with an explicit message when the official experimental MAT file is unavailable.

## FEM dataset and MLP

Run `python src/generate_fem_dataset.py` after `setup_hannover.sh` to regenerate `data/generated/fem_dataset.csv` and its metadata JSON. This uses **only** `LiveBeamData.mat` and the Python FEM; measured modal frequencies never become training labels. Following the original MATLAB convention, each centimetre position label is passed as the model node index.

The table contains **1,176 unique mass–position cases**: positions from 10 to 120 cm every 2 cm, crossed with 1 g and 2.5–50 g every 2.5 g. Each row has MLP inputs `position_cm` and `mass_g`, six targets `f1_hz`–`f6_hz`, plus `node` and `split`. There is no zero-mass row repeated at every position; the six reference frequencies are stored once in the metadata.

| Split | Held-out positions (cm) | Cases | Purpose |
|---|---|---:|---|
| `train` | Other positions in the grid | 966 | Fit the MLP |
| `val` | 30, 50, 70, 90, 110 | 105 | Tune and select the MLP |
| `test` | 40, 60, 80, 100 | 84 | Check interpolation at experimental positions |
| `test_edge` | 120 | 21 | Check the beam-end position separately |

All splits use the same mass values; this design checks **generalization to unseen positions**, not extrapolation to new masses. The 120 cm edge test lies just outside the training position range. The 34 measured configurations remain a separate physical benchmark and should not be used to tune the FEM-trained MLP. The JSON records split membership, units, reference frequencies, and SHA-256 hashes of the source MAT file, Python model, and CSV. A maximum upward frequency difference of `2.83 × 10⁻⁶ Hz` appears in near-zero-shift cases; the generator accepts this as numerical solver tolerance without changing the FEM.

`notebooks/mlp_fem_training.ipynb` trains a **2 → 32 → 32 → 6** network with tanh activations, standardized inputs and targets, Adam optimization, and validation-based checkpoint selection. Scaling is fitted on training rows only. The notebook compares the MLP with a linear regression fitted on the same rows, reports per-mode MAE in Hz, and plots held-out predictions and training history. The executed run used PyTorch 2.2.2, seed 42, and selected epoch 1479 of 1500. Its local checkpoint is saved to `results/mlp_fem_model.pt` and ignored by Git; rerunning the notebook recreates it. The six-mode mean MAE is 0.0220 Hz on the 84 interior test cases and 0.3967 Hz on the 21 edge cases. The higher edge error limits use near the beam end; neither number establishes physical accuracy.

## Grid versus Latin hypercube experiment

`src/run_doe_experiment.py` compares a **grid** and an **anchored Latin hypercube design (LHD)** at equal budgets of 200 and 400 FEM training cases. The grid crosses 20 or 40 permitted positions with 10 mass levels. LHD draws the remaining $N-4$ points in two dimensions, maps positions to permitted integer FEM nodes, allows continuous masses from 1 to 50 g, and adds the same four domain corners as the grid. The original FEM dataset and `src/mass_changes.py` are unchanged. Three seeds are used at each budget; each MLP has the same architecture, optimizer, training-only scaling, and validation-based checkpoint rule. Validation positions (30, 50, 70, 90, 110 cm), interior test positions (40, 60, 80, 100 cm), and the 120 cm edge test remain fixed. Measured frequencies are evaluation-only.

The executed [DOE notebook](notebooks/doe_grid_vs_lhd.ipynb) contains the sampled designs, per-run ranges, and the comparison with all 34 measured configurations. The table gives **three-run means** of six-mode MAE against FEM in Hz; the last column is mean shift MAE against experiments over modes 2–4 in percentage points:

| Training design | FEM cases | Interior test (Hz) | 120 cm edge (Hz) | MLP vs experiment (pp) |
|---|---:|---:|---:|---:|
| Grid | 200 | 0.1394 | 0.6657 | 0.3425 |
| Anchored LHD | 200 | 0.0845 | 0.7946 | 0.2732 |
| Grid | 400 | 0.0631 | 0.5996 | 0.2747 |
| Anchored LHD | 400 | 0.0590 | 0.7108 | 0.2977 |
| Original 966-case grid, separate rerun | 966 | 0.0185 | 0.3943 | 0.2621 |

At 200 cases, LHD helps this MLP interpolate at interior positions, but grid is better at 120 cm. At 400 cases, the interior advantage is small and the three-run ranges overlap; grid remains better at 120 cm. The FEM-to-experiment shift MAE averaged over modes 2–4 remains **0.1885 pp** for every design. The 966-case row is a separate retraining run and differs slightly from the earlier saved MLP notebook result. LHD also uses continuous masses whereas grid uses 10 mass levels, so a gain cannot be attributed solely to Latin-hypercube stratification. With three seeds and a fixed set of positions, these are descriptive results for this MLP rather than a general ranking of designs.

The generated design points, test metrics, experimental metrics, run metadata, and source-hash manifest are saved under `data/generated/doe_comparison/`. After the setup below, run `python src/run_doe_experiment.py` and then `notebooks/doe_grid_vs_lhd.ipynb` to regenerate the figures. The experiment requires the official experimental MAT file for final evaluation.

## Repository structure

```text
structural-surrogate-modeling/
├── README.md                         Project scope, results, and instructions
├── Dockerfile                        Python/JupyterLab image definition
├── compose.yaml                      Local notebook service on port 8888
├── .dockerignore                      Excludes local data and environments from the image
├── .gitignore                         Excludes downloaded data and generated files from Git
├── requirements.txt                   Pinned local Python dependencies
├── requirements-ml.txt                Pinned MLP and notebook dependencies
├── requirements-docker.txt            Local dependencies plus JupyterLab
├── setup_hannover.sh                  Downloads and checks the official source files
├── src/
│   ├── mass_changes.py               Python port of the modal FEM
│   ├── generate_fem_dataset.py       Deterministic FEM table generator
│   ├── run_doe_experiment.py         Equal-budget grid/LHD training and evaluation
│   └── build_web_data.py            Rebuilds the static dashboard data
├── tests/
│   ├── test_mass_changes.py           Synthetic numerical smoke test
│   └── test_dataset.py                Dataset and split integrity checks
├── notebooks/
│   ├── hannover_validacion_fem.ipynb  MATLAB parity and experimental comparison
│   ├── mlp_fem_training.ipynb        PyTorch training, FEM tests, and experimental error chain
│   └── doe_grid_vs_lhd.ipynb         DOE comparison, plots, and conclusions
├── results/                          Local MLP checkpoint (generated; ignored)
├── docs/                             Static, offline-capable results dashboard
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── data.js
└── data/
    ├── raw/                           Official ZIP, PDF, and hashes (downloaded; ignored)
    ├── extracted/                     Original MATLAB and measurement files (ignored)
    ├── generated/                     FEM dataset, DOE designs, metrics, and metadata (tracked)
    └── reference/
        └── matlab_parity_reference.npz Small saved MATLAB parity fixture
```

All three notebooks include saved tables and figures so their results can be viewed without rerunning them. The large Hannover archive and extracted measurements are **not committed**; they are obtained from the university by the setup script. The parity fixture is kept in Git for environments without MATLAB and is checked against source-file hashes before use.

## Reproduce the analysis

### Option A: Docker

Requirements: Docker Desktop or another Docker Engine with Compose, an internet connection for the first dataset download, and sufficient free disk space for the image and dataset. From the repository root:

```bash
docker compose up --build
```

The container runs `setup_hannover.sh`, then starts JupyterLab at <http://127.0.0.1:8888>. Open the token-bearing URL printed in the terminal, open `notebooks/hannover_validacion_fem.ipynb`, and choose **Run All Cells**. The service is bound to localhost. Stop it with:

```bash
docker compose down
```

Docker does not include MATLAB. In the container, the parity check uses `data/reference/matlab_parity_reference.npz`, generated with MATLAB R2025b from the original model. The first launch downloads the official ZIP (approximately 103 MB) and PDF and extracts approximately another 104 MB. Subsequent launches reuse `data/raw/` and `data/extracted/` through the Compose mounts.

The current Docker image does **not** include PyTorch and therefore does not run the MLP or DOE training. Their saved notebook outputs and the static dashboard remain viewable. The Docker image has not been end-to-end verified in this repository; use the local Python setup below for the complete MLP experiment.

### Option B: Local Python (full MLP study)

Use **Python 3.10**, Git, and an internet connection for the first dataset download. The commands below target macOS or Linux; on Windows use WSL with Python 3.10, Bash, and `curl`. PyTorch wheels vary by operating system and hardware; consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) if the pinned wheel in `requirements-ml.txt` is unavailable for your platform.

```bash
git clone https://github.com/analozanocastillo/structural-surrogate-modeling.git
cd structural-surrogate-modeling
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ml.txt
bash setup_hannover.sh
python -m pytest -q -p no:cacheprovider tests/
python src/run_doe_experiment.py
python src/build_web_data.py
python -m jupyter lab
```

Open `notebooks/doe_grid_vs_lhd.ipynb` or `notebooks/mlp_fem_training.ipynb` in JupyterLab and select the `.venv` Python kernel. The DOE script verifies the hashes of the original FEM dataset, Python FEM, and beam MAT file before training; it does not overwrite the original dataset. `python src/build_web_data.py` refreshes `docs/data.js` from the experiment CSV files. The dashboard itself opens directly from `docs/index.html` without Jupyter or Python.

For **FEM-only validation**, `requirements.txt` is sufficient. Run `python src/generate_fem_dataset.py` to regenerate the synthetic table, and open `notebooks/hannover_validacion_fem.ipynb`. This notebook uses live MATLAB when available (set `MATLAB_BIN` if needed); otherwise it checks against the saved MATLAB reference. The saved reference is a numerical fixture, not a substitute for validating a new MATLAB installation.

### Verification status

The saved MLP and DOE notebooks executed without cell errors. The two existing Python tests pass. The dashboard data generator reproduced `docs/data.js` byte for byte from the tracked CSVs; its controls and chart construction passed JavaScript checks with a simulated document. A fresh installation of `requirements-ml.txt`, a complete Docker image build, and visual browser rendering of the dashboard have not been verified in this run.

## Interpretation and limits

- The numerical parity check covers selected mass–position configurations and FEM matrices. It does not prove that every possible input is physically accurate.
- The experimental comparison uses identified modal estimates, averaged within each mass configuration. Measurement uncertainty and model assumptions can both contribute to residual differences.
- The MLP approximates FEM outputs and may inherit the FEM's discrepancies relative to measurements. Experimental configurations should be reserved for independent evaluation; they should not be treated as additional FEM-generated labels without a defined data-fusion method.
- The reported MLP test errors apply to the fixed held-out positions in this dataset. Its first comparison with the 34 measured configurations does not establish performance for other beams, masses, positions, or measurement campaigns.
- The DOE result depends on budget, seed, and test region. Its three repetitions and held-out positions do not establish that LHD or grid is universally superior. The 120 cm position remains outside all DOE training designs.

## Source, attribution, and use

The measurement archive, MATLAB code, and accompanying documentation are by **Jasper Ragnitz, Clemens Jonscher, Benedikt Hofmeister, Stefan Warnken, and Raimund Rolfes**: [*Cantilever Beam Dataset for Structural Change Localisation* (2026)](https://doi.org/10.25835/dr0bu9oq), published by Leibniz Universität Hannover under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The Python FEM is a translation of the supplied MATLAB model and retains that attribution. Cite the original dataset when using its data or model. No separate license has yet been selected for this repository's own code.
