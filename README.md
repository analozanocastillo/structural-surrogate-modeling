# Cantilever Beam: FEM and Experimental Validation

A reproducible study of how an added point mass changes the natural frequencies of a cantilever beam. The project translates the finite element model (FEM) supplied with the [Leibniz Universität Hannover cantilever beam dataset](https://data.uni-hannover.de/en/dataset/cantilever-beam-dataset-for-structural-change-localisation) from MATLAB to Python, checks numerical agreement, and compares FEM predictions with measured modal frequencies.

The longer-term objective is to use the Python FEM to generate training examples for a machine learning surrogate that maps **mass and position** to natural frequencies. That dataset generation and surrogate training have **not yet been implemented**. The current repository establishes and evaluates the numerical baseline first.

## Research questions and workflow

1. **MATLAB to Python:** Does the Python implementation reproduce the original FEM matrices and modal frequencies for the same added-mass configurations?
2. **FEM to experiment:** How closely does the numerical model describe the measured frequency shifts when mass and position change?
3. **Next stage:** Which FEM-generated configurations should be used to train a surrogate, and how should its predictions be evaluated against independent measurements?

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

These are results from the executed [validation notebook](notebooks/hannover_validacion_fem.ipynb), not claims of exact physical agreement. The largest shift errors occur at **100 cm and 50 g**. Python–MATLAB agreement checks the translation; FEM–experiment differences measure limitations of the numerical model and the comparison procedure.

## Repository structure

```text
ML_Structural/
├── README.md                         Project scope, results, and instructions
├── Dockerfile                        Python/JupyterLab image definition
├── compose.yaml                      Local notebook service on port 8888
├── .dockerignore                      Excludes local data and environments from the image
├── .gitignore                         Excludes downloaded data and generated files from Git
├── requirements.txt                   Pinned local Python dependencies
├── requirements-docker.txt            Local dependencies plus JupyterLab
├── setup_hannover.sh                  Downloads and checks the official source files
├── src/
│   └── mass_changes.py               Python port of the modal FEM
├── tests/
│   └── test_mass_changes.py           Synthetic numerical smoke test
├── notebooks/
│   └── hannover_validacion_fem.ipynb  MATLAB parity and experimental comparison
└── data/
    ├── raw/                           Official ZIP, PDF, and hashes (downloaded; ignored)
    ├── extracted/                     Original MATLAB and measurement files (ignored)
    └── reference/
        └── matlab_parity_reference.npz Small saved MATLAB parity fixture
```

The notebook includes saved tables and figures so that its results can be viewed directly on GitHub. The large Hannover archive and extracted measurements are **not committed**; they are obtained from the university by the setup script. The parity fixture is kept in Git for environments without MATLAB and is checked against source-file hashes before use.

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

**Docker verification status (2026-11-11):** Docker Desktop 4.99.0 was installed on an Intel Mac. The engine responded, an official `hello-world` container ran successfully, and `docker compose config --quiet` passed. The project image was **not built** and JupyterLab was **not started** because disk space was limited and further installation was stopped at the user's request. Container-specific execution remains to be verified.

### Option B: Local Python

Use Python 3.10 and an internet connection for the first data setup:

```bash
bash setup_hannover.sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q -p no:cacheprovider tests/
.venv/bin/python src/mass_changes.py \
  --mat data/extracted/Code_Veröffentlichung/Data/BeamData/LiveBeamData.mat \
  --node 80 --mass-g 10
```

Open the notebook in VS Code or Jupyter with the `.venv` kernel and run all cells from the repository root. On this Mac, when `/Applications/MATLAB_R2025b.app/bin/matlab` is available, the notebook runs the original MATLAB model directly. Set `MATLAB_BIN` if MATLAB lives elsewhere. Otherwise, it uses the checked saved reference and reports which path was used. The notebook's embedded outputs were successfully generated with both live MATLAB and the saved-reference path; the basic synthetic test passed.

## Interpretation and limits

- The numerical parity check covers selected mass–position configurations and FEM matrices. It does not prove that every possible input is physically accurate.
- The experimental comparison uses identified modal estimates, averaged within each mass configuration. Measurement uncertainty and model assumptions can both contribute to residual differences.
- A future surrogate trained on FEM outputs will approximate the FEM and may inherit its discrepancies relative to measurements. Experimental configurations should be reserved for independent evaluation; they should not be treated as additional FEM-generated labels without a defined data-fusion method.
- The repository currently contains no automated FEM dataset generator and no PyTorch model.

## Source, attribution, and use

The measurement archive, MATLAB code, and accompanying documentation are by **Jasper Ragnitz, Clemens Jonscher, Benedikt Hofmeister, Stefan Warnken, and Raimund Rolfes**: [*Cantilever Beam Dataset for Structural Change Localisation* (2026)](https://doi.org/10.25835/dr0bu9oq), published by Leibniz Universität Hannover under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The Python FEM is a translation of the supplied MATLAB model and retains that attribution. Cite the original dataset when using its data or model. No separate license has yet been selected for this repository's own code.
