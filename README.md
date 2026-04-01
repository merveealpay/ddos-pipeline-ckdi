# DDoS Pipeline – CKDI & BDRI

A single-command analysis pipeline that quantifies **hardware performance counter (HPC) drift** caused by DDoS attacks on Electric Vehicle (EV) charging infrastructure. It introduces two novel composite indices:

| Index | Full Name | What It Measures |
|-------|-----------|------------------|
| **CKDI** | Composite Knowledge Drift Index | Magnitude of drift between normal and attack samples in PCA-projected feature space |
| **BDRI** | Balanced Drift Resilience Index | Weighted service-level impact score (latency, availability, MTTR) with SLSQP-optimised weights |

The pipeline processes Linux `perf` top recordings from Charging Station (CS) and Grid Station (GS) nodes across multiple attack scenarios, computes per-sample CKDI and BDRI scores, and produces publication-ready figures and CSV tables for LaTeX/Overleaf.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [CKDI – Composite Knowledge Drift Index](#ckdi--composite-knowledge-drift-index)
  - [BDRI – Balanced Drift Resilience Index](#bdri--balanced-drift-resilience-index)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
  - [CSV Files](#csv-files)
  - [Figures](#figures)
- [Configuration](#configuration)
- [License](#license)

---

## Project Structure

```
ddos-pipeline-ckdi/
├── LICENSE
├── README.md
├── requirements.txt
├── pipeline_ckdi_bdri.py        # Main entry point – end-to-end pipeline
├── run_pipeline.py              # Alternative pipeline (synthetic/CSV-based)
├── Processed_Data/              # Processed perf data (final_dataset.json files)
│   ├── Correct_ID/
│   ├── Wrong_CS_TS/
│   ├── Wrong_EV_TS/
│   └── Wrong_ID/
├── Raw_Data/                    # Raw perf recordings & scenario configs
├── results_outputs/             # Generated CSV results (git-ignored)
├── figures/                     # Generated PDF figures (git-ignored)
└── src/                         # Modular library (used by run_pipeline.py)
    ├── __init__.py
    ├── ckdi.py
    ├── bdri.py
    ├── data_loader.py
    └── visualizer.py
```

---

## Dataset

The data originates from a DDoS attack testbed targeting ISO 15118 EV charging communication. Linux `perf top` recordings capture hardware performance counters (branch events, CPU cycles, instructions) on CS and GS nodes.

### Processed Data Layout

```
Processed_Data/
└── {Scenario}/                           # Correct_ID, Wrong_CS_TS, Wrong_EV_TS, Wrong_ID
    └── Random_CS_{On|Off}/
        └── Gaussian_{On|Off}/
            ├── TOP/
            │   ├── CS/final_dataset.json  # Charging Station perf-top features
            │   └── GS/final_dataset.json  # Grid Station perf-top features
            └── STAT/                      # Perf-stat summaries & sampling info
```

Each `final_dataset.json` is structured as:

```
{event_type}/{feature_set}/{feature_name}/{attack|normal}/data_point/{function}/{sample_id} → [values...]
```

- **Event types:** `branch`, `cycles`, `instructions`
- **Feature sets:** `common`, `exclusive`, `all`
- **Classes:** `attack` (DDoS) vs `normal` (benign)

### Attack Scenarios

| Scenario | Description |
|----------|-------------|
| **Correct_ID** | Valid identifiers – baseline + attack under normal auth |
| **Wrong_ID** | Incorrect vehicle/station identifiers |
| **Wrong_CS_TS** | Manipulated Charging Station timestamps |
| **Wrong_EV_TS** | Manipulated Electric Vehicle timestamps |

---

## Methodology

### CKDI – Composite Knowledge Drift Index

CKDI quantifies how much a sample has drifted from the normal baseline in the feature space.

1. **Feature extraction** – For each sample (scenario × role × entity), the mean and standard deviation of HPC readings are aggregated across event types and feature sets, producing a wide feature vector.

2. **Baseline-referenced normalisation** – Features are z-score normalised per (role, perf\_mode) group using the normal-class statistics as reference.

3. **PCA projection** – A single principal component is extracted; the absolute PC1 score represents the drift magnitude.

4. **Min–max scaling** – The absolute PC1 values are scaled to $[0, 1]$:

$$\text{CKDI} = \frac{|\text{PC1}| - \min|\text{PC1}|}{\max|\text{PC1}| - \min|\text{PC1}|}$$

Higher CKDI → greater deviation from normal operation → higher attack severity.

### BDRI – Balanced Drift Resilience Index

BDRI translates CKDI into a service-level resilience score via three analytically modelled sub-indices:

| Sub-index | Formula | Interpretation |
|-----------|---------|----------------|
| **SDI** (Latency drift) | $\text{SDI} = \frac{L - L_{\text{base}}}{L_{\text{base}}}$, where $L = L_{\text{base}} (1 + \delta \cdot \text{CKDI})$ | Relative latency increase |
| **ALR** (Availability loss) | $\text{ALR} = 1 - e^{-\alpha \cdot \text{CKDI}}$ | Probability of service unavailability |
| **RIF** (Recovery cost) | $\text{RIF} = \frac{\text{MTTR} - \text{MTTR}_{\text{base}}}{\text{MTTR}_{\text{base}}}$, where $\text{MTTR} = \text{MTTR}_{\text{base}} + \beta \cdot \text{CKDI}$ | Relative repair-time increase |

The final BDRI is a convex combination:

$$\text{BDRI} = 1 - (w_1 \cdot \text{SDI} + w_2 \cdot \text{ALR} + w_3 \cdot \text{RIF})$$

**Weight optimisation** – Weights $w_1, w_2, w_3$ are found via **SLSQP** by minimising the objective:

$$\min_{w} \; \text{Var}(\text{BDRI}_{\text{normal}}) - \lambda \cdot \text{Corr}(\text{CKDI}, \text{BDRI})$$

subject to $\sum_i w_i = 1, \; w_i \geq 0$. This simultaneously minimises variance within normal samples and maximises correlation with CKDI.

### Configurable Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `L_BASE` | 1.0 | Baseline latency |
| `MTTR_BASE` | 1.0 | Baseline mean time to repair |
| `DELTA` | 1.0 | Latency slope coefficient |
| `ALPHA` | 2.0 | Availability decay rate |
| `BETA` | 1.0 | MTTR slope coefficient |
| `LAMBDA_OPT` | 1.0 | Objective correlation weight |

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/mervealpay/ddos-pipeline-ckdi.git
cd ddos-pipeline-ckdi

# Create virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dependencies:
- `numpy >= 1.24`
- `pandas >= 2.0`
- `scipy >= 1.11`
- `scikit-learn >= 1.3`
- `matplotlib >= 3.8`

---

## Usage

### Main pipeline (real data)

Make sure `Processed_Data/` is in the project root, then run:

```bash
python pipeline_ckdi_bdri.py
```

The pipeline will:
1. Recursively discover all `final_dataset.json` files
2. Parse HPC features into a wide sample-level table
3. Compute baseline-normalised drift features
4. Calculate CKDI via PCA
5. Map CKDI to service-level sub-indices (SDI, ALR, RIF)
6. Optimise BDRI weights via SLSQP
7. Export CSV results to `results_outputs/`
8. Generate PDF figures to `figures/`

### Alternative pipeline (synthetic data)

For testing without the real dataset:

```bash
python run_pipeline.py
```

This uses a built-in synthetic dataset and the modular `src/` library.

---

## Outputs

### CSV Files

| File | Description |
|------|-------------|
| `results_outputs/sample_level_ckdi_bdri.csv` | Full sample-level table with features, CKDI, modelled service metrics (Latency, Availability, MTTR), and BDRI scores (equal-weight & optimised) |
| `results_outputs/bdri_comparison.csv` | Equal-weight vs optimised BDRI: variance within normal class and CKDI–BDRI correlation |
| `results_outputs/optimized_weights.csv` | SLSQP-optimal weights ($w_1^{\text{SDI}}, w_2^{\text{ALR}}, w_3^{\text{RIF}}$) |

### Figures

All figures are saved as publication-ready PDFs for direct use in Overleaf/LaTeX:

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `ckdi_distribution.pdf` | CKDI histogram – normal vs attack class separation |
| Fig 2 | `ckdi_vs_availability.pdf` | Scatter: CKDI vs modelled service availability |
| Fig 3 | `ckdi_vs_latency.pdf` | Scatter: CKDI vs modelled latency |
| Fig 4 | `bdri_escalation.pdf` | BDRI (optimised) as a function of sorted CKDI |
| Fig 5 | `bdri_equal_vs_opt.pdf` | Comparison of equal-weight vs optimised BDRI curves |

---

## Configuration

### Modifying service mapping parameters

Edit the constants at the top of [pipeline_ckdi_bdri.py](pipeline_ckdi_bdri.py):

```python
L_BASE    = 1.0   # Baseline latency
MTTR_BASE = 1.0   # Baseline mean time to repair
DELTA     = 1.0   # Latency slope
ALPHA     = 2.0   # Availability decay rate
BETA      = 1.0   # MTTR slope
LAMBDA_OPT = 1.0  # Optimisation objective correlation weight
```

### Using the modular library

The `src/` modules can be imported independently:

```python
from src.data_loader import load_dataset, split_by_label, feature_columns
from src.ckdi import compute_ckdi, compute_ckdi_detailed
from src.bdri import compute_bdri

df = load_dataset("Processed_Data")
baseline, attacks = split_by_label(df)
feat_cols = feature_columns(df)

ckdi_df = compute_ckdi_detailed(baseline, attacks, feat_cols, alpha=0.5)
bdri_df = compute_bdri(baseline, attacks, feat_cols)
```

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Merve Alpay