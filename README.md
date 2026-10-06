# From Simulation to Reality: Gamma-Ray Analysis with the MAGIC Telescope

## Project Overview

This project investigates gamma-ray event analysis with the **MAGIC (Major Atmospheric Gamma Imaging Cherenkov) Telescope** from two complementary perspectives:

1. **Supervised machine learning on Monte-Carlo simulated events** using the UCI MAGIC Gamma Telescope dataset.
2. **Statistical signal extraction from real MAGIC DL3 observations** using publicly available Crab Nebula data.

The first part addresses the classification of gamma-ray induced air showers and hadronic background events based on reconstructed shower-image parameters.

The second part moves from simulation to real telescope observations. Real DL3 events do **not** provide per-event gamma/hadron ground-truth labels, so the analysis uses an **ON/OFF background-estimation approach** instead of supervised classification.

Together, both parts illustrate an important challenge in scientific machine learning:

> A model can be trained and evaluated with ground truth in simulation, while real observations require statistical inference and careful treatment of background.

The project therefore combines machine learning, domain-specific evaluation, scientific data analysis, and the transition from simulated to real-world data.

---

## Research Questions

1. How accurately can gamma-ray events be separated from hadronic background using simulated MAGIC shower-image parameters?
2. How does model performance change when very low false-positive rates are required?
3. How does the supervised simulation-based classification problem relate to the analysis of real MAGIC telescope observations where event-level ground truth is unavailable?

---

## Data Sources

### 1. Simulated MAGIC events — UCI Machine Learning Repository

The machine-learning part uses the **MAGIC Gamma Telescope** dataset from the UCI Machine Learning Repository:

- 19,020 Monte-Carlo simulated events
- 10 numerical shower-image parameters
- binary target: gamma (`g`) or hadron (`h`)
- no missing values

Source: [UCI MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)  
DOI: [10.24432/C52C8B](https://doi.org/10.24432/C52C8B)

### 2. Real MAGIC observations — DL3 Public Data Release 1

The real-data part uses the **MAGIC Data Level 3 (DL3) Public Data Release 1**, containing approximately 60 hours of Crab Nebula observations acquired between 2013 and 2018.

Source: [MAGIC DL3 Public Data Release 1 on Zenodo](https://zenodo.org/records/11108474)  
DOI: [10.5281/zenodo.11108474](https://doi.org/10.5281/zenodo.11108474)

The DL3 files follow the Gamma Astro Data Formats (GADF) convention and contain reconstructed event quantities and instrument response functions.

---

## Part I — Gamma/Hadron Classification on Simulated Data

### Dataset and Target

The UCI dataset contains ten original numerical features describing the geometry, intensity, concentration, asymmetry, and orientation of recorded shower images.

| Feature | Unit | Description |
|---|---:|---|
| `fLength` | mm | Length of the major axis of the fitted shower ellipse |
| `fWidth` | mm | Length of the minor axis |
| `fSize` | #phot (log) | Logarithmic total light content |
| `fConc` | – | Fraction of light in the two brightest pixels |
| `fConc1` | – | Fraction of light in the brightest pixel |
| `fAsym` | mm | Position of the brightest pixel along the major axis |
| `fM3Long` | mm | Third-moment information along the major axis |
| `fM3Trans` | mm | Third-moment information along the minor axis |
| `fAlpha` | deg | Orientation of the shower ellipse relative to the camera center |
| `fDist` | mm | Distance of the ellipse center from the camera center |

Target encoding:

| Target | Meaning | Encoded value |
|---|---|---:|
| Gamma (`g`) | signal | `1` |
| Hadron (`h`) | background | `0` |

The simulated sample contains more gamma than hadron events. In real telescope data, however, background events are much more abundant. This makes background rejection particularly important.

---

### Why Accuracy Is Not the Main Metric

A generic classification accuracy is not sufficient for this problem.

A **false positive** corresponds to a hadronic background event being accepted as a gamma candidate. In gamma-ray astronomy, such background contamination can be more problematic than rejecting some true gamma events.

With gamma events defined as the positive class:

- **False Positive Rate (FPR):** fraction of hadron events incorrectly classified as gamma
- **True Positive Rate (TPR):** fraction of gamma events correctly identified as gamma
- **TPR** is also referred to as recall or gamma efficiency

The project therefore evaluates the maximum achievable TPR under predefined FPR constraints:

```text
FPR <= 0.01
FPR <= 0.02
FPR <= 0.05
FPR <= 0.10
FPR <= 0.20
```

A custom scorer is used to determine the best TPR that satisfies each FPR limit.

This shifts model optimization toward the scientifically relevant region of the ROC curve instead of optimizing a generic metric such as accuracy.

---

### Exploratory Data Analysis

The exploratory analysis examines:

- class-dependent feature distributions
- correlations and potential feature redundancy
- potential outliers
- nonlinear relationships
- multivariate feature interactions
- physically motivated feature engineering

Several nonlinear relationships are visible in the original features. The orientation feature `fAlpha` shows particularly strong univariate separation between the two classes, while image morphology and light-concentration variables provide complementary information.

<p align="center">
  <img src="results/ml/figures/eda_feature_distributions.png"
       alt="Class-dependent distributions of selected MAGIC features"
       width="900">
</p>

---

### Feature Engineering

Additional features were derived from the original telescope parameters, including:

```text
width_length_ratio
ellipse_area
abs_fAsym
abs_fM3Long
abs_fM3Trans
second_pixel_conc
brightest_pixel_share
alpha_alignment
```

These engineered features capture additional information about:

- shower-image geometry
- absolute asymmetry
- light concentration
- relative dominance of bright pixels
- alignment of the shower image with the camera center

For example, `alpha_alignment` maps the original `fAlpha` orientation into an intuitive 0–1 representation in which larger values correspond to stronger alignment with the camera center.

---

### Modeling Strategy

Model selection is based on **stratified cross-validation** on the training data. The independent test set remains untouched during feature engineering, hyperparameter tuning, and model selection.

The following model families are evaluated:

#### Logistic Regression

Used as the linear baseline and evaluated:

- without PCA
- with PCA
- after Bayesian hyperparameter optimization

#### Support Vector Machine

A nonlinear SVM is investigated because the exploratory analysis indicates nonlinear decision boundaries.

Important hyperparameters such as `C` and `gamma` are optimized.

#### Neural Network

A scikit-learn `MLPClassifier` is used to evaluate whether a neural-network-based approach is beneficial for this dataset.

#### Random Forest

Random Forest performs particularly strongly during model comparison and is subsequently optimized using a broader hyperparameter search.

Bayesian hyperparameter optimization is implemented with **Optuna**.

---

### Hyperparameter Optimization

Instead of optimizing every model for accuracy or generic ROC-AUC, optimization focuses directly on the domain-specific TPR-at-FPR objective.

Separate optimization runs can therefore target different operating points:

```text
TPR @ FPR <= 0.01
TPR @ FPR <= 0.02
TPR @ FPR <= 0.05
TPR @ FPR <= 0.10
TPR @ FPR <= 0.20
```

This is important because the best model configuration can depend on how strongly background contamination must be suppressed.

---

### Model Selection and Independent Test Evaluation

Model selection is performed exclusively on the training data using cross-validation.

The test data are **not used** for:

- feature engineering decisions
- hyperparameter optimization
- model comparison
- model selection

Only after selecting the best-performing model for an operating point is the independent test set evaluated.

<p align="center">
  <img src="results/ml/figures/model_performance_heat_map.png"
       alt="Cross-validation model comparison across FPR operating points"
       width="900">
</p>

The comparison shows that nonlinear models outperform the linear baseline, with the optimized Random Forest providing the strongest overall performance across the investigated operating points.

---

### Model Interpretation

#### Learning Curves

The learning curves demonstrate that the difficulty of the classification problem depends strongly on the FPR constraint.

At strict operating points, training performance can remain substantially above validation performance, indicating a higher-variance regime. At more permissive operating points, training and validation performance converge more closely.

The validation curves also suggest that additional training data could still improve performance, particularly in the strict low-FPR regime.

<p align="center">
  <img src="results/ml/figures/learning_curve_tpr_fpr_005.png"
       alt="Learning curve for FPR 0.05"
       width="48%">
  <img src="results/ml/figures/learning_curve_tpr_fpr_020.png"
       alt="Learning curve for FPR 0.20"
       width="48%">
</p>

#### Feature Importance

Permutation feature importance is used to investigate which information contributes most strongly to the domain-specific TPR-at-FPR metric.

The analysis indicates that important information comes from:

- shower-image orientation
- image morphology
- light concentration
- event size

Because several original and engineered features are correlated, individual feature-importance values should be interpreted with care. Feature groups are more informative than isolated rankings.

<p align="center">
  <img src="results/ml/figures/permutation_feature_importance.png"
       alt="Permutation feature importance across FPR operating points"
       width="850">
</p>

---

### Main Findings from the Simulation Study

The main conclusions from the supervised machine-learning part are:

- nonlinear models substantially outperform the simple linear baseline
- the optimized Random Forest provides the strongest overall performance across the investigated operating points
- classification becomes much more difficult as the allowed false-positive rate decreases
- model quality therefore depends strongly on the selected operating point
- orientation, morphology, light concentration, and event size all contribute useful discriminatory information
- domain-specific evaluation reveals behavior that would be hidden by a single generic metric such as accuracy

The low-FPR regime is especially important because it provides the conceptual bridge to real telescope observations, where background events dominate and individual event labels are unavailable.

---

## Part II — Real MAGIC DL3 Data

### Why the Real-Data Problem Is Different

The UCI data provide event-level ground truth:

```text
event -> gamma or hadron
```

Real MAGIC DL3 data do not.

At DL3 level, the telescope data have already passed through earlier reconstruction and event-selection stages. The files contain reconstructed quantities such as:

- event time
- reconstructed sky position
- reconstructed energy
- effective area
- energy dispersion
- energy-dependent event-selection information

Therefore, the UCI classifier cannot simply be applied to the DL3 events: the original image parameters used by the classifier are not available at this stage, and there is no event-by-event gamma/hadron truth label.

The real-data task is consequently one of **statistical signal extraction**, not supervised classification.

---

### ON/OFF Analysis

The real-data workflow is implemented with **Astropy** and **Gammapy**.

The Crab Nebula position defines the signal region (**ON region**). Background is estimated from reflected **OFF regions** with comparable observational acceptance.

The analysis uses:

- an energy-dependent `RAD_MAX` selection
- reflected background regions
- effective-area information
- energy dispersion
- safe-energy masking
- WStat-based ON/OFF statistics

For an observation with ON counts \(N_\mathrm{on}\), OFF counts \(N_\mathrm{off}\), and exposure ratio \(\alpha\):

```text
estimated background = alpha * N_off
excess               = N_on - alpha * N_off
```

The excess is a statistical estimate of the gamma-ray signal. It does **not** mean that individual events can be tagged as confirmed gamma rays.

---

### Pilot Analysis of a Real MAGIC Observation

The real-data workflow was first validated on a single Crab Nebula observation from the MAGIC DL3 public data release.

The observation has a livetime of approximately 19.6 minutes and was analysed using an energy-dependent signal region together with reflected OFF regions for background estimation.

| Quantity | Result |
|---|---:|
| Observation ID | `5030908` |
| Livetime | 19.59 min |
| ON counts | 426 |
| OFF counts | 446 |
| Alpha | 0.333 |
| Estimated background | 148.67 |
| Excess | 277.33 |
| Significance | 15.14 sigma |
| Energy bins | 17 |
| Fit bins | 16 |

The estimated background is obtained from the OFF regions according to

```text
background = alpha * N_off
```

and the corresponding excess is

```text
excess = N_on - alpha * N_off
```

For this observation:

```text
background = 0.333 * 446 ≈ 148.67
excess     = 426 - 148.67 ≈ 277.33
```

The resulting excess corresponds to a detection significance of approximately **15.1 sigma**, demonstrating a strong Crab Nebula signal within a single observation run of roughly 20 minutes.

#### Energy-Dependent Signal and Background

The following figure shows the number of events in the ON region together with the background estimate derived from the OFF regions and the resulting excess as a function of reconstructed energy.

<p align="center">
  <img src="results/dl3/figures/dl3_on_background_excess.png"
       alt="Energy-binned ON counts, estimated background, and excess for one Crab Nebula observation"
       width="850">
</p>

<p align="center">
  <em>
    Energy-binned ON counts, estimated background, and excess for one Crab Nebula observation.
  </em>
</p>

The low-energy bins contain most of the recorded events and also the largest background contribution. At higher reconstructed energies, substantially fewer events are observed because of the limited exposure of this individual observation run.

The excess represents a **statistical estimate of the gamma-ray signal**. It does not imply that individual events can be identified as confirmed gamma rays.

#### Reflected-Region Background Estimation

The background is estimated using regions at comparable offsets from the telescope pointing direction. In the reflected-regions method, the source region and the background regions are positioned at the same radial distance from the pointing position.

<p align="center">
  <img src="results/dl3/figures/dl3_on_off_geometry_schematic.png"
       alt="Schematic ON/OFF geometry for reflected background estimation"
       width="550">
</p>

<p align="center">
  <em>
    Schematic illustration of the ON region and reflected OFF regions used for background estimation in wobble observations.
  </em>
</p>

The OFF regions provide an estimate of the background under observational conditions that are similar to those of the ON region.

For this analysis, the normalization factor is

```text
alpha = 1 / 3
```

which reflects the relative acceptance of the ON and OFF regions.

The schematic is intended to illustrate the background-estimation concept. The actual analysis uses the observation geometry and energy-dependent selection information provided by the MAGIC DL3 data.

#### Interpretation

This pilot analysis serves two purposes:

1. It validates the technical DL3 processing chain using real MAGIC data.
2. It illustrates the conceptual difference between simulated classification and real observational inference.

In the simulated UCI dataset, every event has a known gamma/hadron label. In the real DL3 observation, no such event-level truth is available. Instead, evidence for gamma-ray emission is obtained statistically by comparing the ON region with an independently estimated background.

This distinction provides the central connection between the two parts of the project: the low-FPR classification problem in simulation and the background-suppression problem in real observations address different stages of the same underlying scientific challenge.

---

## From Simulation to Reality

The two parts of the project address different stages of the same scientific problem.

| Simulation / UCI | Real MAGIC DL3 |
|---|---|
| Monte-Carlo events | Telescope observations |
| event-level ground truth | no event-level truth labels |
| gamma vs. hadron classification | statistical signal extraction |
| image parameters available | reconstructed DL3 quantities available |
| supervised ML | ON/OFF inference |
| TPR/FPR directly measurable | signal/background estimated statistically |

This difference highlights a central challenge of scientific machine learning: **excellent performance on simulated data does not automatically imply equivalent performance on real observations**.

Potential differences between simulation and reality include:

- imperfect detector simulation
- changing observation conditions
- background composition
- calibration effects
- preprocessing and selection effects
- differences between training and deployment distributions

This is a form of **domain shift**.

The low-FPR analysis in Part I is therefore not only a modeling choice. It reflects the same underlying problem that appears in Part II: reliable gamma-ray analysis depends critically on suppressing a much larger background population.

---

## Repository Structure

```text
project/
|
├── data/
│   ├── uci/
│   │   ├── raw/
│   │   │   ├── magic04.data
│   │   │   └── magic04.names
│   │   ├── interim/
│   │   └── processed/
│   │
│   └── dl3/
│       └── raw/
│           └── MAGIC DL3 FITS data
|
├── notebooks/
│   ├── uci_ml/
│   │   ├── Gamma_EDA.ipynb
│   │   └── Gamma_models_thresholds.ipynb
│   │
│   └── dl3_real_data/
│       └── MAGIC_DL3.ipynb
|
├── results/
│   ├── ml/
│   │   ├── figures/
│   │   └── tables/
│   │
│   └── dl3/
│       ├── figures/
│       └── tables/
|
├── src/
│   └── portfolio_projekt/
│       ├── paths.py
│       └── ml/
│           ├── __init__.py
│           ├── baseline_model.py
│           ├── features.py
│           ├── log_reg.py
│           ├── neural_network.py
│           ├── random_forest.py
│           ├── resample.py
│           └── support_vector_machine.py
|
├── pyproject.toml
├── .gitignore
└── README.md
```

The notebooks contain the analytical workflow, visualizations, experiments, and interpretation.

Reusable implementation code is located in the `portfolio_projekt` package under `src/`, while project paths are defined centrally in `paths.py`.

---

## Technologies

### Machine Learning

- Python
- pandas
- NumPy
- Matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- Optuna
- Jupyter

### Scientific / Real-Data Analysis

- Astropy
- Gammapy
- FITS / GADF data

### Project Management

- Git / GitHub
- `uv`
- VS Code

---

## Environment

The project uses a `pyproject.toml` configuration and `uv` for dependency management.

The machine-learning workflow and the real-data workflow are kept conceptually separate because scientific astronomy packages can impose additional dependency constraints.

The DL3 analysis was validated with:

```text
Gammapy 2.1
regions 0.11
```

A separate DL3 virtual environment can therefore be useful when reproducing the astronomy workflow.

---

## Workflow

```text
                         MAGIC Gamma-Ray Analysis
                                   |
                  +----------------+----------------+
                  |                                 |
                  v                                 v
          Monte-Carlo Simulation             Real MAGIC DL3
                (UCI)                          Observations
                  |                                 |
                  v                                 v
        Exploratory Data Analysis             Event Selection
                  |                                 |
                  v                                 v
          Feature Engineering                   ON Region
                  |                                 |
                  v                                 v
           Baseline Model                    Reflected OFF Regions
                  |                                 |
                  v                                 v
       LR / SVM / RF / MLP Models            Background Estimate
                  |                                 |
                  v                                 v
        Optuna Optimization                     Excess
                  |                                 |
                  v                                 v
       TPR @ constrained FPR                  Significance
                  |                                 |
                  +---------------+-----------------+
                                  |
                                  v
                       Simulation-to-Reality
                           Interpretation
```

---

## Current Status and Next Steps

### Completed

- end-to-end ML workflow on the UCI MAGIC dataset
- domain-specific TPR-at-FPR scoring
- comparison of multiple model families
- Bayesian hyperparameter optimization
- independent test evaluation
- learning-curve analysis
- feature-importance analysis
- parsing and inspection of real MAGIC DL3 FITS files
- implementation of a Gammapy ON/OFF analysis
- successful validation on a real Crab Nebula observation

### Planned Extension

The next step is to apply the validated DL3 workflow to the full public observation sample and aggregate the results across multiple runs.

Possible later extensions include:

- stacked analysis of all Crab observations
- energy-dependent excess and significance
- spectral analysis
- comparison between observation conditions
- deeper investigation of simulation-to-reality domain shift

---

## Conclusion

This project combines supervised machine learning on simulated events with statistical analysis of real telescope observations.

The simulation study shows that nonlinear models are well suited to gamma/hadron separation and that performance strongly depends on the allowed false-positive rate. The optimized Random Forest provides the strongest overall performance among the investigated approaches.

The real-data analysis demonstrates why the problem changes fundamentally outside simulation: event-level truth labels disappear, background must be estimated statistically, and conclusions are drawn from populations of events rather than individual classifications.

The central lesson of the project is therefore broader than the choice of classifier:

> Scientific machine learning requires not only predictive performance, but also evaluation metrics, validation strategies, and inference methods that reflect the structure of the real measurement problem.

---

## Data Credits

- R. Bock, **MAGIC Gamma Telescope**, UCI Machine Learning Repository. DOI: [10.24432/C52C8B](https://doi.org/10.24432/C52C8B)
- MAGIC Collaboration, **MAGIC Data Level 3 (DL3) Public Data Release 1 (PDR1)**, Zenodo. DOI: [10.5281/zenodo.11108474](https://doi.org/10.5281/zenodo.11108474)
