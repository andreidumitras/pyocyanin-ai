# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Launch Jupyter for notebooks
uv run jupyter notebook

# Run a notebook non-interactively
uv run jupyter nbconvert --to notebook --execute <notebook>.ipynb
```

## Architecture

This is a two-phase ML pipeline for predicting pyocyanin concentration from electrochemical voltammogram signals.

**Phase 1 — Feature extraction** (`feature_extraction.ipynb`)
Reads raw current/potential measurements from `datasets/Standard calibration in culture media_extended.xlsx`, constructs `Signal` objects, and writes feature vectors to `vectorized/{core,extended,experimental}.csv`.

**Phase 2 — Model training** (`model_training.ipynb`, `core_training.ipynb`, `extended_training.ipynb`, `experimental_training.ipynb`)
Reads the pre-computed CSVs and trains regressors (LinearRegression, Ridge, ElasticNet, SVR, RandomForest, XGBoost) using nested LOOCV: outer `LeaveOneOut` for unbiased evaluation, inner `KFold(n_splits=3)` with `GridSearchCV` or `BayesSearchCV` for hyperparameter tuning. Metrics reported: MAE, RMSE, R².

### Core library

**`Signal` (voltammogram_signal.py)** uses shared class-level state for the potential array and baseline — these must be set once via `Signal.set_common_potential_E(E)` and `Signal.set_common_baseline_I(I_baseline)` before instantiating any signals. On construction each instance: subtracts the baseline, applies Savitzky-Golay smoothing, then detects a `Peak`.

**`Peak` (peak.py)** locates the reduction peak of pyocyanin in the window [-0.55 V, -0.25 V]. It finds the maximum current (Ip/Ep), then walks outward to the nearest local minima to define the true peak boundaries (`start_idx`, `end_idx`).

### Feature tiers

| Tier | Features |
|---|---|
| `core` | Ip, Ep, AUC, FWHM |
| `extended` | + SSA/PCA1, 1st derivative max, 2nd derivative min |
| `experimental` | + left/right slopes, asymmetry, sharpness, compactness, variance, skewness, kurtosis, Chebyshev moment, mean, signal entropy, spectral entropy, FFT power, PCA2, PCA3, wavelet energy |

The `vectorize()` function in `feature_extraction.ipynb` maps a `Signal` to a row in one of these tiers via a `job` parameter (`'core'`, `'extended'`, or `'experimental'`).

### Dataset notes

All raw data is in `datasets/*.xlsx`. The calibration dataset spans concentrations from 0.1 to 100 µM (~40 samples), making LOOCV the appropriate CV strategy. Other Excel files contain P. aeruginosa clinical/ATCC strain data and antibiotic experiments used for validation.
