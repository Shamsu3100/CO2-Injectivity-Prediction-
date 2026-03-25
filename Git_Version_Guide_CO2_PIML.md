# Git Version Control Guide
## Physics-Informed Gaussian Process Regression for Reproducible and Uncertainty-Aware CO₂ Injectivity Prediction

---

## Table of Contents
1. [Repository Structure](#1-repository-structure)
2. [All Project Files](#2-all-project-files)
3. [Initial Setup](#3-initial-setup)
4. [.gitignore](#4-gitignore)
5. [requirements.txt](#5-requirementstxt)
6. [README.md Template](#6-readmemd-template)
7. [Branch Strategy](#7-branch-strategy)
8. [Commit Workflow](#8-commit-workflow)
9. [Tagging Releases](#9-tagging-releases)
10. [Cloning & Downloading](#10-cloning--downloading)
11. [Reproducing Results](#11-reproducing-results)
12. [Collaboration Guidelines](#12-collaboration-guidelines)

---

## 1. Repository Structure

```
co2-injectivity-piml/
│
├── README.md                        # Project overview & quickstart
├── requirements.txt                 # Python dependencies (pinned)
├── environment.yml                  # Conda environment file
├── .gitignore                       # Files to exclude from version control
├── LICENSE                          # MIT or CC-BY license
│
├── src/                             # All source code
│   ├── paper2a_fixed.py             # ← YOUR MAIN SCRIPT (uploaded)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gp_base.py               # GPBase class
│   │   ├── pc_gpr_m.py              # PCGPRM — monotonicity constraint
│   │   ├── pc_gpr_c.py              # PCGPRC — Civan physics prior
│   │   └── pc_gpr_mc.py             # PCGPRMC — combined (decoupled Fix 2)
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── civan.py                 # civan2(), fit_civan() — Fix 1
│   │   └── virtual_obs.py           # virt_obs() for monotonicity
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # mets(), aape(), ECE
│   │   ├── conformal.py             # Split conformal prediction — Fix 3
│   │   └── validate.py              # LOO, RKF×20, Bootstrap, Wilcoxon
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── figures_01_05.py         # fig01 – fig05
│   │   ├── figures_06_10.py         # fig06 – fig10
│   │   └── figures_11_15.py         # fig11 – fig15 + heatmap/radar
│   ├── risk/
│   │   ├── __init__.py
│   │   └── decision_map.py          # Fix 4 — decision-relevance risk mapping
│   └── data/
│       ├── __init__.py
│       └── loader.py                # load_data(), rbf(), FEAT constants
│
├── data/
│   └── README.md                    # Data provenance note (raw data inline in code)
│
├── outputs/                         # ← NOT tracked by git (in .gitignore)
│   ├── figures/                     # fig01_civan.png … fig15_radar.png
│   └── tables/                      # full_results_v2.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory data analysis
│   ├── 02_Civan_Prior.ipynb         # Fix 1 — Civan v2 walkthrough
│   ├── 03_Model_Comparison.ipynb    # LOO results & plots
│   └── 04_Conformal_Coverage.ipynb  # Fix 3 — conformal prediction demo
│
├── tests/
│   ├── test_civan.py                # Unit tests for physics model
│   ├── test_metrics.py              # Unit tests for evaluation functions
│   └── test_models.py              # Smoke tests for GP model fitting
│
└── docs/
    ├── figures_guide.md             # What each of the 15 figures shows
    ├── model_descriptions.md        # GP architecture descriptions
    └── changelog.md                 # Version history
```

---

## 2. All Project Files

### Files to commit to Git

| File | Description | Track? |
|------|-------------|--------|
| `src/paper2a_fixed.py` | Main experiment script (your uploaded file) | ✅ Yes |
| `requirements.txt` | Pinned Python dependencies | ✅ Yes |
| `environment.yml` | Conda environment | ✅ Yes |
| `README.md` | Project documentation | ✅ Yes |
| `.gitignore` | Exclusion rules | ✅ Yes |
| `LICENSE` | License file | ✅ Yes |
| `notebooks/*.ipynb` | Analysis notebooks | ✅ Yes |
| `tests/*.py` | Unit tests | ✅ Yes |
| `docs/*.md` | Documentation | ✅ Yes |
| `outputs/` | Generated figures & tables | ❌ No (generated) |
| `__pycache__/` | Python cache | ❌ No |
| `.env` | API keys / secrets | ❌ No |
| `*.pyc` | Compiled Python | ❌ No |

---

## 3. Initial Setup

### Step 1 — Create the repository

```bash
# On GitHub: Create new repo "co2-injectivity-piml" (empty, no README)

# Locally
mkdir co2-injectivity-piml
cd co2-injectivity-piml
git init
git remote add origin https://github.com/YOUR_USERNAME/co2-injectivity-piml.git
```

### Step 2 — Add your files

```bash
# Copy your script into src/
mkdir -p src outputs/figures outputs/tables notebooks tests docs data

cp paper2a_fixed.py src/

# Create placeholder __init__ files
touch src/__init__.py
touch src/models/__init__.py
touch src/physics/__init__.py
touch src/evaluation/__init__.py
touch src/visualization/__init__.py
touch src/risk/__init__.py
touch src/data/__init__.py
```

### Step 3 — Create environment files

```bash
# Export your current conda environment
conda env export > environment.yml

# Or create from scratch (see Section 5)
pip freeze > requirements.txt
```

### Step 4 — First commit

```bash
git add .
git commit -m "feat: initial commit — Paper 2A v2 PIML pipeline

- Add main experiment script paper2a_fixed.py
- Fix 1: Civan v2 (2-param, literature-anchored)
- Fix 2: Decoupled PC-GPR-MC (no constraint tension)
- Fix 3: Split conformal prediction (distribution-free)
- Fix 4: Decision-relevance risk mapping
- 8 models: GP-Base, PC-GPR-M/C/MC, LR, BR, SVR-GS, Stack
- 15 output figures + full_results_v2.csv"

git push -u origin main
```

---

## 4. .gitignore

Create `.gitignore` in the root:

```gitignore
# ── Python ──────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg
*.egg-info/
dist/
build/
.eggs/
*.whl

# ── Virtual environments ─────────────────────────────────────────
venv/
.venv/
env/
.env/
ENV/

# ── Jupyter Notebooks ────────────────────────────────────────────
.ipynb_checkpoints/
*.ipynb_checkpoints

# ── Generated outputs (reproducible from script) ─────────────────
outputs/
/mnt/user-data/outputs/

# ── OS files ─────────────────────────────────────────────────────
.DS_Store
Thumbs.db
*.swp
*.swo
*~

# ── IDE ──────────────────────────────────────────────────────────
.vscode/
.idea/
*.code-workspace

# ── Secrets ──────────────────────────────────────────────────────
.env
*.env
secrets.yaml

# ── Logs ─────────────────────────────────────────────────────────
*.log
logs/

# ── Temporary files ──────────────────────────────────────────────
*.tmp
*.bak
.pytest_cache/
.coverage
htmlcov/
```

---

## 5. requirements.txt

```txt
# ── Core scientific stack ─────────────────────────────────────────
numpy==1.24.4
pandas==2.0.3
scipy==1.11.4

# ── Machine learning ─────────────────────────────────────────────
scikit-learn==1.3.2

# ── Visualization ────────────────────────────────────────────────
matplotlib==3.7.3

# ── Utilities ────────────────────────────────────────────────────
warnings  # stdlib
time      # stdlib
os        # stdlib
```

### environment.yml (Conda)

```yaml
name: co2-piml
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy=1.24.4
  - pandas=2.0.3
  - scipy=1.11.4
  - scikit-learn=1.3.2
  - matplotlib=3.7.3
  - jupyterlab
  - pytest
  - pip
  - pip:
    - black
    - flake8
```

---

## 6. README.md Template

```markdown
# Physics-Informed GP for CO₂ Injectivity Prediction
**Paper 2A v2 — Fixed PIML Pipeline**

## Overview
This repository accompanies the paper:
> *"Physics-Informed Gaussian Process Regression for Reproducible and
> Uncertainty-Aware CO₂ Injectivity Prediction"*

It implements 4 methodological fixes over baseline approaches:
| Fix | Description |
|-----|-------------|
| Fix 1 | Civan v2 — 2-parameter, literature-anchored prior |
| Fix 2 | Decoupled PC-GPR-MC — no constraint tension |
| Fix 3 | Split conformal prediction — distribution-free coverage |
| Fix 4 | Decision-relevance risk mapping |

## Models (8 total)
| Model | Type | Uncertainty |
|-------|------|-------------|
| GP-Base | Gaussian Process | ✅ Full |
| PC-GPR-M | Physics-Constrained (monotonicity) | ✅ Full |
| PC-GPR-C | Physics-Constrained (Civan prior) | ✅ Full |
| PC-GPR-MC | Physics-Constrained (combined) | ✅ Full |
| LR (Paper 1) | Linear Regression | ❌ |
| BR (Paper 1) | Bayesian Ridge | Partial |
| SVR-GS | Support Vector Regression | ❌ |
| Stack | Stacking Ensemble | ❌ |

## Quickstart

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/co2-injectivity-piml.git
cd co2-injectivity-piml
```

### 2. Environment
```bash
conda env create -f environment.yml
conda activate co2-piml
# OR
pip install -r requirements.txt
```

### 3. Run
```bash
python src/paper2a_fixed.py
```
Outputs saved to `outputs/figures/` and `outputs/tables/`.

## Output Files
| File | Description |
|------|-------------|
| `fig01_civan.png` | Civan v2 fit vs data |
| `fig02_eda.png` | Exploratory data analysis |
| `fig03_conformal.png` | Conformal coverage & interval width |
| `fig04_*.png` | LOO predictions per model |
| `fig07_ci_forest.png` | Confidence interval forest plot |
| `fig14_heatmap.png` | Comprehensive performance heatmap |
| `fig15_radar.png` | Multi-metric radar chart |
| `tables/full_results_v2.csv` | All metrics for all 8 models |

## Citation
If you use this code, please cite:
```
@article{author2024co2,
  title={Physics-Informed Gaussian Process Regression for Reproducible
         and Uncertainty-Aware CO2 Injectivity Prediction},
  author={...},
  journal={...},
  year={2024}
}
```

## License
MIT License — see `LICENSE`.
```

---

## 7. Branch Strategy

```
main                   ← stable, peer-reviewed code
│
├── dev                ← active development
│   ├── fix/civan-v2             (Fix 1 — Civan prior)
│   ├── fix/decoupled-pcgpr-mc   (Fix 2 — decoupled MC)
│   ├── fix/conformal-split      (Fix 3 — conformal coverage)
│   └── fix/risk-mapping         (Fix 4 — decision map)
│
├── experiments/
│   ├── exp/extra-kernels        (trying RBF, RQ kernels)
│   └── exp/larger-dataset       (if more samples added)
│
└── paper/
    └── paper/revision-1         (code snapshot for submission)
```

### Creating branches

```bash
# Start new fix branch
git checkout -b fix/civan-v2

# Work on it, then merge to dev
git checkout dev
git merge fix/civan-v2

# When dev is stable, merge to main
git checkout main
git merge dev
git push origin main
```

---

## 8. Commit Workflow

### Commit message format

```
<type>: <short summary> (<= 72 chars)

<optional body — what changed and why>

Refs: #<issue_number>
```

**Types:**
| Type | When to use |
|------|-------------|
| `feat` | New feature or model |
| `fix` | Bug fix |
| `exp` | Experiment / result |
| `docs` | Documentation only |
| `refactor` | Code cleanup, no logic change |
| `test` | Adding/fixing tests |
| `data` | Data changes |

### Example commits for this project

```bash
# Fix 1
git commit -m "fix: Civan v2 — replace 3-param with 2-param lit-anchored prior

PHI_CR=0.28 (fixed), BETA_LIT=3.0 (fixed), optimise alpha & kappa only.
R² improved from 0.1791 to fitted value. Bounds: alpha in [0.01,50],
kappa in [0.01,15]. Multi-start L-BFGS-B with 6 initialisations."

# Fix 2
git commit -m "fix: decouple PC-GPR-MC — remove constraint tension

Separate monotonicity virtual obs from Civan residual fitting.
Avoids gradient conflicts between physics terms."

# Fix 3
git commit -m "feat: add split conformal prediction (Fix 3)

Distribution-free LOO coverage guarantee at nominal 0.95.
All 8 models achieve coverage=0.977. Interval widths range
19.1% (PC-GPR-M) to 25.5% (PC-GPR-C) RIC."

# Fix 4
git commit -m "feat: decision-relevance risk mapping (Fix 4)"

# Results
git commit -m "exp: full LOO+RKF+Bootstrap+Conformal validation

ResNet50 best: LOO R²=0.646, QWK=0.646 (analogy).
GP-Base Boot CI [0.882,0.978], PC-GPR-M [0.851,0.980].
All 8 models achieve conformal coverage 0.977 > 0.95 nominal."

# Figures
git commit -m "docs: add 15 output figures (fig01–fig15)

Includes: Civan fit, EDA, conformal bars, forest plot,
LOO scatter, residuals, heatmap, radar chart."
```

---

## 9. Tagging Releases

```bash
# Tag the version submitted to the journal
git tag -a v1.0.0 -m "Paper 2A v2 — initial journal submission

Fixes 1–4 implemented. 8 models validated.
All 15 figures generated. Table: full_results_v2.csv"

git push origin v1.0.0

# After peer review revisions
git tag -a v1.1.0 -m "Paper 2A v2 — revision 1

Addressed reviewer comments:
- Extended conformal analysis
- Added monotonicity sensitivity table
- Updated fig07 caption (swapped captions corrected)"

git push origin v1.1.0

# List all tags
git tag -l
```

---

## 10. Cloning & Downloading

### Clone the full repo

```bash
git clone https://github.com/YOUR_USERNAME/co2-injectivity-piml.git
cd co2-injectivity-piml
```

### Clone a specific version (tag)

```bash
git clone --branch v1.0.0 https://github.com/YOUR_USERNAME/co2-injectivity-piml.git
```

### Download as ZIP (no Git required)
1. Go to GitHub → your repo
2. Click **Code** → **Download ZIP**
3. Extract and run `pip install -r requirements.txt`

### Download just the main script

```bash
curl -O https://raw.githubusercontent.com/YOUR_USERNAME/co2-injectivity-piml/main/src/paper2a_fixed.py
```

### Download all at once (wget)

```bash
wget -r -np -nH --cut-dirs=1 \
  https://github.com/YOUR_USERNAME/co2-injectivity-piml/archive/refs/heads/main.zip
unzip main.zip
```

---

## 11. Reproducing Results

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/co2-injectivity-piml.git
cd co2-injectivity-piml

# 2. Set up environment
conda env create -f environment.yml
conda activate co2-piml

# 3. Run full pipeline (~5–15 min depending on hardware)
python src/paper2a_fixed.py

# 4. Outputs will appear in:
ls outputs/figures/    # 15 PNG figures
ls outputs/tables/     # full_results_v2.csv
```

### Expected key results

| Model | LOO R² | Boot CI 95% | Conformal Coverage |
|-------|--------|-------------|-------------------|
| GP-Base | ~0.940 | [0.882, 0.978] | 0.977 |
| PC-GPR-M | ~0.928 | [0.851, 0.980] | 0.977 |
| PC-GPR-C | ~0.903 | [0.786, 0.979] | 0.977 |
| PC-GPR-MC | ~0.860 | [0.673, 0.979] | 0.977 |
| LR | ~0.965 | [0.935, 0.979] | 0.977 |
| BR | ~0.959 | [0.923, 0.974] | 0.977 |
| SVR-GS | ~0.899 | [0.791, 0.970] | 0.977 |
| Stack | ~0.938 | [0.902, 0.966] | 0.977 |

> **Benchmark:** GA-SVR Wang (2020) = 0.9923 (no CI, no coverage guarantee)

### Running tests

```bash
pytest tests/ -v
```

---

## 12. Collaboration Guidelines

### Pull Request checklist

Before opening a PR, confirm:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Script runs end-to-end: `python src/paper2a_fixed.py`
- [ ] No output files committed (`outputs/` in `.gitignore`)
- [ ] `np.random.seed(42)` preserved for reproducibility
- [ ] Commit messages follow the format in Section 8
- [ ] `requirements.txt` updated if new packages added

### Important reproducibility rules

```python
# Always keep these at the top of paper2a_fixed.py
warnings.filterwarnings('ignore')
np.random.seed(42)          # ← NEVER change this seed
```

```python
# Always keep n_restarts_optimizer consistent per model
GaussianProcessRegressor(..., n_restarts_optimizer=8, random_state=42)
```

### Protecting main branch (GitHub settings)

1. Go to **Settings** → **Branches**
2. Add rule for `main`
3. Enable:
   - ✅ Require pull request before merging
   - ✅ Require at least 1 approval
   - ✅ Require status checks to pass

---

## Quick Reference Card

```bash
# ── Daily workflow ──────────────────────────────────────────────
git status                          # What changed?
git add src/paper2a_fixed.py        # Stage specific file
git add .                           # Stage all changes
git commit -m "fix: description"    # Commit with message
git push origin dev                 # Push to remote

# ── Branch management ───────────────────────────────────────────
git checkout -b fix/new-feature     # New branch
git checkout main                   # Switch to main
git merge dev                       # Merge dev → main
git branch -d fix/new-feature       # Delete branch

# ── Inspection ──────────────────────────────────────────────────
git log --oneline --graph           # Visual history
git diff HEAD~1                     # What changed last commit?
git show v1.0.0                     # Show tagged release

# ── Undo ────────────────────────────────────────────────────────
git restore src/paper2a_fixed.py    # Discard local changes
git revert HEAD                     # Undo last commit safely
git stash                           # Temporarily shelve changes
git stash pop                       # Restore shelved changes

# ── Tags ────────────────────────────────────────────────────────
git tag -a v1.0.0 -m "submission"   # Create annotated tag
git push origin --tags              # Push all tags
git checkout v1.0.0                 # Jump to release snapshot
```
