# AI-Driven Calculated Bandgap Data Curation in Materials Project Based on Known Experimental Data

> **M.Sc. Project Work** — Akash Rajeshbhai Vaghela  
> Department of Materials Discovery and Interfaces
> 
> Interdisciplinary Centre for Advanced Materials Simulation (ICAMS)
> 
> Ruhr-Universität Bochum  
> Supervisors: Prof. Dr.-Ing. Alfred Ludwig · Dr. Victor Dudarev

---

## Abstract

This project presents a physics-informed machine learning framework for predicting experimental electronic band gaps of inorganic compounds using strictly **composition-based feature vectors (CBFVs)**. Through careful curation of 1,010 polymorph-specific compounds from theoretical (Materials Project) and experimental ("Bandgap" Database) sources, a feature engineering pipeline was developed that encodes statistical distributions — averages, differences, and variances — of 90 elemental properties into 276 compound-level descriptors.

Models were evaluated across **unary, binary, ternary, quaternary**, and **general (combined)** subsets using 10-Fold CV, Monte Carlo CV, and Leave-One-Group-Out (LOGO) validation. The **Extra Trees Regressor** was the top-performing architecture, achieving **R² = 0.8001** and **MAE = 0.4825 eV** on the full general dataset. Feature importance analysis confirmed that predictions are driven by solid-state physics principles — primarily variance in oxidation states (43% importance) and Mendeleev number differences — rather than spurious correlations.

---

## Repository Structure

```
├── data/
│   ├── general_mp.csv          # DFT-calculated band gaps — all compounds     (127,556 entries)
│   ├── general_exp.csv         # Experimental band gaps  — all compounds       (2,418 entries)
│   ├── binary_mp.csv           # Calculated — binary compounds                 (15,698 entries)
│   ├── binary_exp.csv          # Experimental — binary compounds                  (538 entries)
│   ├── ternary_mp.csv          # Calculated — ternary compounds                (66,660 entries)
│   ├── ternary_exp.csv         # Experimental — ternary compounds                 (697 entries)
│   ├── quaternary_mp.csv       # Calculated — quaternary compounds             (44,861 entries)
│   ├── quaternary_exp.csv      # Experimental — quaternary compounds            (1,159 entries)
│   ├── unary_mp.csv            # Calculated — unary compounds                     (337 entries)
│   ├── unary_exp.csv           # Experimental — unary compounds                    (24 entries)
│   └── Elements_DB.csv         # Elemental properties database (90 descriptors/element)
│
├── notebooks/
│   ├── general.ipynb           # Full dataset (all arities combined)
│   ├── binary.ipynb            # Binary compounds only
│   ├── ternary.ipynb           # Ternary compounds only
│   ├── quaternary.ipynb        # Quaternary compounds only
│   └── unary.ipynb             # Unary compounds only
│
└── README.md
```

---

## Data Description

### Band Gap Datasets

All CSV files share a consistent 3-column schema:

| Column | Description |
|--------|-------------|
| `formula` | Chemical formula (e.g., `GaAs`, `ZnSe`, `BaCuTeF`) |
| `Crystal System` | Crystal symmetry class (Cubic, Hexagonal, Trigonal, Tetragonal, etc.) |
| `band_gap_calculated (eV)` | DFT-computed band gap from Materials Project *(in `_mp` files)* |
| `band_gap_experimenatal (eV)` | Experimentally measured band gap from "Bandgap" Database *(in `_exp` files)* |

The final merged and curated dataset used for ML contains **1,010 unique polymorph-specific compound entries**, distributed as:

| Subset | Compounds | Examples |
|--------|-----------|---------|
| Unary | 21 | C, Si, Ge, S (various crystal structures) |
| Binary | 430 | GaAs, ZnSe, GaP, SiC, ZnS |
| Ternary | 352 | AlGaN, InGaAs, LaZnAsO |
| Quaternary | 206 | BaCuTeF, CuZnSnS, AlGaInP |
| **General (all)** | **1,010** | All of the above combined |

### Elements_DB.csv — Elemental Descriptor Table

Contains **90 numerical physicochemical properties** per element (91 columns including `symbol`):

- **Basic atomic info:** atomic number, atomic weight, valence electron count
- **Periodic positions:** period, group, Mendeleev numbers (Pettifor, Hao-Tackett-Deem, Chemists' Sequence, etc.)
- **Thermodynamic properties:** melting/boiling temperatures, enthalpy of vaporization/melting, cohesive energy, bulk/shear/Young's moduli, entropy of solid
- **Electronic properties:** multiple electronegativity scales (Pauling, Allen, etc.), 1st–3rd ionization energies, work function, electron affinity, oxidation states
- **Geometric descriptors:** covalent, ionic, metallic, and pseudopotential radii, atomic volume, atomic environment numbers
- **Transport & mechanical:** density, electrical resistivity/conductivity, thermal conductivity, Debye temperature, Brinell hardness, thermal expansion coefficient, Poisson's ratio

This descriptor set follows the Magpie/Matminer framework established by Ward et al. (2016).

---

## Methodology

### 1. Data Curation & Merging

- Theoretical (`_mp`) and experimental (`_exp`) datasets are merged on both `formula` **and** `Crystal System` — critical for correctly matching polymorphic compounds to their experimental band gaps
- Normalized ghost keys (uppercase, whitespace-stripped) prevent formula-mismatch errors (e.g., `"GaAs "` vs `"GaAs"`)
- Duplicate entries (same formula + crystal system with multiple measurements) are resolved by **averaging** the band gap values before merging to create a precise one-to-one mapping

### 2. Feature Engineering

Chemical formulas are parsed into atomic fraction dictionaries via regex:
```
GaAs       →  {Ga: 0.5,  As: 0.5}
BaCuTeF    →  {Ba: 0.25, Cu: 0.25, Te: 0.25, F: 0.25}
C (unary)  →  {C: 1.0}
```

Three types of composition-based descriptors are then computed across all 90 elemental properties, yielding **276 features** per compound:

| Feature Type | Formula | Column Prefix | Physical Meaning |
|---|---|---|---|
| Weighted average | `p̄ₖ = Σ wᵢ pₖ(Eᵢ)` | `avg_*` | Typical elemental environment of the compound |
| Max–min difference | `Δpₖ = max pₖ − min pₖ` | `diff_*` | Elemental contrast (zero for unary, grows with dissimilarity) |
| Weighted variance | `σ²ₖ = Σ wᵢ(pₖ(Eᵢ) − p̄ₖ)²` | `var_*` | Chemical heterogeneity / mixing |

All features are standardized (zero mean, unit variance) before model training.

### 3. Machine Learning

**12 regression algorithms** were benchmarked across all 5 compositional subsets:

| Category | Models |
|---|---|
| Linear & Regularized | Linear Regression, Ridge (α=2.0), Bayesian Ridge |
| Tree-Based Ensembles | Random Forest, Extra Trees, Gradient Boosting, Decision Tree |
| Boosted Frameworks | XGBoost, CatBoost |
| Instance/Kernel | k-Nearest Neighbors (k=5), SVR (RBF kernel) |
| Neural Network | MLP Regressor (2 hidden layers × 100 neurons, ReLU) |

### 4. Multi-Stage Validation Pipeline

To rigorously assess generalization and prevent data leakage:

| Stage | Method | Purpose |
|-------|--------|---------|
| 1 | **80/20 Hold-Out Split** | Static unseen test set reserved before all CV |
| 2 | **10-Fold Cross-Validation** | Fair algorithm benchmarking within training set |
| 3 | **Monte Carlo CV (20 iterations)** | Robustness testing; builds confidence interval on metrics |
| 4 | **Leave-One-Group-Out (LOGO)** | True out-of-domain generalization — entire chemical families withheld |

---

## Results

### Best Model per Arity (10-Fold Cross-Validation)

| Arity | Best Model | Hyperparameters | R² | MAE (eV) | RMSE (eV) |
|-------|-----------|----------------|-----|----------|-----------|
| Unary | LinearRegression | Default | -0.0119 | 1.0774 | 1.4586 |
| Binary | **ExtraTrees** | n_estimators=300 | **0.8271** | 0.6225 | 0.9263 |
| Ternary | **ExtraTrees** | n_estimators=300 | **0.8397** | 0.3150 | 0.4989 |
| Quaternary | **GradientBoosting** | n_estimators=100, max_depth=3 | **0.7847** | 0.3980 | 0.5298 |
| **General** | **ExtraTrees** | n_estimators=300 | **0.8001** | **0.4825** | **0.7827** |

> **Note on Unary:** With only 21 samples, no ML model achieves meaningful predictive accuracy. The negative R² reflects insufficient data for tree-based models to learn decision boundaries — this is a fundamental data limitation, not a feature engineering failure. The unary subset is excluded from advanced validation (MCCV and LOGO).

### Full Benchmarking — All 12 Models on the General Dataset

| Model | Hyperparameters | R² | MAE (eV) | RMSE (eV) |
|-------|----------------|-----|----------|-----------|
| **ExtraTrees** | n_estimators=300 | **0.8001** | **0.4825** | **0.7827** |
| GradientBoosting | n_estimators=200, depth=3 | 0.7913 | 0.5185 | 0.7998 |
| CatBoost | n_estimators=300, depth=5 | 0.7895 | 0.5153 | 0.8033 |
| XGBoost | n_estimators=100, depth=3 | 0.7839 | 0.5251 | 0.8140 |
| RandomForest | n_estimators=300 | 0.7712 | 0.5404 | 0.8374 |
| MLPRegressor | (100,50), max_iter=1500 | 0.7622 | 0.5626 | 0.8348 |
| BayesianRidge | Default | 0.7390 | 0.6146 | 0.8745 |
| Ridge | alpha=2.0 | 0.7117 | 0.6449 | 0.9192 |
| KNeighbors | n_neighbors=5 | 0.6812 | 0.6620 | 0.9667 |
| SVR | RBF, C=1.0, ε=0.1 | 0.6542 | 0.6286 | 1.0067 |
| DecisionTree | max_depth=3 | 0.6084 | 0.7521 | 1.0956 |
| LinearRegression | Default | 0.1663 | 1.1130 | 1.5986 |

### Model Robustness — Monte Carlo Cross-Validation (20 iterations)

| Arity | Model | R² | MAE (eV) | RMSE (eV) |
|-------|-------|-----|----------|-----------|
| Binary | ExtraTrees (n=300) | 0.7339 | 0.6110 | 0.8339 |
| Ternary | ExtraTrees (n=300) | 0.7110 | 0.3506 | 0.5590 |
| Quaternary | GradientBoosting | 0.6806 | 0.3797 | 0.4908 |
| General | ExtraTrees (n=300) | 0.6282 | 0.4846 | 0.7319 |

The MAE remains nearly identical to the 10-Fold CV values (e.g., 0.4846 vs 0.4825 eV for the General model), confirming the pipeline captures genuine periodic trends and is not overfitting to specific data distributions.

### Out-of-Domain Generalization — Leave-One-Group-Out (LOGO)

| Arity | Model | R² | MAE (eV) | RMSE (eV) |
|-------|-------|-----|----------|-----------|
| Binary | ExtraTrees (n=300) | 0.6273 | 0.9096 | 1.3515 |
| Ternary | ExtraTrees (n=300) | 0.5552 | 0.5671 | 0.8179 |
| Quaternary | GradientBoosting | 0.6481 | 0.4907 | 0.6758 |
| General | ExtraTrees (n=300) | 0.6432 | 0.6800 | 1.0227 |

Even when predicting entirely **unseen chemical families**, the framework maintains sub-eV MAE errors across all arity levels, demonstrating genuine out-of-domain transferability driven by universal quantum mechanical heuristics.

### Top 10 Most Important Features (General ExtraTrees Model)

| Rank | Feature | Importance | Physical Interpretation |
|------|---------|-----------|------------------------|
| 1 | `var_oxidation_state_first` | **0.430 (43%)** | Oxidation state variance → charge transfer, ionicity, band gap widening |
| 2 | `diff_mendeleev_h_td_right` | 0.021 | Mendeleev number contrast (Hao-Tackett-Deem scale) |
| 3 | `diff_mendeleev_h_dt_left` | 0.017 | Mendeleev number contrast |
| 4 | `diff_mendeleev_pettifor_regular` | 0.016 | Pettifor scale electronegativity contrast |
| 5 | `diff_mendeleev_chemists_sequence` | 0.015 | Chemists' sequence Mendeleev difference |
| 6 | `diff_mendeleev_dt_left` | 0.014 | Mendeleev number contrast |
| 7 | `avg_entropy_of_solid` | 0.014 | Average thermodynamic lattice entropy → orbital overlap |
| 8 | `diff_mendeleev_td_right` | 0.013 | Mendeleev number contrast |
| 9 | `diff_mendeleev_h_dt_right` | 0.012 | Mendeleev number contrast |
| 10 | `diff_mendeleev_h_td_left` | 0.011 | Mendeleev number contrast |

`var_oxidation_state_first` dominates with 43% of predictive weight: greater oxidation state variance → stronger cation–anion electrostatic interactions (ionicity) → wider valence–conduction band separation → larger band gap. The remaining top features are all expressions of Mendeleev number difference, encoding electronegativity, atomic radius, and valence configuration in a single periodic metric.

---

## Installation & Reproducibility

### Requirements

No local installation needed. All packages are installed automatically inside Google Colab when you run the first cell of any notebook.

### Steps to Reproduce (Google Colab)

1. **Open a notebook in Google Colab**

   Go to [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub tab** → paste the repo URL and select a notebook, e.g. `general.ipynb`

2. **Run the first cell** — it does everything automatically:

   ```python
   # Installs required packages
   !pip install xgboost catboost --quiet

   # Downloads all data files directly from this GitHub repo
   !git clone https://github.com/Akash251041/AI---Driven-calculated-bandgap-datacuration-in-materials-Project-based-onknown-experimental-data.git --quiet

   # Points to the downloaded data
   DATA_DIR = 'AI---Driven-calculated-bandgap-datacuration-in-materials-Project-based-onknown-experimental-data'
   ```

   When successful you will see:
   ```
   ✅ Setup complete! Files available:
   ['general_mp.csv', 'general_exp.csv', 'Elements_DB.csv', ...]
   ```

3. **Run all remaining cells** (`Runtime → Run all`)

   Each notebook executes the full pipeline:
   - Downloads and loads all CSV files directly from GitHub
   - Merges `_mp` + `_exp` datasets on formula + crystal system
   - Parses formulas into composition dictionaries
   - Engineers 276 composition-based features from `Elements_DB.csv`
   - Benchmarks all 12 models with 10-Fold CV
   - Runs MCCV and LOGO validation
   - Outputs parity plots, feature importance charts, and metric tables


---

## Key Findings

- **Tree-based ensembles dominate.** Extra Trees and Gradient Boosting consistently outperform linear models and neural networks, confirming the non-linear relationship between elemental composition and band gap.
- **Compositional complexity aids learning.** Model performance improves from binary → ternary, reflecting that richer chemical diversity provides more statistical signal for periodic trends.
- **Physics-grounded features.** The dominance of oxidation state variance (43%) and Mendeleev number differences validates that the model learns ionicity and electronegativity-driven physics, not formula memorization.
- **True generalization is achievable.** Sub-eV MAE under LOGO validation confirms the framework can assist genuine materials discovery for new, unseen chemical families.

---

## Data Sources & References

- **Materials Project** (calculated band gaps): Jain, A. et al. *APL Materials* 1, 011002 (2013). https://materialsproject.org
- **"Bandgap" Database** (experimental band gaps): Bandgaps of Inorganic Materials database : https://bg.imet-db.ru/

- **Elemental descriptors** : https://elements.imet-db.ru/

Full bibliography available in the project report PDF.

---

## Citation

If you use this code, data, or methodology, please cite:

```bibtex
@misc{vaghela2026bandgap,
  author       = {Vaghela, Akash Rajeshbhai},
  title        = {AI-Driven Calculated Bandgap Data Curation in Materials Project
                  based on Known Experimental Data},
  year         = {2026},
  institution  = {Ruhr-Universit{\"a}t Bochum},
  note         = {M.Sc. Project Work, Department of Materials Discovery and Interfaces and Interdisciplinary Centre for Advanced Materials Simulation (ICAMS)},
  url          = {https://github.com/Akash251041/AI---Driven-calculated-bandgap-datacuration-in-materials-Project-based-onknown-experimental-data}
}
```

---

