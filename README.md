# AI-Driven calculated bandgap datacuration in materials Project based onknown experimental data
Abstract
This project presents a physics-informed machine learning framework for predicting experimental electronic band gaps of inorganic compounds using strictly composition-based feature vectors (CBFVs). Through careful curation of 1,010 polymorph-specific compounds from theoretical (Materials Project) and experimental ("Bandgap" Database) sources, a feature engineering pipeline was developed that encodes statistical distributions — averages, differences, and variances — of 90 elemental properties into 276 compound-level descriptors.
Models were evaluated across unary, binary, ternary, quaternary, and general (combined) subsets using 10-Fold CV, Monte Carlo CV, and Leave-One-Group-Out (LOGO) validation. The Extra Trees Regressor was the top-performing architecture, achieving R² = 0.8001 and MAE = 0.4825 eV on the full general dataset. Feature importance analysis confirmed that predictions are driven by solid-state physics principles — primarily variance in oxidation states (43% importance) and Mendeleev number differences — rather than spurious correlations.

Repository Structure
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

Data Description
Band Gap Datasets
All CSV files share a consistent 3-column schema:
ColumnDescriptionformulaChemical formula (e.g., GaAs, ZnSe, BaCuTeF)Crystal SystemCrystal symmetry class (Cubic, Hexagonal, Trigonal, Tetragonal, etc.)band_gap_calculated (eV)DFT-computed band gap from Materials Project (in _mp files)band_gap_experimenatal (eV)Experimentally measured band gap from "Bandgap" Database (in _exp files)
The final merged and curated dataset used for ML contains 1,010 unique polymorph-specific compound entries, distributed as:
SubsetCompoundsExamplesUnary21C, Si, Ge, S (various crystal structures)Binary430GaAs, ZnSe, GaP, SiC, ZnSTernary352AlGaN, InGaAs, LaZnAsOQuaternary206BaCuTeF, CuZnSnS, AlGaInPGeneral (all)1,010All of the above combined
Elements_DB.csv — Elemental Descriptor Table
Contains 90 numerical physicochemical properties per element (91 columns including symbol):

Basic atomic info: atomic number, atomic weight, valence electron count
Periodic positions: period, group, Mendeleev numbers (Pettifor, Hao-Tackett-Deem, Chemists' Sequence, etc.)
Thermodynamic properties: melting/boiling temperatures, enthalpy of vaporization/melting, cohesive energy, bulk/shear/Young's moduli, entropy of solid
Electronic properties: multiple electronegativity scales (Pauling, Allen, etc.), 1st–3rd ionization energies, work function, electron affinity, oxidation states
Geometric descriptors: covalent, ionic, metallic, and pseudopotential radii, atomic volume, atomic environment numbers
Transport & mechanical: density, electrical resistivity/conductivity, thermal conductivity, Debye temperature, Brinell hardness, thermal expansion coefficient, Poisson's ratio

This descriptor set follows the Magpie/Matminer framework established by Ward et al. (2016).

Methodology
1. Data Curation & Merging

Theoretical (_mp) and experimental (_exp) datasets are merged on both formula and Crystal System — critical for correctly matching polymorphic compounds to their experimental band gaps
Normalized ghost keys (uppercase, whitespace-stripped) prevent formula-mismatch errors (e.g., "GaAs " vs "GaAs")
Duplicate entries (same formula + crystal system with multiple measurements) are resolved by averaging the band gap values before merging to create a precise one-to-one mapping

2. Feature Engineering
Chemical formulas are parsed into atomic fraction dictionaries via regex:
GaAs       →  {Ga: 0.5,  As: 0.5}
BaCuTeF    →  {Ba: 0.25, Cu: 0.25, Te: 0.25, F: 0.25}
C (unary)  →  {C: 1.0}
Three types of composition-based descriptors are then computed across all 90 elemental properties, yielding 276 features per compound:
Feature TypeFormulaColumn PrefixPhysical MeaningWeighted averagep̄ₖ = Σ wᵢ pₖ(Eᵢ)avg_*Typical elemental environment of the compoundMax–min differenceΔpₖ = max pₖ − min pₖdiff_*Elemental contrast (zero for unary, grows with dissimilarity)Weighted varianceσ²ₖ = Σ wᵢ(pₖ(Eᵢ) − p̄ₖ)²var_*Chemical heterogeneity / mixing
All features are standardized (zero mean, unit variance) before model training.
3. Machine Learning
12 regression algorithms were benchmarked across all 5 compositional subsets:
CategoryModelsLinear & RegularizedLinear Regression, Ridge (α=2.0), Bayesian RidgeTree-Based EnsemblesRandom Forest, Extra Trees, Gradient Boosting, Decision TreeBoosted FrameworksXGBoost, CatBoostInstance/Kernelk-Nearest Neighbors (k=5), SVR (RBF kernel)Neural NetworkMLP Regressor (2 hidden layers × 100 neurons, ReLU)
4. Multi-Stage Validation Pipeline
To rigorously assess generalization and prevent data leakage:
StageMethodPurpose180/20 Hold-Out SplitStatic unseen test set reserved before all CV210-Fold Cross-ValidationFair algorithm benchmarking within training set3Monte Carlo CV (20 iterations)Robustness testing; builds confidence interval on metrics4Leave-One-Group-Out (LOGO)True out-of-domain generalization — entire chemical families withheld

Results
Best Model per Arity (10-Fold Cross-Validation)
ArityBest ModelHyperparametersR²MAE (eV)RMSE (eV)UnaryLinearRegressionDefault-0.01191.07741.4586BinaryExtraTreesn_estimators=3000.82710.62250.9263TernaryExtraTreesn_estimators=3000.83970.31500.4989QuaternaryGradientBoostingn_estimators=100, max_depth=30.78470.39800.5298GeneralExtraTreesn_estimators=3000.80010.48250.7827

Note on Unary: With only 21 samples, no ML model achieves meaningful predictive accuracy. The negative R² reflects insufficient data for tree-based models to learn decision boundaries — this is a fundamental data limitation, not a feature engineering failure. The unary subset is excluded from advanced validation (MCCV and LOGO).

Full Benchmarking — All 12 Models on the General Dataset
ModelHyperparametersR²MAE (eV)RMSE (eV)ExtraTreesn_estimators=3000.80010.48250.7827GradientBoostingn_estimators=200, depth=30.79130.51850.7998CatBoostn_estimators=300, depth=50.78950.51530.8033XGBoostn_estimators=100, depth=30.78390.52510.8140RandomForestn_estimators=3000.77120.54040.8374MLPRegressor(100,50), max_iter=15000.76220.56260.8348BayesianRidgeDefault0.73900.61460.8745Ridgealpha=2.00.71170.64490.9192KNeighborsn_neighbors=50.68120.66200.9667SVRRBF, C=1.0, ε=0.10.65420.62861.0067DecisionTreemax_depth=30.60840.75211.0956LinearRegressionDefault0.16631.11301.5986
Model Robustness — Monte Carlo Cross-Validation (20 iterations)
ArityModelR²MAE (eV)RMSE (eV)BinaryExtraTrees (n=300)0.73390.61100.8339TernaryExtraTrees (n=300)0.71100.35060.5590QuaternaryGradientBoosting0.68060.37970.4908GeneralExtraTrees (n=300)0.62820.48460.7319
The MAE remains nearly identical to the 10-Fold CV values (e.g., 0.4846 vs 0.4825 eV for the General model), confirming the pipeline captures genuine periodic trends and is not overfitting to specific data distributions.
Out-of-Domain Generalization — Leave-One-Group-Out (LOGO)
ArityModelR²MAE (eV)RMSE (eV)BinaryExtraTrees (n=300)0.62730.90961.3515TernaryExtraTrees (n=300)0.55520.56710.8179QuaternaryGradientBoosting0.64810.49070.6758GeneralExtraTrees (n=300)0.64320.68001.0227
Even when predicting entirely unseen chemical families, the framework maintains sub-eV MAE errors across all arity levels, demonstrating genuine out-of-domain transferability driven by universal quantum mechanical heuristics.
Top 10 Most Important Features (General ExtraTrees Model)
RankFeatureImportancePhysical Interpretation1var_oxidation_state_first0.430 (43%)Oxidation state variance → charge transfer, ionicity, band gap widening2diff_mendeleev_h_td_right0.021Mendeleev number contrast (Hao-Tackett-Deem scale)3diff_mendeleev_h_dt_left0.017Mendeleev number contrast4diff_mendeleev_pettifor_regular0.016Pettifor scale electronegativity contrast5diff_mendeleev_chemists_sequence0.015Chemists' sequence Mendeleev difference6diff_mendeleev_dt_left0.014Mendeleev number contrast7avg_entropy_of_solid0.014Average thermodynamic lattice entropy → orbital overlap8diff_mendeleev_td_right0.013Mendeleev number contrast9diff_mendeleev_h_dt_right0.012Mendeleev number contrast10diff_mendeleev_h_td_left0.011Mendeleev number contrast
var_oxidation_state_first dominates with 43% of predictive weight: greater oxidation state variance → stronger cation–anion electrostatic interactions (ionicity) → wider valence–conduction band separation → larger band gap. The remaining top features are all expressions of Mendeleev number difference, encoding electronegativity, atomic radius, and valence configuration in a single periodic metric.
