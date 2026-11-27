# Contributions – Group AALT (DIA4)

## Timéo Brajon – Data quality and EDA
Notebook: notebooks/01_eda_data_quality.ipynb
Missing values, duplicates, variable types, class imbalance, outliers, zero grades, cohort variables, correlations.
Reviewed by: Léo

## Léo Chen – Cleaning, features and split
Files: src/features.py, notebooks/02_preprocessing_split.ipynb, data/processed/
Column renaming, engineered features, scenarios A/B/C, encoding pipeline, train/test split, leakage checks.
Reviewed by: Adam

## Adam Gaabi – Baseline and evaluation
Notebook: notebooks/03_baseline.ipynb
CV protocol, metrics, dummy and logistic regression baselines, tuning of C, calibration, sensitivity test.
Reviewed by: Axel

## Axel Warot – AutoML v0.1
Notebook: notebooks/04_automl_v0_1.ipynb
Search space, randomised search over 6 models, comparison with the baseline.
Reviewed by: Timéo

## Week 3

## Léo Chen – Boosting and nested CV
Notebook: notebooks/05_boosting_nested_cv.ipynb
Limitations of v0.1, random forest, HistGradientBoosting, LightGBM, XGBoost, nested CV, final hyperparameters.
Reviewed by: Adam

## Adam Gaabi – Imbalance and decision threshold
Notebook: notebooks/06_imbalance_threshold.ipynb
Class weights vs SMOTE vs undersampling, dropout threshold tuned inside the training folds.
Reviewed by: Axel

## Timéo Brajon – TabNet
Notebook: notebooks/07_tabnet.ipynb
Out-of-course model (TabNet), comparison with an MLP, feature importance from the masks.
Reviewed by: Léo

## Axel Warot – AutoML v0.2
Notebook: notebooks/08_automl_v0_2_ensembles.ipynb
Soft voting and stacking, scenarios A/B/C, leave-one-cohort-out, overfitting control, final comparison.
Reviewed by: Timéo

## Shared
Week 2 and week 3 reports and ai_governance files: everyone, compiled by Axel.
