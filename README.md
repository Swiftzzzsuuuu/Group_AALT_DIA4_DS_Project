# Group AALT (DIA4) – Early Detection of Students at Risk of Dropout

Group AALT (DIA4) – Léo Chen, Adam Gaabi, Timéo Brajon, Axel Warot

ESILV A4 DIA – Machine Learning project

Predict the final status of a student (Dropout / Enrolled / Graduate) at enrollment (scenario A), after semester 1 (scenario B, main use case) and after semester 2 (scenario C, upper bound).

## Data
UCI *Predict Students' Dropout and Academic Success* (id 697, CC BY 4.0): https://archive.ics.uci.edu/dataset/697
`data/raw/data.csv` is the original semicolon-separated file (4,424 rows, 37 columns).

## Structure
```
data/raw/                 original data
data/processed/           clean dataset, train.csv, test.csv (test sealed until week 4)
src/features.py           column groups, scenarios, cleaning, preprocessing pipeline
src/evaluation.py         shared outer folds, metrics, threshold rule (week 3)
notebooks/01_eda_data_quality.ipynb      data quality and EDA              (Timéo Brajon)
notebooks/02_preprocessing_split.ipynb   cleaning, features, split          (Léo Chen)
notebooks/03_baseline.ipynb              baseline and evaluation protocol   (Adam Gaabi)
notebooks/04_automl_v0_1.ipynb           AutoML pipeline v0.1               (Axel Warot)
notebooks/05_boosting_nested_cv.ipynb    limitations of v0.1, boosting      (Léo Chen)
notebooks/06_imbalance_threshold.ipynb   imbalance, dropout threshold       (Adam Gaabi)
notebooks/07_tabnet.ipynb                out-of-course model: TabNet        (Timéo Brajon)
notebooks/08_automl_v0_2_ensembles.ipynb AutoML v0.2, ensembles, comparison (Axel Warot)
results/                  CV results (csv)
report/                   Report_Group_AALT_DIA4 (weeks 2 and 3, LaTeX source, figures, PDF)
ai_governance/            CONTRIBUTIONS.md, PROMPT_LOG.md, AI_AUDIT.md
```

## Run
```
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```
Run the notebooks in order: 02 writes the split, 03 the baseline results used by 04, 05 the hyperparameters used by 06 and 08, and 08 reads the results of 05–07. All paths are relative, all random seeds are fixed (42). Notebooks 05 and 07 take about 13 and 6 minutes on 2 CPU cores.
