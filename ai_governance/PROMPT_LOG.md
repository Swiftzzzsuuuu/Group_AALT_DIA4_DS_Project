# Prompt log – Group AALT (DIA4)

## Week 1 – specification

We did not use AI for the week 1 report.

## Week 2 – prompts

### 1. Timéo – EDA

**Prompt:**
```
I have the UCI student dropout dataset (data.csv, separator ";", 4424 rows, 37 columns, target = Dropout /
Enrolled / Graduate). Write the data quality checks in pandas: missing values, duplicates, types, outliers
with IQR, and the target distribution. Don't scale or encode anything here, it's only exploration.
Only write comments that the outputs actually show.
```
**Kept / Rejected:** kept the checks; removed the comment about a hidden tab in column names, our file
had none (AI_AUDIT §5).

### 2. Léo – features and split

**Prompt:**
```
For the same dataset, write a function that adds approval_rate = approved / enrolled and a no_grade flag
for semester 1 and 2. Then do a stratified 80/20 train/test split with random_state=42 and save both files.
Scenario A must not use any semester column and scenario B no semester 2 column.
The encoding and scaling must not be fitted here, only later in a Pipeline.
```
**Kept / Rejected:** kept the functions; checked that the dataset version had the 37 UCI columns
(the first one was a Kaggle copy, AI_AUDIT §1).

### 3. Adam – baseline

**Prompt:**
```
Using only train.csv, compare a DummyClassifier and a LogisticRegression (with and without
class_weight="balanced") with 5-fold stratified CV repeated 3 times. Metrics: macro-F1, balanced accuracy
and recall on Dropout. Put the preprocessing inside the Pipeline so there is no leakage.
Don't write the conclusions, I will write them from the results.
```
**Kept / Rejected:** kept the code; rewrote two conclusions that didn't match the outputs
(AI_AUDIT §3, §4).

### 4. Axel – AutoML v0.1

**Prompt:**
```
Write a RandomizedSearchCV on scenario B that tries several models (logistic regression, tree, KNN, SVM,
naive Bayes, random forest) and some feature selection methods, scored with macro-F1. Everything must be
in one Pipeline. The target has 3 classes so check that the solvers support multiclass, and use
error_score="raise" so errors are not hidden.
```
**Kept / Rejected:** kept the search; replaced solver="liblinear" by "saga" for the L1 selector because it
failed on 3 classes (AI_AUDIT §2).

## Week 3 – prompts

### 5. Léo – boosting and nested CV

**Prompt:**
```
I want to add HistGradientBoosting, LightGBM and XGBoost to our project and compare them with the
logistic regression baseline on scenario B. Can you write the search spaces for RandomizedSearchCV
(20 iterations, inner 3-fold, macro-F1)? I already have a run_folds function for the outer 5-fold, the
search must be done inside each outer training fold (nested CV). Keep the preprocessing in the Pipeline.
```
**Kept / Rejected:** kept the search spaces (reduced some ranges); rejected the first version that
reported best_score_ of the search instead of the outer fold score (AI_AUDIT §6). Comments on the
results written by us.

### 6. Adam – imbalance and threshold

**Prompt:**
```
How can I compare class_weight="balanced", SMOTE and random undersampling for LogReg and LightGBM
without leakage? The resampling must only be done on the training folds. Then I need a decision
threshold on P(Dropout) to reach 80% dropout recall: how do I choose it without using the validation fold?
```
**Kept / Rejected:** kept the idea of choosing the threshold on inner out-of-fold probabilities; replaced
the sklearn Pipeline by the imblearn Pipeline because SMOTE raised an error (AI_AUDIT §7). We wrote the
threshold loop and the business interpretation ourselves.

### 7. Timéo – TabNet

**Prompt:**
```
Explain TabNet (Arik & Pfister, 2021) in a few lines: attentive transformer, sparsemax masks, feature
importance. Then give an example of how to train pytorch_tabnet TabNetClassifier with early stopping and
class weights on a numpy array. I have about 3,500 rows with 3 classes.
```
**Kept / Rejected:** used the explanation as a starting point for the markdown (checked against the
paper); rejected the early stopping on the outer validation fold and the claim that TabNet should beat
boosting (AI_AUDIT §8). Added the inner 15% validation split and the MLP comparison ourselves.

### 8. Axel – AutoML v0.2 and ensembles

**Prompt:**
```
Using the best params saved in best_params.json, how do I build a soft VotingClassifier and a
StackingClassifier with LogReg, random forest and LightGBM, each in a Pipeline with our preprocessor?
Also, how do I do a leave-one-group-out validation where the group is the cohort
(unemployment, inflation, GDP)?
```
**Kept / Rejected:** kept the VotingClassifier / StackingClassifier syntax and the LeaveOneGroupOut call;
removed the macro variables from the preprocessor because they are constant in the test cohort
(AI_AUDIT §9). The final comparison table and the analysis are ours.

### 9. Shared – week 3 report

**Prompt:**
```
Turn this CSV (week3_comparison.csv) into a LaTeX table with booktabs, 3 decimals, best macro-F1 in bold.
```
**Kept / Rejected:** kept the table format; the numbers were checked against the CSV.
