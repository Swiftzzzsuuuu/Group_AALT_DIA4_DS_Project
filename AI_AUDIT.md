# AI audit – Group AALT (DIA4)

We used Claude to draft code and text. We ran every notebook and compared the outputs with what the AI had written. Errors found so far:

## 1. Wrong version of the dataset (Léo – data)
The first CSV the AI found was the Kaggle re-upload: 34 features instead of 36 (no admission grade, no previous qualification grade) and course codes 1–17 instead of the UCI codes.
Caught: column count and codes did not match the UCI variable table.
Fix: used a copy with the UCI format (37 columns, 2209 / 1421 / 794 students per class), checked against a second copy.

## 2. Silent failure in the AutoML search (Axel – AutoML)
The L1 feature selector used solver="liblinear", which does not handle 3 classes. With the default error_score=nan, all these configurations failed without any message and were just left out of the ranking.
Caught: re-ran with error_score="raise", which gave "liblinear does not support multiclass".
Fix: solver="saga", and error_score="raise" kept.

## 3. Calibration conclusion written before the results (Adam – baseline)
The text said class weights over-estimate the dropout probability.
Caught: the output shows a mean predicted probability of 0.309 vs a real rate of 0.321, so no over-estimation.
Fix: conclusion rewritten from the actual numbers.

## 4. Wrong comment on the C parameter (Adam – baseline)
The text said the score is flat for C >= 0.1.
Caught: nested CV picked C between 0.03 and 0.3, best score at C = 0.1.
Fix: text corrected (differences smaller than one std, C = 1 kept as reference).

## 5. False claim about column names (Timéo – EDA)
The summary said the column names contained a hidden tab.
Caught: the check cell printed an empty list for our file.
Fix: claim removed.

## Week 3

## 6. Optimistic score in the nested CV (Léo – boosting)
The first version returned the best_score_ of the inner search as the result of each fold, so the score was measured on the folds used for tuning.
Caught: the scores were higher than the outer fold scores for every model.
Fix: only the outer fold score is reported (run_folds), the search stays inside the outer training fold.

## 7. SMOTE in a scikit-learn Pipeline (Adam – imbalance)
The AI put SMOTE as a step of a sklearn Pipeline, which does not accept samplers.
Caught: TypeError when fitting the pipeline.
Fix: imblearn Pipeline, so SMOTE and undersampling are only applied to the training folds.

## 8. Early stopping on the validation fold (Timéo – TabNet)
The example used the outer validation fold as eval_set for early stopping, so this fold was used to choose the number of epochs. The text also said TabNet should beat boosting.
Caught: reading the code; the results then showed TabNet is the weakest model (0.626 vs 0.702 for LightGBM).
Fix: 15% inner validation split for early stopping and the choice of configuration; claim replaced by the critical benchmark (Grinsztajn et al., 2022) and our results.

## 9. Macro variables in the cohort validation (Axel – AutoML v0.2)
The leave-one-cohort-out code kept unemployment, inflation and GDP as features, but they define the cohort and are constant in the test cohort.
Caught: checking which columns were used in each fold.
Fix: the 3 macro variables are removed from the preprocessor for this validation.

## Points from week 2
- SMOTE before the split: checked, resampling is only inside the imblearn pipeline on the training folds (notebook 06).
- Enrolled F1 of 0.84 (Villar & de Andrade, 2024): not reached with our protocol (best Enrolled F1 = 0.527, soft voting). We do not resample the validation data and we only use semester-1 information, so the results are not directly comparable (see the literature part of the report).
