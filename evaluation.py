import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, recall_score, precision_score, roc_auc_score
from features import CLASSES, SEED

LABELS = {c: i for i, c in enumerate(CLASSES)}
DROPOUT = LABELS["Dropout"]
OUTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
INNER = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


def encode(y):
    return y.map(LABELS).to_numpy()


def scores(y_true, y_pred, proba=None):
    s = {"macro_f1": f1_score(y_true, y_pred, average="macro"),
         "bal_acc": balanced_accuracy_score(y_true, y_pred),
         "accuracy": accuracy_score(y_true, y_pred),
         "dropout_recall": recall_score(y_true, y_pred, labels=[DROPOUT], average="macro", zero_division=0),
         "dropout_precision": precision_score(y_true, y_pred, labels=[DROPOUT], average="macro", zero_division=0),
         "enrolled_f1": f1_score(y_true, y_pred, labels=[LABELS["Enrolled"]], average="macro", zero_division=0)}
    s["roc_auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr") if proba is not None else np.nan
    return s


def sk(estimator):
    def fit_predict(X_tr, y_tr, X_va):
        m = clone(estimator).fit(X_tr, y_tr)
        return m.predict(X_va), m.predict_proba(X_va)
    return fit_predict


def run_folds(fit_predict, X, y, name, scenario, cv=OUTER, groups=None):
    rows = []
    for k, (a, b) in enumerate(cv.split(X, y, groups)):
        pred, proba = fit_predict(X.iloc[a], y[a], X.iloc[b])
        rows.append({"model": name, "scenario": scenario, "fold": k, **scores(y[b], pred, proba)})
    return pd.DataFrame(rows)


def summary(df, metrics=("macro_f1", "bal_acc", "dropout_recall", "enrolled_f1", "roc_auc_ovr")):
    g = df.groupby(["scenario", "model"], sort=False)
    out = g[list(metrics)].mean()
    out.insert(1, "macro_f1_std", g["macro_f1"].std())
    return out.sort_values("macro_f1", ascending=False)


def threshold_predict(proba, t):
    other = np.where(proba[:, LABELS["Enrolled"]] >= proba[:, LABELS["Graduate"]], LABELS["Enrolled"], LABELS["Graduate"])
    return np.where(proba[:, DROPOUT] >= t, DROPOUT, other)
