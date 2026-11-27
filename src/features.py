import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]

RENAME = {
    "Marital status": "marital_status", "Application mode": "application_mode", "Application order": "application_order",
    "Course": "course", "Daytime/evening attendance": "daytime_attendance", "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_qualification_grade", "Nacionality": "nationality",
    "Mother's qualification": "mother_qualification", "Father's qualification": "father_qualification",
    "Mother's occupation": "mother_occupation", "Father's occupation": "father_occupation", "Admission grade": "admission_grade",
    "Displaced": "displaced", "Educational special needs": "special_needs", "Debtor": "debtor",
    "Tuition fees up to date": "tuition_fees_up_to_date", "Gender": "gender", "Scholarship holder": "scholarship_holder",
    "Age at enrollment": "age_at_enrollment", "International": "international",
    "Unemployment rate": "unemployment_rate", "Inflation rate": "inflation_rate", "GDP": "gdp", "Target": TARGET}
for s, k in [("1st", "sem1"), ("2nd", "sem2")]:
    for m in ["credited", "enrolled", "evaluations", "approved", "grade", "without evaluations"]:
        RENAME[f"Curricular units {s} sem ({m})"] = f"{k}_{m.replace(' ', '_')}"

NOMINAL = ["marital_status", "application_mode", "course", "previous_qualification", "nationality",
           "mother_qualification", "father_qualification", "mother_occupation", "father_occupation"]
BINARY = ["daytime_attendance", "displaced", "special_needs", "debtor", "tuition_fees_up_to_date", "gender",
          "scholarship_holder", "international"]
NUMERIC_ENROLL = ["application_order", "previous_qualification_grade", "admission_grade", "age_at_enrollment",
                  "unemployment_rate", "inflation_rate", "gdp"]
SEM = ["credited", "enrolled", "evaluations", "approved", "grade", "without_evaluations"]
SEM1 = [f"sem1_{m}" for m in SEM]
SEM2 = [f"sem2_{m}" for m in SEM]
ENG1 = ["sem1_approval_rate", "sem1_no_grade"]
ENG2 = ["sem2_approval_rate", "sem2_no_grade"]

SCENARIOS = {
    "A_enrollment": {"nominal": NOMINAL, "binary": BINARY, "numeric": NUMERIC_ENROLL},
    "B_semester1": {"nominal": NOMINAL, "binary": BINARY, "numeric": NUMERIC_ENROLL + SEM1 + ENG1},
    "C_all": {"nominal": NOMINAL, "binary": BINARY, "numeric": NUMERIC_ENROLL + SEM1 + ENG1 + SEM2 + ENG2},
}


def load_raw(path):
    d = pd.read_csv(path, sep=";")
    d.columns = d.columns.str.strip()
    return d.rename(columns=RENAME)


def add_features(d):
    d = d.copy()
    for k in ["sem1", "sem2"]:
        d[f"{k}_approval_rate"] = np.where(d[f"{k}_enrolled"] > 0, d[f"{k}_approved"] / d[f"{k}_enrolled"].clip(lower=1), 0.0)
        d[f"{k}_no_grade"] = (d[f"{k}_grade"] == 0).astype(int)
    return d


def columns(scenario):
    s = SCENARIOS[scenario]
    return s["nominal"] + s["binary"] + s["numeric"]


def preprocessor(scenario, min_frequency=20):
    s = SCENARIOS[scenario]
    return ColumnTransformer([
        ("nom", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency, sparse_output=False), s["nominal"]),
        ("bin", "passthrough", s["binary"]),
        ("num", StandardScaler(), s["numeric"])])


def split_xy(d, scenario):
    return d[columns(scenario)], d[TARGET]
