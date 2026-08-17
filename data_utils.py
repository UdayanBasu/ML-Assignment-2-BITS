import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

TARGET = "Revenue"

NUMERIC_COLUMNS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay"
]

CATEGORICAL_COLUMNS = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend"
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

def normalize_bool(value):
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    return 1 if text in {"true", "1", "yes"} else 0

def fetch_full_dataset():
    dataset = fetch_ucirepo(id=468)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    X = X[FEATURE_COLUMNS].copy()
    y = y.map(normalize_bool).astype(int)
    return X, y

def build_preprocessor(scale_numeric=False):
    numeric_steps = []
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = (
        Pipeline(numeric_steps) if numeric_steps else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("model", DecisionTreeClassifier(max_depth=8, random_state=42))
        ]),
        "kNN": Pipeline([
            ("prep", build_preprocessor(scale_numeric=True)),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": Pipeline([
            ("prep", build_preprocessor(scale_numeric=True)),
            ("model", GaussianNB())
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ]),
    }

def prepare_uploaded_test(df):
    df = df.copy()
    if TARGET not in df.columns:
        raise ValueError(f"Uploaded CSV must contain the '{TARGET}' column.")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].map(normalize_bool).astype(int)
    return X, y
