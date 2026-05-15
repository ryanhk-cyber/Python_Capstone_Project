import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import KENYA_COUNTIES, MODEL_FILE
from data.kenya_flood_drainage_dataset import (
    DRAINAGE_QUALITY_MAP,
    FLOOD_RISK_LABEL_MAP,
    get_training_samples,
)


def build_county_lookup():
    return {c["name"]: c for c in KENYA_COUNTIES}


def extract_features(sample, county_lookup):
    c = county_lookup.get(sample["County"], {})
    return [
        DRAINAGE_QUALITY_MAP.get(sample["Drainage_Quality"], 2),
        c.get("elevation", 1000),
        c.get("near_water", 0.30),
        c.get("slope_class", 3),
        c.get("hist_flood", 0.30),
    ]


def build_training_data():
    county_lookup = build_county_lookup()
    samples = get_training_samples()
    X, y = [], []
    for s in samples:
        X.append(extract_features(s, county_lookup))
        y.append(FLOOD_RISK_LABEL_MAP[s["Flood_Risk_Level"]])
    return np.array(X), np.array(y)


def train_model(force=False):
    if os.path.exists(MODEL_FILE) and not force:
        return joblib.load(MODEL_FILE)

    X, y = build_training_data()
    # Some stub datasets may not have enough samples per class for stratification.
    # If any class has <2 members, fall back to a non-stratified split.
    unique, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0

    split_kwargs = {"test_size": 0.20, "random_state": 42}
    if min_count < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **split_kwargs
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **split_kwargs, stratify=y
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train_s, y_train)

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler}, MODEL_FILE)
    return {"model": clf, "scaler": scaler}


def predict_flood_probability(risk_df):
    bundle = train_model()
    clf = bundle["model"]
    scaler = bundle["scaler"]
    county_lookup = build_county_lookup()
    label_to_level = {v: k for k, v in FLOOD_RISK_LABEL_MAP.items()}

    rows = []
    for _, row in risk_df.iterrows():
        c = county_lookup.get(row["county"], {})
        features = [
            c.get("drainage", 3),
            c.get("elevation", 1000),
            c.get("near_water", 0.30),
            c.get("slope_class", 3),
            c.get("hist_flood", 0.30),
        ]
        X = scaler.transform([features])
        proba = clf.predict_proba(X)[0]
        predicted = int(clf.predict(X)[0])
        flood_prob = float(proba[2] + proba[3]) if len(proba) > 3 else float(proba[-1])

        rows.append({
            "county":             row["county"],
            "ml_flood_prob":      round(flood_prob, 3),
            "ml_predicted_level": label_to_level.get(predicted, "Moderate"),
        })

    return pd.DataFrame(rows)
