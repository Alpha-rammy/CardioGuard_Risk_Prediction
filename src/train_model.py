import os
import warnings
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)



# Load Data


def load_data():
    path = os.path.join(
        DATA_PROCESSED,
        "cardioguard_model_features.csv"
    )

    print("Loading:", path)
    print("Exists:", os.path.exists(path))

    return pd.read_csv(path)


# Train Model


def train_model():
    df = load_data()

    target = "high_risk_flag"

    exclude = [
        "participant_id",
        target
    ]

    feature_cols = [
        col for col in df.columns
        if col not in exclude
    ]

    X = df[feature_cols]
    y = df[target]

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    scale_pos_weight = (
        y_train.value_counts()[0] / y_train.value_counts()[1]
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred)
        })

        trained_models[name] = pipe

        print(f"{name} trained successfully")

    results_df = pd.DataFrame(results).sort_values(
        by="f1",
        ascending=False
    )

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    best_threshold = 0.50

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "cardioguard_best_model.pkl")
    )

    joblib.dump(
        feature_cols,
        os.path.join(MODEL_DIR, "cardioguard_feature_cols.pkl")
    )

    joblib.dump(
        best_threshold,
        os.path.join(MODEL_DIR, "cardioguard_best_threshold.pkl")
    )

    results_df.to_csv(
        os.path.join(DATA_PROCESSED, "cardioguard_model_results.csv"),
        index=False
    )

    print("\nModel training completed.")
    print(f"Best model: {best_model_name}")
    print(results_df.round(4))


if __name__ == "__main__":
    train_model()