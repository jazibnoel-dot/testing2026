# noshow_iq/model.py
# Trains and saves a RandomForest with class-imbalance handling.

import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

MODEL_PATH = Path('models/model.pkl')
_FEATURE_ORDER = None   # set during train(), read during predict()


def train(X, y, save: bool = True):
    """Train a classifier and optionally save it to disk."""
    global _FEATURE_ORDER
    _FEATURE_ORDER = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',   # <-- addresses 80/20 imbalance
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    metrics = evaluate(clf, X_test, y_test)

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({'model': clf, 'features': _FEATURE_ORDER}, MODEL_PATH)

    return clf, metrics


def evaluate(clf, X_test, y_test) -> dict:
    """Return per-class precision/recall/F1 as a dict."""
    preds = clf.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average=None)
    return {
        'show':    {'precision': p[0], 'recall': r[0], 'f1': f1[0]},
        'no_show': {'precision': p[1], 'recall': r[1], 'f1': f1[1]},
        'report':  classification_report(y_test, preds),
    }


def load():
    """Load the saved model bundle from disk."""
    bundle = joblib.load(MODEL_PATH)
    return bundle['model'], bundle['features']


def predict(record: dict) -> dict:
    """Single-record prediction.  record must be a cleaned, engineered feature dict."""
    import pandas as pd
    clf, features = load()
    X = pd.DataFrame([record]).reindex(columns=features, fill_value=0)
    proba = float(clf.predict_proba(X)[0, 1])

    risk = 'HIGH' if proba >= 0.5 else 'MEDIUM' if proba >= 0.3 else 'LOW'
    rec = {
        'HIGH':   'Send SMS reminder + confirmation call. Consider double-booking the slot.',
        'MEDIUM': 'Send SMS reminder 24 h before.',
        'LOW':    'No action needed.',
    }[risk]
    return {'risk_level': risk, 'probability': round(proba, 3), 'recommendation': rec}
