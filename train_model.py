"""
Train a symptom -> likely-condition classifier and evaluate it.

Compares a Logistic Regression baseline against a Random Forest, picks
the better one by cross-validated macro-F1, refits it on the full
training split, reports a held-out test evaluation, and saves the
model + label encoder + feature order to model/ for reuse (see
predict.py).
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from definitions import FEATURE_NAMES

DATA_PATH = 'data/symptom_dataset.csv'
MODEL_PATH = 'model/condition_classifier.joblib'
LABELS_PATH = 'model/label_encoder.joblib'
META_PATH = 'model/metadata.json'


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES].values
    y_raw = df['label'].values
    return X, y_raw


def main():
    X, y_raw = load_data()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        'logistic_regression': LogisticRegression(max_iter=2000),
        'random_forest': RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42
        ),
    }

    print('=== 5-fold cross-validation on training split (macro-F1) ===')
    cv_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
        cv_scores[name] = scores.mean()
        print(f'{name:20s} mean={scores.mean():.4f}  std={scores.std():.4f}')

    best_name = max(cv_scores, key=cv_scores.get)
    best_model = candidates[best_name]
    print(f'\nSelected model: {best_name}')

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print('\n=== Held-out test report ===')
    report = classification_report(
        y_test, y_pred, target_names=encoder.classes_, zero_division=0
    )
    print(report)
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f'Test macro-F1: {test_macro_f1:.4f}')

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    print('\n=== Confusion matrix (rows=true, cols=predicted) ===')
    print(cm_df.to_string())

    if hasattr(best_model, 'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_NAMES)
        print('\n=== Top 10 feature importances ===')
        print(importances.sort_values(ascending=False).head(10).to_string())

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoder, LABELS_PATH)
    with open(META_PATH, 'w') as f:
        json.dump({
            'model_name': best_name,
            'cv_macro_f1': cv_scores,
            'test_macro_f1': test_macro_f1,
            'feature_names': FEATURE_NAMES,
            'classes': encoder.classes_.tolist(),
        }, f, indent=2)

    print(f'\nSaved model -> {MODEL_PATH}')
    print(f'Saved label encoder -> {LABELS_PATH}')
    print(f'Saved metadata -> {META_PATH}')


if __name__ == '__main__':
    main()
