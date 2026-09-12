"""
Example: use the trained model the way the app would -- a partial set
of selected symptoms, and yes/no answers to only the follow-up
questions that were actually asked. Anything not asked is filled with
UNKNOWN (0.5) rather than guessed as 0, since "not asked" isn't the
same evidence as "answered no".

Run: python3 predict.py
"""
import joblib
import numpy as np

from definitions import FEATURE_NAMES, UNKNOWN

MODEL_PATH = 'model/condition_classifier.joblib'
LABELS_PATH = 'model/label_encoder.joblib'


def build_feature_vector(symptoms: set[str], answers: dict[str, bool]) -> np.ndarray:
    """symptoms: set of symptom ids the user selected (all others = 0).
    answers: dict of question_id -> True/False for questions actually asked.
    Unasked questions are encoded as 0.5 (unknown), not 0.
    """
    vec = []
    for name in FEATURE_NAMES:
        if name in symptoms:
            vec.append(1.0)
        elif name in answers:
            vec.append(1.0 if answers[name] else 0.0)
        elif name.startswith('q_'):
            vec.append(UNKNOWN)
        else:
            vec.append(0.0)  # symptom not selected
    return np.array(vec).reshape(1, -1)


def predict(symptoms: set[str], answers: dict[str, bool], top_k=3):
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(LABELS_PATH)

    x = build_feature_vector(symptoms, answers)
    probs = model.predict_proba(x)[0]

    ranked = sorted(zip(encoder.classes_, probs), key=lambda p: p[1], reverse=True)
    return ranked[:top_k]


if __name__ == '__main__':
    # Example: a patient who reports fever, cough, sore throat, and
    # answered a few follow-up questions.
    selected_symptoms = {'fever', 'cough', 'soreThroat', 'bodyAche'}
    answered_questions = {
        'q_duration': True,
        'q_worsening': False,
        'q_highFever': False,
        'q_contact': True,
    }

    results = predict(selected_symptoms, answered_questions)
    print('Top matches:')
    for label, prob in results:
        print(f'  {label:20s} {prob*100:5.1f}%')
