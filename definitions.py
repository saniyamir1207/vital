"""
Symptom / question / condition definitions.

These are copied 1:1 from the CONDITIONS / SYMPTOM_IDS / QUESTION_IDS
objects in the app's index.html so the ML model speaks the same
vocabulary as the UI. If you edit CONDITIONS in index.html, mirror the
change here and re-run data_gen.py + train_model.py.
"""

SYMPTOM_IDS = [
    'fever', 'cough', 'headache', 'fatigue', 'soreThroat', 'bodyAche',
    'nausea', 'diarrhea', 'chestPain', 'breathShort', 'dizziness', 'rash',
    'jointPain', 'abdominalPain', 'runnyNose', 'lossAppetite',
    'frequentUrination', 'anxietyMood', 'insomnia', 'burningUrination',
]

QUESTION_IDS = [
    'q_duration', 'q_worsening', 'q_highFever', 'q_contact', 'q_breath',
    'q_rashSpread', 'q_vomit', 'q_chestPressure', 'q_weightLoss',
    'q_chronic', 'q_travel', 'q_lightSound', 'q_burning', 'q_stress',
    'q_seasonal', 'q_painScale', 'q_appetite', 'q_recurring',
]

# condition -> {symptoms: {id: weight}, questions: [ids], specialty}
CONDITIONS = {
    'commonCold': {
        'symptoms': {'cough': 2, 'soreThroat': 2, 'runnyNose': 2, 'fever': 1, 'fatigue': 1, 'bodyAche': 1},
        'questions': ['q_duration', 'q_worsening', 'q_highFever', 'q_contact'],
        'specialty': 'generalPhysician',
    },
    'migraine': {
        'symptoms': {'headache': 3, 'dizziness': 1, 'nausea': 1, 'fatigue': 1},
        'questions': ['q_lightSound', 'q_painScale', 'q_recurring', 'q_stress'],
        'specialty': 'neurologist',
    },
    'gastro': {
        'symptoms': {'nausea': 2, 'diarrhea': 2, 'abdominalPain': 2, 'lossAppetite': 1, 'fatigue': 1},
        'questions': ['q_vomit', 'q_duration', 'q_appetite', 'q_contact'],
        'specialty': 'gastroenterologist',
    },
    'allergicRhinitis': {
        'symptoms': {'runnyNose': 2, 'cough': 1, 'soreThroat': 1, 'fatigue': 1},
        'questions': ['q_seasonal', 'q_rashSpread', 'q_duration', 'q_recurring'],
        'specialty': 'entAllergist',
    },
    'uti': {
        'symptoms': {'burningUrination': 3, 'frequentUrination': 3, 'abdominalPain': 1, 'fever': 1},
        'questions': ['q_burning', 'q_highFever', 'q_duration', 'q_recurring'],
        'specialty': 'urologist',
    },
    'dengue': {
        'symptoms': {'fever': 2, 'headache': 1, 'jointPain': 2, 'rash': 1, 'fatigue': 1, 'nausea': 1},
        'questions': ['q_highFever', 'q_rashSpread', 'q_travel', 'q_painScale'],
        'specialty': 'infectiousDisease',
    },
    'anxietyCond': {
        'symptoms': {'anxietyMood': 3, 'insomnia': 2, 'fatigue': 1, 'dizziness': 1, 'headache': 1},
        'questions': ['q_stress', 'q_recurring', 'q_appetite', 'q_worsening'],
        'specialty': 'psychiatrist',
    },
    'asthma': {
        'symptoms': {'breathShort': 3, 'cough': 2, 'chestPain': 1, 'fatigue': 1},
        'questions': ['q_breath', 'q_chronic', 'q_worsening', 'q_seasonal'],
        'specialty': 'pulmonologist',
    },
    'skinAllergy': {
        'symptoms': {'rash': 3, 'fatigue': 1},
        'questions': ['q_rashSpread', 'q_seasonal', 'q_duration', 'q_recurring'],
        'specialty': 'dermatologist',
    },
    'hypertensionRisk': {
        'symptoms': {'headache': 1, 'dizziness': 2, 'chestPain': 2, 'fatigue': 1},
        'questions': ['q_chestPressure', 'q_chronic', 'q_worsening', 'q_weightLoss'],
        'specialty': 'cardiologist',
    },
}

CONDITION_IDS = list(CONDITIONS.keys())

# extra class for "answers didn't clearly match anything" -- mirrors the
# app's low_confidence fallback message.
UNCLEAR_LABEL = 'unclear'
ALL_LABELS = CONDITION_IDS + [UNCLEAR_LABEL]

FEATURE_NAMES = SYMPTOM_IDS + QUESTION_IDS
UNKNOWN = 0.5  # sentinel used for a question that wasn't asked
