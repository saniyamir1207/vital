"""
Generate a synthetic training set for the ASHA CARE symptom checker.

There is no real patient dataset attached to this project, so this
script simulates one from the same symptom-weight structure already
encoded in the app (CONDITIONS in index.html / definitions.py). For
each simulated "patient":

  1. A true condition is chosen (or 'unclear' for a noisy, no-pattern case).
  2. Symptoms are switched on probabilistically: symptoms tied to the
     condition (and weighted more heavily) appear more often; a small
     background rate lets unrelated symptoms appear too (comorbidity /
     noise), the way real presentations overlap.
  3. Questions tied to the condition are answered "yes" more often;
     unrelated questions get a low base "yes" rate.
  4. A slice of labels are randomly reassigned (label noise) so the
     classifier can't just memorize a lookup table -- it has to learn
     a genuinely probabilistic mapping, same as the real problem.

This is clearly a synthetic proxy, not clinical data -- see README for
why that matters and what would be needed to make this a real model.
"""
import numpy as np
import pandas as pd

from definitions import (
    CONDITIONS, CONDITION_IDS, SYMPTOM_IDS, QUESTION_IDS,
    UNCLEAR_LABEL, FEATURE_NAMES,
)

RNG_SEED = 42
N_PER_CONDITION = 700
N_UNCLEAR = 900          # pure-noise cases with no dominant condition
LABEL_NOISE_RATE = 0.08  # fraction of rows whose label gets scrambled
BACKGROUND_SYMPTOM_RATE = 0.07
BACKGROUND_QUESTION_YES_RATE = 0.30


def _weight_to_prob(weight: int) -> float:
    # weight is 1, 2, or 3 in CONDITIONS -> presence probability
    return {1: 0.45, 2: 0.65, 3: 0.85}.get(weight, 0.5)


def _simulate_case(rng: np.random.Generator, condition: str | None) -> dict:
    row = {}
    cond_symptoms = CONDITIONS[condition]['symptoms'] if condition else {}
    cond_questions = set(CONDITIONS[condition]['questions']) if condition else set()

    for s in SYMPTOM_IDS:
        if s in cond_symptoms:
            p = _weight_to_prob(cond_symptoms[s])
        else:
            p = BACKGROUND_SYMPTOM_RATE
        row[s] = int(rng.random() < p)

    for q in QUESTION_IDS:
        p = 0.72 if q in cond_questions else BACKGROUND_QUESTION_YES_RATE
        row[q] = int(rng.random() < p)

    return row


def generate(n_per_condition=N_PER_CONDITION, n_unclear=N_UNCLEAR,
             label_noise_rate=LABEL_NOISE_RATE, seed=RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []

    for cid in CONDITION_IDS:
        for _ in range(n_per_condition):
            rows.append(_simulate_case(rng, cid))
            labels.append(cid)

    for _ in range(n_unclear):
        rows.append(_simulate_case(rng, None))
        labels.append(UNCLEAR_LABEL)

    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    df['label'] = labels

    # label noise: reassign a random subset of rows to a random label,
    # independent of their features, so the model has to learn
    # probabilities rather than a perfect deterministic map.
    n = len(df)
    noise_idx = rng.choice(n, size=int(n * label_noise_rate), replace=False)
    all_labels = CONDITION_IDS + [UNCLEAR_LABEL]
    df.loc[noise_idx, 'label'] = rng.choice(all_labels, size=len(noise_idx))

    # shuffle row order
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = generate()
    out_path = 'data/symptom_dataset.csv'
    df.to_csv(out_path, index=False)
    print(f'Wrote {len(df)} rows, {len(FEATURE_NAMES)} features -> {out_path}')
    print(df['label'].value_counts())
