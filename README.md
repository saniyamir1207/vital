# ASHA CARE — symptom-to-condition classifier

A scikit-learn model that replaces the fixed weighted-sum scoring in
`index.html` (`computeResults()`) with a trained probabilistic
classifier, using the same 11 outcome classes (10 conditions +
"unclear") and the same 20 symptoms / 17 yes-no questions already
defined in the app.

## Files

| File | Purpose |
|---|---|
| `definitions.py` | Symptom/question/condition schema, copied from `index.html` so the model and the UI agree on vocabulary. |
| `data_gen.py` | Builds a **synthetic** training set from that schema (see "About the data" below). |
| `train_model.py` | Trains and evaluates Logistic Regression vs. Random Forest, picks the better one by cross-validated macro-F1, saves it. |
| `predict.py` | Example of calling the saved model the way the app would — partial symptoms + only the questions actually asked. |
| `data/symptom_dataset.csv` | Generated dataset (7,900 rows). |
| `model/condition_classifier.joblib` | Trained classifier. |
| `model/label_encoder.joblib` | Maps class indices back to condition names. |
| `model/metadata.json` | Feature order, class list, and evaluation scores, for anything that loads the model later. |

## Run it yourself

```bash
pip install -r requirements.txt
python3 data_gen.py      # regenerate data/symptom_dataset.csv
python3 train_model.py   # train, evaluate, save model/*.joblib
python3 predict.py       # example prediction on a sample case
```

## Results

Logistic Regression was selected over Random Forest (5-fold CV
macro-F1 0.763 vs 0.743). On a held-out 20% test split:

- **Accuracy: 76%**, **macro-F1: 0.76** across 11 classes
- Weakest class is `unclear` (F1 0.65) — expected, since by
  construction it's pure noise with no symptom signature to learn
- Strongest classes are `uti` and `asthma` (F1 0.82–0.84), which have
  the most distinctive, high-weight symptoms in the original schema

Full classification report and confusion matrix are printed by
`train_model.py` and are worth re-reading before trusting this
anywhere real.

## About the data — please read before using this for real

**There is no real patient dataset in this project.** `data_gen.py`
manufactures training examples straight from the symptom weights
already in `CONDITIONS` (e.g. `dengue: {fever:2, jointPain:2, ...}`):
a simulated "dengue" patient gets `fever` and `jointPain` switched on
with high probability, unrelated symptoms with a low background rate,
and 8% of labels get scrambled at random so the model can't just
memorize a lookup table.

That means the model is, at best, **a smoothed, probabilistic
reproduction of the same hand-written rules the app already had** —
it will not know anything about symptom co-occurrence, demographics,
comorbidities, or disease prevalence that isn't already implicit in
those hand-picked weights. Concretely:
- It cannot be more medically accurate than the rule table it was
  trained to imitate.
- Its confidence scores reflect how the synthetic generator was
  written, not real diagnostic uncertainty.
- Feature importances / probabilities should be read as "what the
  schema encodes," not "what real patients look like."

If you want a model that adds real value over the current rule-based
scoring, you'd need actual (properly licensed, de-identified) symptom
checker or triage data to train on instead of `data_gen.py`'s output —
the rest of this pipeline (train/eval/predict) would work unchanged,
you'd just point `train_model.py` at a real CSV with the same column
schema.

## Wiring it into the app

This is intentionally kept outside the browser (Python service, not
JS), per your choice. To use it from `index.html` you'd need a small
backend endpoint (e.g. Flask/FastAPI) that:
1. loads `model/condition_classifier.joblib` and `label_encoder.joblib` once at startup,
2. accepts the same `{selectedSymptoms, answers}` shape the app already builds,
3. calls `predict.build_feature_vector` + `model.predict_proba`,
4. returns the top-3 `{cid, pct}` pairs in the same shape `computeResults()` currently returns, so `renderResults()` needs no changes.

I can build that Flask endpoint next if you want the two connected.

## Disclaimer

Same as the app itself: this is an informational demo, not a
diagnostic tool, and shouldn't be used to make real medical decisions.
