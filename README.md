# CSIT332 Semester Project - Component B (The Runs)

**Group 4**

Contributors: Gazal, Brij, Heena

This is our Week 1 submission for Component B. The task was to build the experiment runner and run it on the pilot dataset (`synthetic_pilot.csv`, 20,000 rows, features f1 to f10 and a binary label y with about 20% positives).

For each sample size (50, 100, 200, 500) we do 100 replicates. In each replicate we draw a stratified random sample of that size, run stratified 5-fold cross-validation, and record the AUC of the six classifiers on every fold. That gives 4 x 100 x 6 x 5 = 12,000 rows in `results.csv`.

## Files

| File | What it is |
|---|---|
| `run_pilot.py` | the experiment runner, produces `results.csv` |
| `results.csv` | 12,000 rows: dataset_id, protocol_id, n_sub, replicate_id, classifier_id, fold_id, auc |
| `check_results.py` | checks the format of `results.csv`, reruns a few replicates to confirm they match, and prints the summary table |
| `summary_by_classifier.csv` | mean, spread and min/max of AUC per classifier and sample size (made by `check_results.py`) |
| `Component_B_Group4_note.pdf` | the short note: run time, what surprised us, and our list of ambiguities |
| `run_log.txt` | console output from the full run |
| `check_output.txt` | console output from `check_results.py` |
| `requirements.txt` | python packages needed |
| `data/` | put `synthetic_pilot.csv` here (not committed, download it from the LMS) |

## How to run

```
pip install -r requirements.txt
python run_pilot.py
python check_results.py
```

`run_pilot.py` reads `data/synthetic_pilot.csv` by default. You can also give it a different path, for example `python run_pilot.py path/to/file.csv`. The full run takes a few minutes.

We used Python 3.12.3, numpy 2.4.4, pandas 3.0.2 and scikit-learn 1.8.0.

## Classifiers

| classifier_id | model | settings |
|---|---|---|
| logreg | LogisticRegression | C=1.0, max_iter=1000, random_state=0 |
| knn | KNeighborsClassifier | n_neighbors=5 |
| dtree | DecisionTreeClassifier | max_depth=5, random_state=0 |
| svm | SVC | kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=0 |
| rf | RandomForestClassifier | n_estimators=100, random_state=0 |
| gb | HistGradientBoostingClassifier | max_iter=100, random_state=0 |

## Choices we made where the instructions did not say

- The seed for a replicate is `n_sub * 10000 + replicate_id`. The same seed is used for drawing the sample and for shuffling the folds.
- `replicate_id` goes from 1 to 100 and `fold_id` from 1 to 5.
- The sample is drawn with `train_test_split(..., stratify=y)`, so the share of positives stays close to the 20% in the full data.
- Folds come from `StratifiedKFold` with `shuffle=True`.
- A fresh copy of each classifier is fitted on every fold.
- AUC is computed from `predict_proba(...)[:, 1]` for all six classifiers, including the SVM.
- AUC values are saved without rounding.
- Rows in `results.csv` are ordered by n_sub, replicate, classifier, then fold.

More detail on these is in the note.
