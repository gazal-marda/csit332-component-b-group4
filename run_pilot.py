import sys
import time

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/synthetic_pilot.csv"
OUT_PATH = "results.csv"
DATASET_ID = "synthetic_pilot"
PROTOCOL_ID = "cv5"
SAMPLE_SIZES = [50, 100, 200, 500]
N_REPLICATES = 100
N_FOLDS = 5
FEATURES = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]
COLUMNS = ["dataset_id", "protocol_id", "n_sub", "replicate_id", "classifier_id", "fold_id", "auc"]


def get_classifiers():
    return {
        "logreg": LogisticRegression(C=1.0, max_iter=1000, random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "dtree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=0),
        "rf": RandomForestClassifier(n_estimators=100, random_state=0),
        "gb": HistGradientBoostingClassifier(max_iter=100, random_state=0),
    }


def make_seed(n_sub, replicate_id):
    return n_sub * 10000 + replicate_id


def draw_sample(data, n_sub, replicate_id):
    seed = make_seed(n_sub, replicate_id)
    sample, _ = train_test_split(data, train_size=n_sub, stratify=data["y"], random_state=seed)
    return sample.reset_index(drop=True)


def make_splits(y, n_sub, replicate_id):
    seed = make_seed(n_sub, replicate_id)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(y, y))


def run_replicate(data, n_sub, replicate_id):
    sample = draw_sample(data, n_sub, replicate_id)
    X = sample[FEATURES].to_numpy()
    y = sample["y"].to_numpy()
    splits = make_splits(y, n_sub, replicate_id)
    base_models = get_classifiers()
    rows = []
    for classifier_id, base_model in base_models.items():
        for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
            model = clone(base_model)
            model.fit(X[train_idx], y[train_idx])
            scores = model.predict_proba(X[test_idx])[:, 1]
            auc = roc_auc_score(y[test_idx], scores)
            rows.append([DATASET_ID, PROTOCOL_ID, n_sub, replicate_id, classifier_id, fold_id, auc])
    return rows


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    data = pd.read_csv(data_path)
    print("loaded", data.shape[0], "rows from", data_path, flush=True)

    all_rows = []
    start = time.time()
    for n_sub in SAMPLE_SIZES:
        for replicate_id in range(1, N_REPLICATES + 1):
            all_rows.extend(run_replicate(data, n_sub, replicate_id))
            if replicate_id % 20 == 0:
                elapsed = time.time() - start
                print(f"n_sub={n_sub}  replicate {replicate_id}/{N_REPLICATES}  {elapsed:.0f}s elapsed", flush=True)

    results = pd.DataFrame(all_rows, columns=COLUMNS)
    results.to_csv(OUT_PATH, index=False)
    total = time.time() - start
    print(f"wrote {len(results)} rows to {OUT_PATH}")
    print(f"total time: {total:.0f} seconds ({total / 60:.1f} minutes)")


if __name__ == "__main__":
    main()
