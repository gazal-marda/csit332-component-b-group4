import os

import numpy as np
import pandas as pd

from run_pilot import (COLUMNS, DATA_PATH, N_FOLDS, N_REPLICATES, OUT_PATH,
                       SAMPLE_SIZES, draw_sample, get_classifiers, make_splits,
                       run_replicate)


def check_format(results):
    problems = []
    expected_rows = len(SAMPLE_SIZES) * N_REPLICATES * len(get_classifiers()) * N_FOLDS
    if list(results.columns) != COLUMNS:
        problems.append("column names or order are wrong")
    if len(results) != expected_rows:
        problems.append(f"expected {expected_rows} rows but found {len(results)}")
    if results.isna().any().any():
        problems.append("there are missing values")
    if not results["auc"].between(0, 1).all():
        problems.append("some auc values are outside 0 to 1")
    if set(results["dataset_id"]) != {"synthetic_pilot"}:
        problems.append("dataset_id is not synthetic_pilot everywhere")
    if set(results["protocol_id"]) != {"cv5"}:
        problems.append("protocol_id is not cv5 everywhere")
    if set(results["classifier_id"]) != set(get_classifiers()):
        problems.append("classifier ids do not match the spec")
    if sorted(results["n_sub"].unique()) != SAMPLE_SIZES:
        problems.append("sample sizes are wrong")
    if sorted(results["replicate_id"].unique()) != list(range(1, N_REPLICATES + 1)):
        problems.append("replicate ids are wrong")
    if sorted(results["fold_id"].unique()) != list(range(1, N_FOLDS + 1)):
        problems.append("fold ids are wrong")
    counts = results.groupby(["n_sub", "replicate_id", "classifier_id"]).size()
    if (counts != N_FOLDS).any():
        problems.append("some classifier has a number of folds other than 5")
    if results.duplicated(["n_sub", "replicate_id", "classifier_id", "fold_id"]).any():
        problems.append("duplicate rows found")
    return problems


def check_samples(data):
    problems = []
    for n_sub in SAMPLE_SIZES:
        first = draw_sample(data, n_sub, 1)
        again = draw_sample(data, n_sub, 1)
        other = draw_sample(data, n_sub, 2)
        if len(first) != n_sub:
            problems.append(f"sample for n_sub={n_sub} has {len(first)} rows")
        if not first.equals(again):
            problems.append(f"same seed gave different rows for n_sub={n_sub}")
        if first.equals(other):
            problems.append(f"different replicates gave the same rows for n_sub={n_sub}")
    return problems


def check_folds(data):
    problems = []
    smallest_positives = {}
    for n_sub in SAMPLE_SIZES:
        smallest = n_sub
        for replicate_id in range(1, N_REPLICATES + 1):
            y = draw_sample(data, n_sub, replicate_id)["y"].to_numpy()
            splits = make_splits(y, n_sub, replicate_id)
            all_test = np.concatenate([test for _, test in splits])
            if sorted(all_test) != list(range(n_sub)):
                problems.append(f"folds do not cover the sample once, n_sub={n_sub} rep={replicate_id}")
            for train, test in splits:
                if len(np.unique(y[test])) < 2 or len(np.unique(y[train])) < 2:
                    problems.append(f"a fold has one class only, n_sub={n_sub} rep={replicate_id}")
                smallest = min(smallest, int(y[test].sum()))
        smallest_positives[n_sub] = smallest
    return problems, smallest_positives


def check_reproducible(data, results):
    problems = []
    for n_sub, replicate_id in [(50, 1), (100, 37), (200, 64), (500, 100)]:
        again = pd.DataFrame(run_replicate(data, n_sub, replicate_id), columns=COLUMNS)
        stored = results[(results["n_sub"] == n_sub) & (results["replicate_id"] == replicate_id)]
        stored = stored.reset_index(drop=True)
        if not np.allclose(again["auc"].to_numpy(), stored["auc"].to_numpy(), rtol=0, atol=1e-12):
            problems.append(f"rerun of n_sub={n_sub} rep={replicate_id} does not match results.csv")
    return problems


def make_summary(results):
    rep = (results.groupby(["n_sub", "replicate_id", "classifier_id"])["auc"]
           .mean().reset_index(name="rep_auc"))
    summary = (rep.groupby(["n_sub", "classifier_id"])["rep_auc"]
               .agg(mean_auc="mean", sd_across_reps="std", min_auc="min", max_auc="max")
               .reset_index())
    order = list(get_classifiers())
    summary["classifier_id"] = pd.Categorical(summary["classifier_id"], categories=order, ordered=True)
    summary = summary.sort_values(["n_sub", "classifier_id"]).reset_index(drop=True)
    rep["is_top"] = rep.groupby(["n_sub", "replicate_id"])["rep_auc"].transform(
        lambda s: np.isclose(s, s.max(), rtol=0, atol=1e-12))
    win_counts = rep.groupby(["n_sub", "classifier_id"], observed=True)["is_top"].sum().reset_index(name="times_top")
    win_counts["classifier_id"] = win_counts["classifier_id"].astype(str)
    summary["classifier_id"] = summary["classifier_id"].astype(str)
    summary = summary.merge(win_counts, on=["n_sub", "classifier_id"], how="left")
    summary["times_top"] = summary["times_top"].astype(int)
    return summary, rep


def report(name, problems):
    if problems:
        print(f"FAIL  {name}")
        for p in problems[:5]:
            print("      ", p)
    else:
        print(f"ok    {name}")


def main():
    results = pd.read_csv(OUT_PATH)
    report("format of results.csv", check_format(results))

    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        report("same seed gives same sample", check_samples(data))
        fold_problems, smallest = check_folds(data)
        report("every fold has both classes", fold_problems)
        print("      fewest positives in any test fold:", smallest)
        report("rerun matches stored results", check_reproducible(data, results))
    else:
        print("data file not found, skipped the sample, fold and rerun checks")

    summary, rep = make_summary(results)
    summary.to_csv("summary_by_classifier.csv", index=False, float_format="%.4f")
    print()
    print(summary.round(4).to_string(index=False))

    tops = rep.groupby(["n_sub", "replicate_id"])["is_top"].sum()
    tied = tops[tops > 1]
    print()
    print("times_top counts every classifier that shares the top score, so a column can add up to more than 100")
    print("replicates where the top mean auc was tied:", len(tied))
    print(tied.reset_index().groupby("n_sub").size().to_string())


if __name__ == "__main__":
    main()
