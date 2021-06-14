# python3
# coding: utf-8

import argparse
import os
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
from helpers import cat_plot

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--lang", "-l", help="Language to test", required=True)

    args = parser.parse_args()

    inputfile = f"distances/{args.lang}.tsv"
    gold1_file = f"gold/task1/{args.lang}.txt"
    gold2_file = f"gold/task2/{args.lang}.txt"

    features = pd.read_csv(inputfile, delimiter="\t")
    feature_names = features.columns[1:-1].values

    if os.path.exists(gold2_file):
        gold2 = pd.read_csv(gold2_file, delimiter="\t", names=["Word", "Distance"])
        logger.info("Categories with significant correlations (task 2)")
        features["Distance"] = gold2["Distance"]
        corrs = []
        for category in feature_names:
            corr = spearmanr(features[category].values, features["Distance"].values)
            corrs.append(corr)
        top_corr = sorted(
            zip(corrs, feature_names), key=lambda i: i[0], reverse=True
        )
        for corr, category in top_corr:
            if corr[1] <= 0.06:
                logger.info(f"{category}: {corr[0]:.3f} ({corr[1]:.3f})")

    if not os.path.exists(gold1_file):
        raise SystemExit("No gold file for subtask 1 found!")
    gold = pd.read_csv(gold1_file, delimiter="\t", names=["Word", "Class"])

    features["Class"] = gold["Class"]

    logger.info(features.head())

    X = features[feature_names].values
    y = features.Class.values

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    logger.info("Fitting logistic regression to the data...")
    clf = LogisticRegression(
        penalty="l2", class_weight="balanced", solver="liblinear"
    ).fit(X, y)
    logger.info("Evaluating...")
    logger.info("Predictions on the train data:")
    predictions = clf.predict(X)
    logger.info(predictions)
    accuracy = clf.score(X, y)
    logger.info(f"Accuracy on the train data: {accuracy:.3f}")
    f1 = f1_score(y, predictions, average="macro")
    logger.info(f"Macro F1 on the train data: {f1:.3f}")

    logger.info("Feature importance (task 1):")
    importance = sorted(
        zip(clf.coef_[0], feature_names), key=lambda i: i[0], reverse=True
    )
    for coeff, category in importance:
        logger.info(f"{category}: {coeff:.3f}")

    # _ = cat_plot([el[0] for el in importance if el[0] != 0],
    #             [el[1] for el in importance if el[0] != 0])

