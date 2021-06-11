# python3
# coding: utf-8

import argparse
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input", "-i", help="Path to a category distances file (tsv)", required=True)
    arg("--gold", "-g", help="Path to gold binary values", required=True)

    args = parser.parse_args()

    features = pd.read_csv(args.input, delimiter="\t")
    gold = pd.read_csv(args.gold, delimiter="\t", names=["Word", "Class"])

    features["Class"] = gold["Class"]

    print(features.head())

    X = features[features.columns[1:-1]].values
    y = features.Class.values

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    feature_names = features.columns[1:-1].values

    logger.info("Fitting logistic regression to the data...")
    clf = LogisticRegression(
        penalty="l2", class_weight="balanced", solver="liblinear"
    ).fit(X, y)
    logger.info("Evaluating...")
    logger.info("Predictions on the train data:")
    logger.info(clf.predict(X))
    accuracy = clf.score(X, y)
    logger.info(f"Accuracy on the train data: {accuracy:.3f}")

    logger.info("Feature importance:")
    importance = sorted(
        zip(clf.coef_[0], feature_names), key=lambda i: i[0], reverse=True
    )
    for coeff, category in importance:
        logger.info(f"{category}: {coeff:.3f}")
