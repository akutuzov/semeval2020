# python3
# coding: utf-8

import argparse
import logging
from smart_open import open
import json
import numpy as np

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input1", "-i1", help="Path to a JSON file 1", required=True)
    arg("--input2", "-i2", help="Path to a JSON file 2", required=True)
    arg("--output", "-o", help="Output path (tsv)", required=False)
    arg("--threshold", "-t", nargs='?', const=0, help="Minimal percentage to keep a feature", default=0, type=int, required=False)

    args = parser.parse_args()

    with open(args.input1, "r") as f:
        properties_1 = json.loads(f.read())

    with open(args.input2, "r") as f:
        properties_2 = json.loads(f.read())

    assert properties_1.keys() == properties_2.keys()

    buffer = ''
    for word in properties_1.keys():
        logger.info(word)
        buffer += '> {}\n'.format(word)

        p1 = properties_1[word]
        p2 = properties_2[word]
        features = list(p1.keys() | p2.keys())

        prop_count = {k: p1.get(k, 0) + p2.get(k, 0) for k in features}
        total = sum(prop_count.values())
        features = [f for f in features if prop_count[f] / total * 100 > args.threshold]

        vector_1 = np.zeros(len(features))
        vector_2 = np.zeros(len(features))
        feat2idx = {}
        for nr, feature in enumerate(features):
            feat2idx[feature] = nr
            try:
                vector_1[nr] = properties_1[word][feature]
            except KeyError:
                pass
            try:
                vector_2[nr] = properties_2[word][feature]
            except KeyError:
                pass

        vector_1 /= vector_1.sum()
        vector_2 /= vector_2.sum()
        changes = {}
        for nr, feature in enumerate(features):
            changes[feature] = abs(vector_2[nr] - vector_1[nr])

        ordered_features = [k for k, v in sorted(changes.items(), key=lambda item: item[1], reverse=True)]

        for feature in ordered_features:
            # logger.info('{}: {:.2f} -> {:.2f}'.format(
            #     feature, vector_1[feat2idx[feature]] * 100, vector_2[feat2idx[feature]] * 100))
            buffer += '{}: {:.2f} -> {:.2f}\n'.format(
                feature, vector_1[feat2idx[feature]] * 100, vector_2[feat2idx[feature]] * 100)

        buffer += '\n'

        with open(args.output, "w") as f:
            f.write(buffer)