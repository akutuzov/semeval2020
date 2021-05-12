# python3
# coding: utf-8

import argparse
import logging
from smart_open import open
import json
import numpy as np
from scipy.spatial.distance import cosine

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input1', '-i1', help='Path to a JSON file 1', required=True)
    arg('--input2', '-i2', help='Path to a JSON file 2', required=True)
    arg('--output', '-o', help='Output path (tsv)', required=False)

    args = parser.parse_args()

    with open(args.input1, "r") as f:
        properties_1 = json.loads(f.read())

    with open(args.input2, "r") as f:
        properties_2 = json.loads(f.read())

    assert properties_1.keys() == properties_2.keys()

    words = {w: 0 for w in properties_1.keys()}

    for word in words:
        logger.info(word)
        features = list(properties_1[word].keys() | properties_2[word].keys())
        vector_1 = np.zeros(len(features))
        vector_2 = np.zeros(len(features))
        for nr, feature in enumerate(features):
            try:
                vector_1[nr] = properties_1[word][feature]
            except KeyError:
                pass
            try:
                vector_2[nr] = properties_2[word][feature]
            except KeyError:
                pass
        distance = cosine(vector_1, vector_2)
        words[word] = distance

    with open(f"{args.output}_graded.tsv", "w") as f:
        for w in words:
            f.write(f"{w}\t{words[w]}\n")

    with open(f"{args.output}_binary.tsv", "w") as f:
        values = sorted(words, key=words.get, reverse=True)
        # threshold = int(len(values) / 2)
        threshold = 17
        for val in values[:threshold]:
            f.write(f"{val}\t1\n")
        for val in values[threshold:]:
            f.write(f"{val}\t0\n")
