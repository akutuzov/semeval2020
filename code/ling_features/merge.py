# python3
# coding: utf-8

import argparse
import logging
import numpy as np

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input1", "-i1", help="Path to a tsv file 1", required=True)
    arg("--input2", "-i2", help="Path to a tsv file 2", required=True)
    args = parser.parse_args()

    words1 = {}
    words2 = {}

    for line in open(args.input1, "r"):
        word, val = line.strip().split("\t")
        words1[word] = float(val)

    for line in open(args.input2, "r"):
        word, val = line.strip().split("\t")
        words2[word] = float(val)

    assert words1.keys() == words2.keys()

    for word in words1:
        average = np.mean([words1[word], words2[word]])
        if "binary" in args.input1:
            average = int(average)
        print(f"{word}\t{average}")
