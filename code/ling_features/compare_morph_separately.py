# python3
# coding: utf-8

import argparse
import logging
from smart_open import open
import json
import numpy as np
from scipy.spatial.distance import cosine, jensenshannon
from collections import defaultdict


def collect_word_properties(properties):
    props = defaultdict(lambda: defaultdict(int))
    for features, count in properties.items():
        separate_features = features.split("|")
        for feat in separate_features:
            try:
                k, v = feat.split("=")
            except:
                continue
            else:
                props[k][v] += count
    return props
    

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
    arg(
        "--threshold",
        "-t",
        nargs="?",
        const=0,
        help="Minimal percentage to keep a feature",
        default=0,
        type=int,
        required=False,
    )
    arg(
        "--distance",
        "-d",
        help="Choose between cosine and jsd",
        choices=["cos", "jsd"],
        default="cos",
    )
    arg(
        "--agregate",
        "-a",
        help="How to agregate distances across features",
        choices=["max", "avg"],
        default="max",
    )
    
    
    args = parser.parse_args()


    with open(args.input1, "r") as f:
        properties_1 = json.loads(f.read())

    with open(args.input2, "r") as f:
        properties_2 = json.loads(f.read())

    assert properties_1.keys() == properties_2.keys()
    words = {w: 0 for w in properties_1.keys()}

    all_features = {}

    for word in words:
        distance = {}
        
        p1 = collect_word_properties(properties_1[word])
        p2 = collect_word_properties(properties_2[word])
        
        feature_classes = list(p1.keys() | p2.keys())
        
        for f_class in feature_classes:
            features = list(p1[f_class].keys() | p2[f_class].keys())

            prop_count = {k: p1[f_class][k] + p2[f_class][k] for k in features}
            total = sum(prop_count.values())
            features = [f for f in features if prop_count[f] / total * 100 > args.threshold]
            
            vector_1 = np.zeros(len(features))
            vector_2 = np.zeros(len(features))

            
            
            for nr, feature in enumerate(features):
                vector_1[nr] = p1[f_class][feature]
                vector_2[nr] = p2[f_class][feature]

            if args.distance == "cos":
                distance[f_class] = cosine(vector_1, vector_2)
            elif args.distance == "jsd":
                distance[f_class] = jensenshannon(vector_1, vector_2)
            else:
                raise NotImplementedError("Unknown distance: %s" % args.distance)

        distance = [d for d in distance.values() if not np.isnan(d)]

        if distance:
            if args.agregate == "max":
                words[word] = max(distance)
            elif args.agregate == "avg":
                words[word] = np.mean(distance)
        else:
            # empty, e.g. german cos, max
            words[word] = np.nan
            
    if args.output:
        with open(f"{args.output}_graded.tsv", "w") as f:
            for w in words:
                f.write(f"{w}\t{words[w]}\n")

        with open(f"{args.output}_binary.tsv", "w") as f:
            values = sorted(words, key=words.get, reverse=True)
            threshold = int(len(values) * 0.43)
            for val in values[:threshold]:
                f.write(f"{val}\t1\n")
            for val in values[threshold:]:
                f.write(f"{val}\t0\n")
    else:
        logger.info(words)
