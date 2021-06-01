# python3
# coding: utf-8

import argparse
import logging
from smart_open import open
import json
import numpy as np
from scipy.spatial.distance import cosine, jensenshannon
from collections import defaultdict

informative = [
    "nominal_function",
    "function",
    "modifier",
    "nominal_modifier",
    "core_nominals",
    "nominal_dependents",
]

groups = json.load(open("../../data/features/synt_groups.json", "r"))
feature_to_group = {}
for k, v in groups.items():
    for f in v:
        feature_to_group[f] = k


def synt_group(properties, filtering):
    new_properties = defaultdict(int)
    for current_feature, count in properties.items():
        group = feature_to_group[current_feature.split(":")[0]]
        if filtering == "group":
            new_properties[group] += count
        elif filtering == "partial":
            group = feature_to_group[current_feature.split(":")[0]]
            if group in informative:
                new_properties[current_feature] = count
            else:
                new_properties[group] += count
        elif filtering == "delete":
            if group in informative:
                new_properties[current_feature] = count
        else:
            raise NotImplementedError
    return new_properties


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
        "--filtering",
        "-f",
        help="Organizing syntactic features according to UD classification: "
        "'group' - grouping all, 'delete' - deleting non-informative, "
        "'partial' - grouping non-informative",
        choices=["group", "partial", "delete", "none"],
        default="none",
    )
    arg(
        "--distance",
        "-d",
        help="Choose between cosine and jsd",
        choices=["cos", "jsd"],
        default="cos",
    )
    arg(
        "--separation",
        "-s",
        help="Morphological feature separation by |",
        choices=["yes", "no"],
        default="no",
    )

    args = parser.parse_args()

    with open(args.input1, "r") as f:
        properties_1 = json.loads(f.read())

    with open(args.input2, "r") as f:
        properties_2 = json.loads(f.read())

    assert properties_1.keys() == properties_2.keys()
    words = {w: 0 for w in properties_1.keys()}

    all_features = set()

    for word in words:
        if args.separation == "yes":
            p1 = defaultdict(int)
            p2 = defaultdict(int)
            for el in properties_1[word]:
                if "|" in el:
                    separate_features = el.split("|")
                    for feat in separate_features:
                        p1[feat] += properties_1[word][el]
                else:
                    p1[el] += properties_1[word][el]
            for el in properties_2[word]:
                if "|" in el:
                    separate_features = el.split("|")
                    for feat in separate_features:
                        p2[feat] += properties_2[word][el]
                else:
                    p2[el] += properties_2[word][el]

        else:
            p1 = properties_1[word]
            p2 = properties_2[word]

        if args.filtering != "none":
            p1 = synt_group(p1, args.filtering)
            p2 = synt_group(p2, args.filtering)

        features = list(p1.keys() | p2.keys())

        prop_count = {k: p1.get(k, 0) + p2.get(k, 0) for k in features}
        total = sum(prop_count.values())
        features = [f for f in features if prop_count[f] / total * 100 > args.threshold]

        all_features.update(features)

        vector_1 = np.zeros(len(features))
        vector_2 = np.zeros(len(features))

        for nr, feature in enumerate(features):
            try:
                vector_1[nr] = p1[feature]
            except KeyError:
                pass
            try:
                vector_2[nr] = p2[feature]
            except KeyError:
                pass

        if args.distance == "cos":
            distance = cosine(vector_1, vector_2)
        elif args.distance == "jsd":
            distance = jensenshannon(vector_1, vector_2)
        else:
            raise NotImplementedError("Unknown distance: %s" % args.distance)

        if np.isnan(distance):
            distance = 0.0  # A word was not present in one of the time periods
        words[word] = distance

    if args.output:
        with open(f"{args.output}_graded.tsv", "w") as f:
            for w in words:
                f.write(f"{w}\t{words[w]}\n")

        with open(f"{args.output}_binary.tsv", "w") as f:
            values = sorted(words, key=words.get, reverse=True)
            # threshold = int(len(values) / 2)
            threshold = int(len(values) * 0.43)
            # threshold = 17  # This is for English
            for val in values[:threshold]:
                f.write(f"{val}\t1\n")
            for val in values[threshold:]:
                f.write(f"{val}\t0\n")
    else:
        logger.info(words)
