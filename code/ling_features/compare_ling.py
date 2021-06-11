# python3
# coding: utf-8

import argparse
import logging
from smart_open import open
import json
from helpers import *
import csv

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
        "--filtering",
        "-f",
        nargs="?",
        const=0,
        help="Minimal percentage to keep a feature",
        default=0,
        type=int,
        required=False,
    )
    arg(
        "--syntax_filtering",
        "-sf",
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
        choices=["yes", "no", "2step"],
        default="no",
    )
    arg(
        "--aggregate",
        "-a",
        help="If separation is '2step', distances are computed separately "
             "and then aggregated using either maximum or average",
        choices=["max", "avg"],
        default="max",
    )
    arg(
        "--added_features1",
        "-af1",
        help="Path to JSON file to add syntax features when separation is 2step",
    )
    arg(
        "--added_features2",
        "-af2",
        help="Path to JSON file to add syntax features when separation is 2step",
    )

    arg(
        "--changepoint",
        "-cp",
        help="How to detect the change point in distance sequences?",
        choices=["semeval", "half", "automatic"],
        default="automatic",
    )
    arg(
        "--store_distances",
        "-sd",
        help="Save categories and distances to file",
        required=False,
        default="distances.json"
    )

    args = parser.parse_args()

    properties_1 = json.load(open(args.input1, "r"))
    properties_2 = json.load(open(args.input2, "r"))

    if args.added_features1:
        added_features1 = json.load(open(args.added_features1))
        added_features2 = json.load(open(args.added_features2))

    feature_to_group = {}
    if args.syntax_filtering:
        groups = json.load(open("../../data/features/synt_groups.json", "r"))
        for k, v in groups.items():
            for f in v:
                feature_to_group[f] = k

    assert properties_1.keys() == properties_2.keys()
    words = {w: 0 for w in properties_1.keys()}

    word_distances = {}

    for word in words:
        if args.separation == "2step":

            distance = {}

            p1 = collect_word_properties(properties_1[word])
            p2 = collect_word_properties(properties_2[word])

            feature_classes = list(p1.keys() | p2.keys())

            for f_class in feature_classes:
                distance[f_class] = \
                    compute_distance_from_common_features(p1[f_class],
                                                          p2[f_class],
                                                          args.filtering,
                                                          args.distance)

            if args.added_features1:
                a1 = added_features1[word]
                a2 = added_features2[word]
                distance["syntax"] = \
                    compute_distance_from_common_features(a1,
                                                          a2,
                                                          args.filtering,
                                                          args.distance)
            if args.store_distances:
                word_distances[word] = distance
            distance = [d for d in distance.values() if not np.isnan(d)]

            if distance:
                if args.aggregate == "max":
                    words[word] = max(distance)
                elif args.aggregate == "avg":
                    words[word] = np.mean(distance)
            else:
                # empty, e.g. german
                words[word] = np.nan

        else:
            if args.separation == "yes":
                p1 = feature_separation(properties_1[word])
                p2 = feature_separation(properties_2[word])

            else:
                p1 = properties_1[word]
                p2 = properties_2[word]

            if args.syntax_filtering != "none":
                p1 = synt_group(p1, args.syntax_filtering, feature_to_group)
                p2 = synt_group(p2, args.syntax_filtering, feature_to_group)

            distance = compute_distance_from_common_features(p1,
                                                             p2,
                                                             args.filtering,
                                                             args.distance)
            if np.isnan(distance):
                distance = 0.0  # A word was not present in one of the time periods
            words[word] = distance

    if args.store_distances:
        categories = set()
        for word in word_distances:
            categories.update(word_distances[word].keys())
        categories = list(categories)
        with open(args.store_distances, 'w') as tsvfile:
            catwriter = csv.writer(tsvfile, delimiter='\t', dialect="unix",
                                   quoting=csv.QUOTE_MINIMAL)
            catwriter.writerow(["Word"] + [c for c in categories])
            for word in word_distances:
                output = [word]
                for cat in categories:
                    if cat in word_distances[word]:
                        output.append(word_distances[word][cat])
                    else:
                        output.append(0.0)
                catwriter.writerow(output)

    if args.output:
        print_results(words, args.output, args.changepoint)
    else:
        logger.info(words)
